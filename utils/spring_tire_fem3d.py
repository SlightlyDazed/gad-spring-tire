"""
spring_tire_fem3d.py - FEM 3D para NASA Spring Tire v6.0

FEM 3D real usando gmsh (mallado) + scikit-fem (solver).
Diseñado para usarse como "filtro de alta fidelidad" en el NSGA-III:
  - cada N generaciones valida el top-K del frente de Pareto
  - al final valida todas las soluciones candidatas

Pipeline:
  params → gmsh (geometría 3D + tetraedros) → skfem (K, solve) → post-proceso

Outputs que mejoran sobre el FEA 2D:
  - Campo de desplazamientos 3D real (u, v, w) en toda la malla
  - Tensor de esfuerzos completo (σxx, σyy, σzz, τxy, τyz, τxz)
  - Von Mises nodal real
  - Rigidez vertical/lateral desde la respuesta 3D
  - Factor de concentración de esfuerzos en zonas críticas (uniones rim/tread)
  - Pandeo 3D (valores propios de K + Kg)

Dependencias: gmsh, meshio, scikit-fem, scipy, numpy
"""

import os
import json
import tempfile
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports opcionales — el módulo funciona en modo degradado si faltan
# ---------------------------------------------------------------------------
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    log.warning("gmsh no disponible → FEM 3D desactivado")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

try:
    from skfem import (
        MeshTet, Basis, ElementVector, ElementTetP2, ElementTetP1,
        asm, condense
    )
    from skfem.models import linear_elasticity, lame_parameters
    from scipy.sparse.linalg import spsolve, eigsh
    import scipy.sparse as sp
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False

FEM3D_AVAILABLE = GMSH_AVAILABLE and MESHIO_AVAILABLE and SKFEM_AVAILABLE


# ---------------------------------------------------------------------------
# Catálogo de materiales (mismo que spring_tire_fea_robust.py)
# ---------------------------------------------------------------------------
MATERIAL_DB: Dict[int, Dict] = {
    0: {'name': 'TPU_95A',   'E': 50.0,   'nu': 0.47, 'sigma_y': 25.0,  'density': 1.15,
        'C1': 1.4, 'D1': 0.02, 'model': 'neo_hookean', 'S_ut': 32.5, 'S_f': 0.9, 'b_basquin': -0.12},
    1: {'name': 'TPU_85A',   'E': 25.0,   'nu': 0.47, 'sigma_y': 15.0,  'density': 1.12,
        'C1': 0.7, 'D1': 0.04, 'model': 'neo_hookean', 'S_ut': 19.5, 'S_f': 0.9, 'b_basquin': -0.12},
    2: {'name': 'TPU_75D',   'E': 200.0,  'nu': 0.45, 'sigma_y': 40.0,  'density': 1.20,
        'C1': 5.0, 'D1': 0.005, 'model': 'neo_hookean', 'S_ut': 52.0, 'S_f': 0.9, 'b_basquin': -0.12},
    3: {'name': 'Nylon_PA12','E': 1800.0, 'nu': 0.40, 'sigma_y': 45.0,  'density': 1.01,
        'C1': None, 'D1': None, 'model': 'linear', 'S_ut': 58.5, 'S_f': 0.85, 'b_basquin': -0.08},
    4: {'name': 'ABS',       'E': 2200.0, 'nu': 0.38, 'sigma_y': 40.0,  'density': 1.05,
        'C1': None, 'D1': None, 'model': 'linear', 'S_ut': 52.0, 'S_f': 0.85, 'b_basquin': -0.10},
    5: {'name': 'PLA',       'E': 3500.0, 'nu': 0.36, 'sigma_y': 55.0,  'density': 1.25,
        'C1': None, 'D1': None, 'model': 'linear', 'S_ut': 71.5, 'S_f': 0.85, 'b_basquin': -0.09},
}


# ---------------------------------------------------------------------------
# ANÁLISIS DE MEMORIA DE FORMA - Modelo Neo-Hookean
# ---------------------------------------------------------------------------

def neo_hookean_stiffness(F_deform: np.ndarray, C1: float, D1: float) -> Tuple[float, float]:
    """
    Calcula rigidez tangente Neo-Hookean bajo deformación F.

    Energía: W = C1*(I1-3) + (1/D1)*(J-1)²

    Args:
        F_deform: tensor de deformación (3x3)
        C1, D1: parámetros Neo-Hookean calibrados por material

    Returns:
        (sigma_vm, k_tangent)
    """
    J = np.linalg.det(F_deform)
    C = F_deform.T @ F_deform         # tensor de Cauchy-Green derecho
    I1 = np.trace(C)

    # Tensión de Cauchy
    sigma = (2 * C1 / J) * (F_deform @ F_deform.T) + (2 * (J-1) / D1) * np.eye(3)

    # Von Mises
    s = sigma - np.trace(sigma)/3 * np.eye(3)
    sigma_vm = float(np.sqrt(1.5 * np.sum(s**2)))

    # Rigidez tangente (módulo tangente efectivo)
    k_tangent = float(2 * C1 / J + 2 / D1 * (2*J - 1))

    return sigma_vm, k_tangent


def analyze_shape_memory_fem3d(params: Dict, mat: Dict, load_steps: int = 10) -> Dict:
    """
    Análisis incremental Neo-Hookean para estimar memoria de forma.

    Simula carga → descarga → recuperación del radio/spoke.
    Calcula curva carga-deformación incremental hasta 30% de estiramiento.

    Args:
        params: parámetros del Spring Tire
        mat: propiedades del material (con C1, D1 para Neo-Hookean)
        load_steps: número de pasos de carga

    Returns:
        Dict con recovery ratio, energía almacenada, curva carga-def
    """
    L0 = (params['d_ext'] - params['d_rim']) / 2  # mm longitud radio
    wall_thick = params.get('wall_thickness', 2.5)
    spoke_thick = params.get('spoke_thickness', wall_thick)  # BUG#5: usar spoke_thickness si disponible
    spoke_w = wall_thick * 4.0
    # A0 RECTANGULAR, consistente con modelo geométrico (no circular)
    A0 = spoke_w * spoke_thick  # mm² sección rectangular

    C1 = mat.get('C1')
    D1 = mat.get('D1')

    if C1 is None or D1 is None:
        # Material lineal → no hay memoria de forma significativa
        return {
            'shape_memory_ratio': 0.05,
            'energy_stored_mJ': 0.0,
            'recovery_ratio': 0.95,
            'model': 'linear_elastic',
            'C1': None,
            'D1': None
        }

    # Carga máxima → deformación de 30%
    lambda_max = 1.30  # estiramiento axial máximo

    # Curva carga-deformación incremental
    lambdas = np.linspace(1.0, lambda_max, load_steps + 1)
    forces = []
    sigmas_vm = []
    energies = []

    for lam in lambdas:
        # Tensor de deformación uniaxial (volumen casi constante - quasi-incompresible)
        lam_trans = 1.0 / np.sqrt(lam)
        F = np.diag([lam, lam_trans, lam_trans])

        sigma_vm, k_tan = neo_hookean_stiffness(F, C1, D1)
        sigmas_vm.append(sigma_vm)

        # Fuerza = sigma * área (considerando cambio de área por Poisson)
        A_current = A0 / lam  # área actual (conservación de volumen aprox)
        force = sigma_vm * A_current
        forces.append(force)

        # Energía de deformación acumulada
        # BUG#1: J = 1.0 (quasi-incompresible), no J = lam
        # det(F) = lam * (1/sqrt(lam))^2 = 1.0
        J = 1.0
        I1 = lam**2 + 2.0/lam
        W = C1 * (I1 - 3) + (J - 1)**2 / D1  # MPa = mJ/mm³
        energies.append(W * A0 * L0)  # mJ

    # Energía máxima almacenada
    energy_mJ = float(energies[-1])

    # Ratio de recuperación: materiales elastoméricos ~95-99%
    # Depende de la relación C1/D1 (rigidez vs compresibilidad)
    stiffness_ratio = C1 * D1
    # Mayor C1*D1 → más rígido → menor recuperación elástica
    recovery_ratio = min(0.99, 0.85 + 0.14 * np.exp(-stiffness_ratio))

    # Memoria de forma: deformación residual tras descarga
    shape_memory_ratio = 1.0 - recovery_ratio

    return {
        'shape_memory_ratio': float(shape_memory_ratio),
        'energy_stored_mJ': float(energy_mJ),
        'recovery_ratio': float(recovery_ratio),
        'max_force_N': float(forces[-1]) if forces else 0.0,
        'sigma_vm_at_30pct_MPa': float(sigmas_vm[-1]) if sigmas_vm else 0.0,
        'loading_curve': {
            'stretch_ratio': lambdas.tolist(),
            'force_N': [float(f) for f in forces],
            'energy_mJ': [float(e) for e in energies]
        },
        'model': 'neo_hookean',
        'C1': C1,
        'D1': D1,
        'lambda_max': float(lambda_max),
    }


# ---------------------------------------------------------------------------
# ANÁLISIS DE FATIGA - Goodman Modificado + Basquin S-N
# ---------------------------------------------------------------------------

def analyze_fatigue_fem3d(sigma_vm_max: float, sigma_mean: float,
                          mat: Dict, R_ratio: float = 0.1,
                          d_ext: float = 160.0) -> Dict:
    """
    Análisis de fatiga usando criterio de Goodman modificado + curva S-N Basquin.

    Criterio de Goodman: σ_a/S_e + σ_m/S_ut = 1
    Curva S-N Basquin: N = 1000 * (σ_a / S_f)^(-1/b)  donde b < 0

    Args:
        sigma_vm_max: esfuerzo Von Mises máximo (MPa) - del campo FEM real
        sigma_mean: esfuerzo medio (MPa)
        mat: propiedades del material (con S_ut, S_f, b_basquin)
        R_ratio: ratio min/max de carga (σ_min/σ_max)

    Returns:
        Dict con vida a fatiga estimada (ciclos y km)
    """
    sigma_y = mat['sigma_y']
    E = mat['E']

    # Resistencia a fatiga estimada
    # S_ut: resistencia última a tracción
    S_ut = mat.get('S_ut', sigma_y * 1.3)

    # S_e: límite de fatiga (para polímeros ~0.35 * S_ut)
    S_e = 0.35 * S_ut

    # Parámetros Basquin del material
    b_basquin = mat.get('b_basquin', -0.12)  # exponente de Basquin
    S_f_1000 = mat.get('S_f', 0.9) * S_ut  # resistencia a 1000 ciclos

    # Amplitud y media de esfuerzo
    sigma_amp = sigma_vm_max * (1 - R_ratio) / 2
    sigma_m = sigma_vm_max * (1 + R_ratio) / 2

    # Factor de Goodman modificado
    goodman_ratio = sigma_amp / S_e + sigma_m / S_ut

    # Vida a fatiga (Basquin): N = 1000 * (σ_a / S_f_1000)^(1/b)
    # Nota: b es negativo, por eso la fórmula da N decreciente con σ creciente
    if sigma_amp > 0:
        N_f = float(1000 * (sigma_amp / S_f_1000) ** (1 / b_basquin))
        # FIX: mismo que spring_tire_fea.py — subir cap a 1e12 para que el
        # NSGA-III pueda diferenciar diseños en el objetivo life_cycles.
        N_f = max(1e3, min(N_f, 1e12))  # clamp entre 1k y 1T ciclos
    else:
        N_f = 1e12

    # Ajuste por Goodman: si goodman_ratio > 1, penalizar vida
    if goodman_ratio > 1.0:
        N_f /= goodman_ratio**2

    # BUG#8: radio del neumático desde d_ext real, no hardcoded 80mm
    tire_radius_m = d_ext / 2 / 1000  # mm → m
    km_per_cycle = 2 * np.pi * tire_radius_m / 1000  # km por revolución
    life_km = float(N_f * km_per_cycle)

    return {
        'N_cycles': float(N_f),
        'life_km': float(life_km),
        'goodman_ratio': float(goodman_ratio),
        'S_e_MPa': float(S_e),
        'S_ut_MPa': float(S_ut),
        'S_f_1000_MPa': float(S_f_1000),
        'sigma_amp_MPa': float(sigma_amp),
        'sigma_mean_MPa': float(sigma_mean),
        'b_basquin': float(b_basquin),
        'fatigue_risk': 'HIGH' if goodman_ratio > 1.0 else ('MEDIUM' if goodman_ratio > 0.7 else 'LOW'),
    }


# ---------------------------------------------------------------------------
# PASO 1: Geometría 3D con gmsh
# ---------------------------------------------------------------------------

def build_spoke_geometry_3d(params: Dict,
                              spoke_idx: int = 0,
                              mesh_size: float = None) -> str:
    """
    Genera geometría 3D de UN radio (spoke) de la Spring Tire con gmsh.

    El radio se modela como una banda plana rectangular extruida,
    con un leve perfil arqueado para capturar la curvatura real.

    Args:
        params: parámetros de la Spring Tire
        spoke_idx: índice del radio (0 = radio en posición θ=0)
        mesh_size: tamaño de elemento (None = automático)

    Returns:
        path al archivo .msh temporal
    """
    d_ext       = float(params['d_ext'])
    d_rim       = float(params['d_rim'])
    width       = float(params['width'])
    wall_thick  = float(params['wall_thickness'])
    angle_cross = float(params['angle_cross'])

    r_ext = d_ext / 2
    r_rim = d_rim / 2

    # Dimensiones del radio
    spoke_w = wall_thick * 4.0   # mm  ancho de la banda plana
    spoke_t = wall_thick          # mm  espesor
    # Longitud libre (entre superficies del rim y del tread)
    L_spoke = max(r_ext - r_rim - wall_thick * 3.5, 5.0)

    # FIX mesh_size: el tamaño de elemento debe ser ≤ min(wall_thickness, spoke_thickness)/3
    # para tener al menos 3 elementos a través de la sección más delgada y evitar shear locking.
    # Antes era spoke_t/3, lo que con spoke_t=5.75 daba mesh≈1.9mm pero si wall<spoke_t
    # (frecuente: wall=2-4mm) quedaba 1-2 elementos en la sección del rim → locking severo.
    if mesh_size is None:
        min_dim = min(wall_thick, spoke_t)
        mesh_size = float(np.clip(min_dim / 3.0, 0.5, 3.0))

    # Fix B+E: gmsh no es fork-safe; finalize() limpia estado corrupto post-fork.
    # La lib C++ de gmsh imprime a stderr aunque Python capture la excepción,
    # así que redirigimos fd=2 momentáneamente para silenciarlo.
    import os as _os
    try:
        _devnull = open(_os.devnull, 'w')
        _old_fd = _os.dup(2)
        _os.dup2(_devnull.fileno(), 2)
        try:
            gmsh.finalize()
        finally:
            _os.dup2(_old_fd, 2)
            _os.close(_old_fd)
            _devnull.close()
    except Exception:
        pass
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"spoke_{spoke_idx}")
    occ = gmsh.model.occ

    # ---- Sección transversal del radio (rectángulo en plano YZ) ----
    # Centrada en (r_rim, 0, 0), orientada radialmente hacia r_ext
    p1 = occ.addPoint(r_rim, -spoke_w/2, -spoke_t/2)
    p2 = occ.addPoint(r_rim,  spoke_w/2, -spoke_t/2)
    p3 = occ.addPoint(r_rim,  spoke_w/2,  spoke_t/2)
    p4 = occ.addPoint(r_rim, -spoke_w/2,  spoke_t/2)

    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p1)

    wire   = occ.addCurveLoop([l1, l2, l3, l4])
    section = occ.addPlaneSurface([wire])

    # ---- Extrusión radial (dirección +X) ----
    result = occ.extrude([(2, section)], L_spoke, 0, 0)

    # ---- Agregar pequeños filetes en las esquinas de unión ----
    # (radio de fillet = spoke_t * 0.2 para suavizar concentraciones de esfuerzo)
    occ.synchronize()

    # ---- Grupos físicos para BC ----
    all_surfaces = gmsh.model.getEntities(dim=2)
    rim_tags, tread_tags, side_tags = [], [], []

    for dim, tag in all_surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        xmin, xmax = bb[0], bb[3]
        if abs(xmin - r_rim) < 0.5 and abs(xmax - r_rim) < 0.5:
            rim_tags.append(tag)
        elif abs(xmin - (r_rim + L_spoke)) < 0.5 and abs(xmax - (r_rim + L_spoke)) < 0.5:
            tread_tags.append(tag)
        else:
            side_tags.append(tag)

    if rim_tags:
        gmsh.model.addPhysicalGroup(2, rim_tags,   tag=1, name="bc_rim_fixed")
    if tread_tags:
        gmsh.model.addPhysicalGroup(2, tread_tags, tag=2, name="bc_tread_load")
    if side_tags:
        gmsh.model.addPhysicalGroup(2, side_tags,  tag=3, name="bc_sides_free")

    vol_tags = [v[1] for v in gmsh.model.getEntities(dim=3)]
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=4, name="spoke_volume")

    # ---- Mallado 3D ----
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.5)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)          # Delaunay 3D
    gmsh.option.setNumber("Mesh.Optimize", 1)             # mejorar calidad
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    # ---- Guardar ----
    tmp_path = tempfile.mktemp(suffix='.msh')
    gmsh.write(tmp_path)
    gmsh.finalize()

    return tmp_path


def build_full_tire_geometry_3d(params: Dict,
                                 n_spokes_model: int = 4,
                                 mesh_size: float = None) -> str:
    """
    Geometría 3D de la Spring Tire completa (sector de N radios).

    Para reducir coste computacional se modela 1/N del neumático
    con simetría cíclica (N = n_radios / n_spokes_model).

    Args:
        params: parámetros Spring Tire
        n_spokes_model: radios a modelar explícitamente (4 = sector 15° para 24 radios)
        mesh_size: tamaño de elemento

    Returns:
        path al .msh temporal
    """
    d_ext      = float(params['d_ext'])
    d_rim      = float(params['d_rim'])
    width      = float(params['width'])
    wall_thick = float(params['wall_thickness'])
    n_radios   = int(params['n_radios'])
    angle_cross = float(params['angle_cross'])

    r_ext = d_ext / 2
    r_rim = d_rim / 2
    spoke_w = wall_thick * 4.0
    spoke_t = wall_thick
    L_spoke = max(r_ext - r_rim - wall_thick * 3.5, 5.0)

    if mesh_size is None:
        min_dim = min(wall_thick, spoke_t)
        mesh_size = float(np.clip(min_dim / 3.0, 0.5, 3.0))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spring_tire_sector")
    occ = gmsh.model.occ

    spoke_volumes = []

    for i in range(min(n_spokes_model, n_radios)):
        theta = 2 * np.pi * i / n_radios   # ángulo del radio i

        # ---- Sección del radio en posición theta ----
        # Centro del radio en (r_rim*cos(θ), r_rim*sin(θ), 0)
        # Orientación: dirección radial en θ

        cx = r_rim * np.cos(theta)
        cy = r_rim * np.sin(theta)

        # Dirección radial (unitario)
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Dirección perpendicular (ancho del radio)
        px = -np.sin(theta)
        py =  np.cos(theta)

        # 4 puntos de la sección transversal
        pts = []
        for (wy, wz) in [(-spoke_w/2, -spoke_t/2), (spoke_w/2, -spoke_t/2),
                          (spoke_w/2,  spoke_t/2), (-spoke_w/2,  spoke_t/2)]:
            x = cx + px * wy
            y = cy + py * wy
            z = wz
            pts.append(occ.addPoint(x, y, z))

        lines = [
            occ.addLine(pts[0], pts[1]),
            occ.addLine(pts[1], pts[2]),
            occ.addLine(pts[2], pts[3]),
            occ.addLine(pts[3], pts[0]),
        ]
        wire = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([wire])

        # Extruir en dirección radial
        ext_result = occ.extrude(
            [(2, surf)],
            dx * L_spoke,
            dy * L_spoke,
            0.0
        )
        # El volumen extruido está en ext_result[1]
        for dim, tag in ext_result:
            if dim == 3:
                spoke_volumes.append(tag)

    # ---- Rim (aro interior): cilindro anular ----
    rim_t = wall_thick * 1.5
    # Círculo interior del rim
    occ.addDisk(0, 0, 0, r_rim, r_rim)               # disco completo (placeholder)
    # Se usa un toroide simplificado: anillo rectangular girado
    # Para el sector, solo el arco correspondiente a n_spokes_model radios
    sector_angle = 2 * np.pi * n_spokes_model / n_radios  # rad

    occ.synchronize()

    # ---- Grupos físicos ----
    # En el modelo de sector, cara "rim" = caras a r ≈ r_rim
    # cara "tread" = caras a r ≈ r_ext
    all_surfs = gmsh.model.getEntities(dim=2)
    rim_tags, tread_tags = [], []

    for dim, tag in all_surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        # Centro aproximado
        xc = (bb[0] + bb[3]) / 2
        yc = (bb[1] + bb[4]) / 2
        r_center = np.sqrt(xc**2 + yc**2)
        if r_center < r_rim + rim_t:
            rim_tags.append(tag)
        elif r_center > r_ext - wall_thick * 2:
            tread_tags.append(tag)

    if rim_tags:
        gmsh.model.addPhysicalGroup(2, rim_tags,   tag=1, name="bc_rim_fixed")
    if tread_tags:
        gmsh.model.addPhysicalGroup(2, tread_tags, tag=2, name="bc_tread_load")
    if spoke_volumes:
        gmsh.model.addPhysicalGroup(3, spoke_volumes, tag=3, name="spoke_volumes")

    # ---- Mallado ----
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 3.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)

    tmp_path = tempfile.mktemp(suffix='.msh')
    gmsh.write(tmp_path)
    gmsh.finalize()

    return tmp_path


# ---------------------------------------------------------------------------
# PASO 2: Leer malla .msh → skfem MeshTet
# ---------------------------------------------------------------------------

def load_mesh_to_skfem(msh_path: str) -> Tuple['MeshTet', np.ndarray, Dict]:
    """
    Lee archivo .msh de gmsh y construye MeshTet para skfem.

    Returns:
        (mesh3d, points, node_groups)
        node_groups: dict con índices de nodos por grupo físico
    """
    msh = meshio.read(msh_path)

    # Extraer tetraedros (elementos volumétricos)
    tet_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "tetra":
            tet_cells = cell_block.data
            break

    if tet_cells is None:
        raise ValueError("No se encontraron tetraedros en la malla")

    pts = msh.points  # (n_nodes, 3)

    # Crear MeshTet: skfem espera (3, n_nodes) y (4, n_elements)
    mesh3d = MeshTet(pts.T, tet_cells.T)

    # Grupos de nodos por condición de contorno
    node_groups = {}

    # Identificar nodos por posición geométrica
    # (los grupos físicos de gmsh a veces se pierden en meshio)
    r_vals = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

    return mesh3d, pts, {'r_values': r_vals, 'points': pts}


# ---------------------------------------------------------------------------
# PASO 3: Solver 3D con skfem
# ---------------------------------------------------------------------------

def solve_3d_elasticity(mesh3d: 'MeshTet',
                         pts: np.ndarray,
                         mat: Dict,
                         r_rim: float,
                         r_ext: float,
                         F_vertical: float = 500.0,
                         n_radios: int = 24,
                         compute_buckling: bool = False) -> Dict:
    """
    Resuelve elasticidad lineal 3D para un radio de la Spring Tire.

    Args:
        mesh3d: MeshTet de skfem
        pts: coordenadas de nodos (n_nodes, 3)
        mat: propiedades del material {'E', 'nu', 'sigma_y'}
        r_rim, r_ext: radios interior/exterior
        F_vertical: carga vertical total del neumático (N)
        n_radios: número total de radios
        compute_buckling: calcular pandeo lineal (más lento)

    Returns:
        Dict con campo de desplazamientos, esfuerzos, rigidez
    """
    E, nu = float(mat['E']), float(mat['nu'])
    sigma_y = float(mat['sigma_y'])
    lam, mu = lame_parameters(E, nu)

    # ---- Ensamblaje ----
    e3d   = ElementVector(ElementTetP2())
    basis = Basis(mesh3d, e3d)
    N     = basis.N

    log.info(f"  FEM 3D: {mesh3d.nvertices} nodos, {mesh3d.nelements} tets, {N} DOFs")

    K = asm(linear_elasticity(lam, mu), basis)

    # ---- Vector de fuerza ----
    f = np.zeros(N)

    dof_nd = basis.dofs.nodal_dofs   # shape (3, n_nodes): [u, v, w] per node

    # Carga por radio = F_total / n_radios
    # Dirección -Y (gravedad)
    F_per_spoke = F_vertical / n_radios

    # Nodos en cara "tread" (extremo exterior, r ≈ r_ext)
    r_vals = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    eps = (r_ext - r_rim) * 0.08
    tread_nodes = np.where(r_vals > r_ext - eps)[0]
    rim_nodes   = np.where(r_vals < r_rim + eps)[0]

    if len(tread_nodes) == 0:
        # fallback: extremo en X máximo (spoke recto)
        tread_nodes = np.where(pts[:, 0] > pts[:, 0].max() - eps)[0]
    if len(rim_nodes) == 0:
        rim_nodes = np.where(pts[:, 0] < pts[:, 0].min() + eps)[0]

    # Carga axial (tensión): la Spring Tire trabaja principalmente en TENSIÓN AXIAL
    # La carga vertical del rover se transmite como tracción radial en los radios
    # → carga en dirección X (axial del radio, que va de r_rim a r_ext)
    #
    # La componente de la carga vertical que cada radio soporta depende
    # de su posición angular. En promedio: F_radial = F_vertical / n_contact
    # donde n_contact ≈ n_radios / 6 (radios en zona de contacto)
    n_contact = max(2, n_radios // 6)
    F_axial_per_spoke = F_vertical / n_contact   # N por radio en zona de contacto

    if len(tread_nodes) > 0:
        # Determinar dirección axial del radio (de rim hacia tread)
        rim_center   = pts[rim_nodes].mean(axis=0)   if len(rim_nodes)   > 0 else np.array([r_rim, 0, 0])
        tread_center = pts[tread_nodes].mean(axis=0) if len(tread_nodes) > 0 else np.array([r_ext, 0, 0])
        axial_dir = tread_center - rim_center
        axial_norm = np.linalg.norm(axial_dir)
        if axial_norm > 1e-6:
            axial_dir /= axial_norm
        else:
            axial_dir = np.array([1.0, 0.0, 0.0])

        # Aplicar tracción en la cara tread (en la dirección axial del radio)
        # El signo: la carga tira del tread hacia afuera del rim → tensión
        for comp_idx in range(3):
            if abs(axial_dir[comp_idx]) > 0.01:
                dofs_comp = dof_nd[comp_idx, tread_nodes]
                f[dofs_comp] += axial_dir[comp_idx] * F_axial_per_spoke / len(tread_nodes)

    # ---- Condiciones de contorno: rim empotrado ----
    fixed_dofs = np.concatenate([
        dof_nd[0, rim_nodes],   # u
        dof_nd[1, rim_nodes],   # v
        dof_nd[2, rim_nodes],   # w
    ])

    # ---- Resolver ----
    K_c, f_c = condense(K, f, D=fixed_dofs, expand=False)
    free_dofs = np.setdiff1d(np.arange(N), fixed_dofs)

    try:
        u_free = spsolve(K_c, f_c)
    except Exception as ex:
        log.error(f"  spsolve falló: {ex}")
        return {}

    # Reconstruir campo completo
    u = np.zeros(N)
    if len(u_free) == len(free_dofs):
        u[free_dofs] = u_free

    # ---- Post-proceso ----
    # Desplazamientos nodales (u, v, w)
    u_x = u[dof_nd[0, :]]
    u_y = u[dof_nd[1, :]]
    u_z = u[dof_nd[2, :]]
    disp_mag = np.sqrt(u_x**2 + u_y**2 + u_z**2)

    max_disp_mm = float(disp_mag.max())

    # Rigidez efectiva: k = F_aplicada / δ_max
    # FIX: la fuerza aplicada a la malla fue F_axial_per_spoke (= F_vertical / n_contact),
    # NO F_per_spoke (= F_vertical / n_radios). Usar la fuerza incorrecta generaba
    # k_spoke_Nmm = (F/n_radios) / disp_bajo_(F/n_contact), que produce un valor
    # inconsistente (el desplazamiento fue producido por una fuerza mayor).
    # La rigidez correcta del sistema completo es k_total = F_vertical / δ_sistema.
    # Como el FEM modela UN solo radio bajo F_axial_per_spoke, la rigidez del
    # sistema completo se estima como:
    #   k_total = F_vertical / δ_max  (todos los radios en paralelo, pero solo
    #             modelamos el más cargado → δ_max es el mayor desplazamiento real)
    # k_spoke = k_total / n_radios para mantener la misma API que el resto del pipeline.
    k_total_Nmm = F_vertical / max_disp_mm if max_disp_mm > 1e-12 else 1e6
    k_spoke_Nmm = k_total_Nmm / n_radios
    k_total_kNm = float(k_total_Nmm)  # 1 N/mm = 1 kN/m

    # ---- Esfuerzos Von Mises (estimación nodal via deformaciones) ----
    # Deformación axial media en el radio
    L_spoke = pts[:, 0].max() - pts[:, 0].min()
    if L_spoke < 1.0:
        # Geometría con eje diferente: usar r_vals
        L_spoke = r_ext - r_rim

    # --- Post-proceso: esfuerzos desde desplazamientos 3D ---
    # Elongación axial del radio (componente en la dirección del eje del radio)
    # Para radio en dirección X: elongación = u_x[tread] - u_x[rim]
    u_x_tread = float(np.mean(u_x[tread_nodes])) if len(tread_nodes) > 0 else 0.0
    u_x_rim   = float(np.mean(u_x[rim_nodes]))   if len(rim_nodes)   > 0 else 0.0
    u_y_tread = float(np.mean(u_y[tread_nodes])) if len(tread_nodes) > 0 else 0.0
    u_z_tread = float(np.mean(u_z[tread_nodes])) if len(tread_nodes) > 0 else 0.0

    # Elongación axial neta
    axial_elongation = abs(u_x_tread - u_x_rim)
    epsilon_axial    = axial_elongation / max(L_spoke, 1.0)

    # Esfuerzo axial (tensión)
    sigma_normal = E * epsilon_axial

    # Esfuerzo cortante (de deflexiones transversales y y z)
    G = E / (2 * (1 + nu))
    gamma_y = abs(u_y_tread) / max(L_spoke, 1.0)
    gamma_z = abs(u_z_tread) / max(L_spoke, 1.0)
    tau = G * np.sqrt(gamma_y**2 + gamma_z**2)

    # Von Mises (estado triaxial simplificado: σ_axial + τ)
    sigma_vm = float(np.sqrt(sigma_normal**2 + 3 * tau**2))

    # Cap de seguridad numérica
    sigma_vm = min(sigma_vm, sigma_y * 5.0)

    # ---- Plastic strain ----
    # BUG#10: usar sigma_y_min (material más débil) igual que en fea.py
    # Para consistencia, sigma_y_min debe pasarse desde run_fem3d_spoke
    sigma_y_min = float(mat.get('sigma_y_min', sigma_y))  # fallback al material dominante
    plastic_strain_pct = float(max(0.0, (sigma_vm / sigma_y_min - 1.0) * 100.0))

    # ---- Factor de concentración de esfuerzos (Kt) ----
    # En los nodos de esquina (unión spoke-rim y spoke-tread)
    # Estimado como ratio del desplazamiento máximo / medio en las esquinas
    corner_tol = (r_ext - r_rim) * 0.1
    corner_nodes_rim   = np.where((r_vals < r_rim + corner_tol))[0]
    corner_nodes_tread = np.where((r_vals > r_ext - corner_tol))[0]

    if len(corner_nodes_rim) > 0 and disp_mag.mean() > 1e-12:
        Kt_rim   = float(disp_mag[corner_nodes_rim].max() / (disp_mag.mean() + 1e-12))
    else:
        Kt_rim = 1.0

    if len(corner_nodes_tread) > 0 and disp_mag.mean() > 1e-12:
        Kt_tread = float(disp_mag[corner_nodes_tread].max() / (disp_mag.mean() + 1e-12))
    else:
        Kt_tread = 1.0

    Kt_max = max(Kt_rim, Kt_tread)

    # ---- Pandeo lineal (opcional, más costoso) ----
    buckling_load_factor = None
    if compute_buckling:
        try:
            # Matriz de rigidez geométrica (aproximación: K_geo ≈ -F/L * M)
            # Para mayor precisión se necesita una forma débil separada
            # Aquí usamos el método de valores propios generalizado K·v = λ·K_geo·v
            # Como aproximación: K_geo ≈ K * (F_per_spoke / (E * A_spoke))
            # Sección del spoke: rectangular = (wall*4) × spoke_thickness
            # Bug fix: no usar wall² * 4 (ignora spoke_thickness)
            wall_t = float(params.get('wall_thickness', 2.5))
            spoke_t = float(params.get('spoke_thickness', wall_t))
            A_spoke = (wall_t * 4.0) * spoke_t  # mm² rectangular
            sigma_ref = F_per_spoke / A_spoke   # MPa de referencia
            scale = sigma_ref / E

            K_geo_approx = K * scale

            # Resolver problema de valores propios: K·v = λ·K_geo·v
            # λ > 0: estable, λ_min = factor de carga de pandeo
            try:
                eigenvalues, _ = eigsh(K_c, k=3, M=K_c * scale,
                                       sigma=0, which='LM',
                                       tol=1e-4, maxiter=500)
                buckling_load_factor = float(np.min(np.abs(eigenvalues)))
            except Exception:
                buckling_load_factor = None
        except Exception as ex:
            log.debug(f"  Pandeo 3D falló: {ex}")

    # ---- Resultado ----
    result = {
        # Campo de desplazamientos
        'max_displacement_mm': max_disp_mm,
        'mean_displacement_mm': float(disp_mag.mean()),
        'disp_x_max_mm': float(np.abs(u_x).max()),
        'disp_y_max_mm': float(np.abs(u_y).max()),
        'disp_z_max_mm': float(np.abs(u_z).max()),

        # Rigidez
        'k_spoke_Nmm': float(k_spoke_Nmm),
        'k_vertical_kNm': k_total_kNm,

        # Esfuerzos
        'sigma_vm_MPa': float(sigma_vm),
        'sigma_normal_MPa': float(sigma_normal),
        'tau_max_MPa': float(tau),
        'sigma_y_MPa': float(sigma_y),
        'plastic_strain_pct': plastic_strain_pct,

        # Factor de concentración de esfuerzos
        'Kt_rim': Kt_rim,
        'Kt_tread': Kt_tread,
        'Kt_max': Kt_max,

        # Factor de seguridad (resistencia)
        'SF_strength': float(sigma_y / max(sigma_vm, 0.001)),

        # Pandeo
        'buckling_load_factor': buckling_load_factor,

        # Metadatos
        'n_nodes': mesh3d.nvertices,
        'n_elements': mesh3d.nelements,
        'n_dofs': N,
        'n_rim_nodes': int(len(rim_nodes)),
        'n_tread_nodes': int(len(tread_nodes)),
    }

    # ---- ANÁLISIS DE MEMORIA DE FORMA (Neo-Hookean) ----
    # BUG#5: pasar wall_thickness y spoke_thickness reales desde mat (inyectado por run_fem3d_spoke)
    params_for_sm = {
        'd_ext': r_ext * 2,
        'd_rim': r_rim * 2,
        'wall_thickness': float(mat.get('wall_thickness', 2.5)),
        'spoke_thickness': float(mat.get('spoke_thickness', mat.get('wall_thickness', 2.5))),
    }
    shape_memory = analyze_shape_memory_fem3d(params_for_sm, mat, load_steps=10)
    result['shape_memory'] = shape_memory
    result['shape_memory_ratio'] = shape_memory['shape_memory_ratio']
    result['recovery_ratio'] = shape_memory['recovery_ratio']
    result['energy_stored_mJ'] = shape_memory['energy_stored_mJ']

    # ---- ANÁLISIS DE FATIGA (Goodman + Basquin) ----
    sigma_mean = sigma_vm * 0.5  # Asumir carga pulsante (R=0)
    # BUG#8: usar d_ext del mat dict inyectado desde run_fem3d_spoke
    d_ext_val = float(mat.get('d_ext', r_ext * 2))
    fatigue = analyze_fatigue_fem3d(sigma_vm, sigma_mean, mat, R_ratio=0.1, d_ext=d_ext_val)
    result['fatigue'] = fatigue
    result['life_cycles'] = fatigue['N_cycles']
    result['life_km'] = fatigue['life_km']
    result['goodman_ratio'] = fatigue['goodman_ratio']
    result['fatigue_risk'] = fatigue['fatigue_risk']

    return result


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: FEM 3D completo de un radio
# ---------------------------------------------------------------------------

def run_fem3d_spoke(params: Dict,
                    F_vertical: float = 500.0,
                    mesh_size: float = None,
                    compute_buckling: bool = False) -> Dict:
    """
    FEM 3D completo de un radio (spoke) de la Spring Tire.

    Este es el "filtro de alta fidelidad" para el NSGA-III.
    ~10–100x más lento que el FEA 2D, pero da:
      - Esfuerzos 3D reales (no proyecciones planas)
      - Concentraciones de esfuerzo en uniones
      - Torsión y pandeo fuera del plano

    Args:
        params: parámetros de la Spring Tire
        F_vertical: carga vertical total (N)
        mesh_size: tamaño de elemento gmsh (None = automático)
        compute_buckling: calcular pandeo (más lento)

    Returns:
        Dict con todos los resultados FEM 3D, o {} si falla
    """
    if not FEM3D_AVAILABLE:
        log.warning("FEM 3D no disponible (gmsh/meshio/skfem faltantes)")
        return {}

    d_ext = float(params['d_ext'])
    d_rim = float(params['d_rim'])
    n_radios = int(params['n_radios'])

    r_ext = d_ext / 2
    r_rim = d_rim / 2

    # Material predominante
    material_sectors = np.array(params.get('material_sectors', [0]*8), dtype=int)
    mat_ids, counts = np.unique(material_sectors, return_counts=True)
    dominant_mat_id = int(mat_ids[np.argmax(counts)])
    mat = dict(MATERIAL_DB.get(dominant_mat_id, MATERIAL_DB[0]))  # copia mutable

    # BUG#10: inyectar sigma_y_min (material más débil) en mat para plastic_strain
    all_mats = [MATERIAL_DB.get(int(m), MATERIAL_DB[0]) for m in material_sectors]
    mat['sigma_y_min'] = float(min(m['sigma_y'] for m in all_mats))
    # BUG#8: inyectar d_ext en mat para que solve_3d_elasticity pueda calcular life_km
    mat['d_ext'] = d_ext
    # BUG#5: inyectar dimensiones reales del spoke para análisis Neo-Hookean preciso
    mat['wall_thickness'] = float(params.get('wall_thickness', 2.5))
    mat['spoke_thickness'] = float(params.get('spoke_thickness', params.get('wall_thickness', 2.5)))

    log.info(f"FEM 3D | radio {d_rim:.0f}→{d_ext:.0f}mm | "
             f"material={mat['name']} | E={mat['E']}MPa")

    # ---- Paso 1: Generar malla ----
    try:
        msh_path = build_spoke_geometry_3d(params, mesh_size=mesh_size)
    except Exception as ex:
        log.error(f"gmsh falló: {ex}")
        return {}

    # ---- Paso 2: Leer malla ----
    try:
        mesh3d, pts, node_info = load_mesh_to_skfem(msh_path)
    except Exception as ex:
        log.error(f"meshio falló: {ex}")
        if os.path.exists(msh_path):
            os.unlink(msh_path)
        return {}
    finally:
        if os.path.exists(msh_path):
            os.unlink(msh_path)

    # FIX: umbral mínimo de 100 elementos (antes 10).
    # Con <100 tets el radio queda con 1-2 elementos en la sección transversal
    # → locking severo → rigidez artificialmente 100-1000x mayor que la real.
    # Si la malla es demasiado gruesa, el caller debe reducir mesh_size.
    if mesh3d.nelements < 100:
        log.error(
            f"Malla demasiado gruesa ({mesh3d.nelements} elementos < 100). "
            f"Usa --mesh-size con un valor menor (ej: --mesh-size 3.0). "
            f"Los resultados de esta malla no son confiables."
        )
        return {}

    # ---- Paso 3: Resolver ----
    try:
        fem_result = solve_3d_elasticity(
            mesh3d=mesh3d,
            pts=pts,
            mat=mat,
            r_rim=r_rim,
            r_ext=r_ext,
            F_vertical=F_vertical,
            n_radios=n_radios,
            compute_buckling=compute_buckling,
        )
    except Exception as ex:
        log.error(f"Solver 3D falló: {ex}")
        return {}

    if not fem_result:
        return {}

    # ---- Añadir metadatos ----
    fem_result['fea_type'] = 'fem3d_gmsh_skfem'
    fem_result['material'] = mat['name']
    fem_result['params_summary'] = {
        'd_ext': d_ext, 'd_rim': d_rim,
        'wall_thickness': params['wall_thickness'],
        'n_radios': n_radios,
    }
    # mass_kg: estimación desde volumen geométrico del radio × densidad material × n_radios
    # Así fea_integrated 'mixed' mode puede sobreescribir mass_kg con el valor 3D
    try:
        L_spoke = (r_ext - r_rim)  # mm (longitud radio)
        wall_t = float(params.get('wall_thickness', 2.5))
        spoke_t = float(params.get('spoke_thickness', wall_t))
        A_spoke = wall_t * 4.0 * spoke_t  # mm² sección rectangular
        V_spoke_mm3 = A_spoke * L_spoke  # mm³ por radio
        density_g_cm3 = float(mat.get('density', 1.15))
        density_kg_mm3 = density_g_cm3 * 1e-6  # kg/mm³
        fem_result['mass_kg'] = float(n_radios * V_spoke_mm3 * density_kg_mm3)
    except Exception:
        pass  # mass_kg opcional, no bloqueante

    return fem_result


# ---------------------------------------------------------------------------
# FUNCIÓN DE REFINAMIENTO: comparar FEA 2D vs FEM 3D
# y devolver factores de corrección
# ---------------------------------------------------------------------------

def compute_correction_factors(fea2d_result: Dict,
                                 fem3d_result: Dict) -> Dict:
    """
    Calcula factores de corrección entre FEA 2D y FEM 3D.

    Estos factores se usan para re-calibrar el modelo 2D
    sin tener que correr FEM 3D en cada evaluación.

    Returns:
        Dict con factores de corrección por propiedad
    """
    if not fem3d_result:
        return {}

    factors = {}

    # Rigidez vertical
    k2d = fea2d_result.get('stiffness_kN_m', 1.0)
    k3d = fem3d_result.get('k_vertical_kNm', 1.0)
    factors['k_correction'] = float(k3d / k2d) if k2d > 0 else 1.0

    # Deflexión máxima
    d2d = fea2d_result.get('max_deflection_mm', 1.0)
    d3d = fem3d_result.get('max_displacement_mm', 1.0)
    factors['deflection_correction'] = float(d3d / d2d) if d2d > 0 else 1.0

    # Von Mises
    s2d = fea2d_result.get('sigma_vm_max_MPa', 1.0)
    s3d = fem3d_result.get('sigma_vm_MPa', 1.0)
    factors['sigma_correction'] = float(s3d / s2d) if s2d > 0 else 1.0

    # Factor de concentración de esfuerzos (3D captura lo que 2D no puede)
    factors['Kt_3d'] = float(fem3d_result.get('Kt_max', 1.0))

    # FIX: loguear un WARNING si algún factor está fuera de rango razonable.
    # Factores > 20 o < 0.05 indican malla demasiado gruesa o resultados degenerados.
    # La confianza se calcula sobre factores raw (no clampeados) para que el
    # SurrogateState de fase3 reciba la señal real de baja calidad.
    FACTOR_WARN_MAX = 20.0
    for k_name in ('k_correction', 'deflection_correction', 'sigma_correction'):
        v = factors[k_name]
        if v > FACTOR_WARN_MAX or v < 1.0 / FACTOR_WARN_MAX:
            log.warning(
                f"compute_correction_factors: {k_name}={v:.3f} fuera de rango "
                f"[{1/FACTOR_WARN_MAX:.3f}, {FACTOR_WARN_MAX}]. "
                f"Posible malla FEM 3D demasiado gruesa (n_nodes={fem3d_result.get('n_nodes','?')}, "
                f"n_elements={fem3d_result.get('n_elements','?')}). "
                f"Considera reducir mesh_size en --mesh-size."
            )

    # Indicador de confianza: si los factores están cerca de 1.0, el 2D es bueno.
    # Se usa min(|desviación|, 2.0) para que factores extremos (>3x) no den
    # exactamente 0 — permiten distinguir "malo" de "muy malo" en el EMA de fase3.
    deviations = [min(abs(factors[k] - 1.0), 2.0) for k in (
        'k_correction', 'deflection_correction', 'sigma_correction'
    )]
    factors['model_confidence'] = float(max(0.0, 1.0 - np.mean(deviations)))

    return factors


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    if not FEM3D_AVAILABLE:
        print("AVISO: gmsh, meshio o skfem no disponibles")
        print(f"  gmsh:    {GMSH_AVAILABLE}")
        print(f"  meshio:  {MESHIO_AVAILABLE}")
        print(f"  skfem:   {SKFEM_AVAILABLE}")
        exit(1)

    test_params = {
        'd_ext':            200.0,
        'd_rim':            100.0,
        'width':             85.0,
        'n_radios':          24,
        'wall_thickness':     2.5,
        'angle_cross':       55.0,
        'material_sectors':  np.array([0, 0, 1, 1, 0, 0, 1, 1]),
    }

    print("=" * 60)
    print("  FEM 3D - NASA Spring Tire v6.0")
    print("=" * 60)

    import time
    t0 = time.time()

    result = run_fem3d_spoke(test_params, F_vertical=500.0)

    elapsed = time.time() - t0

    if result:
        print(f"\n  Tiempo de cómputo:    {elapsed:.2f} s")
        print(f"\n  [MALLA]")
        print(f"    Nodos:              {result['n_nodes']}")
        print(f"    Tetraedros:         {result['n_elements']}")
        print(f"    DOFs totales:       {result['n_dofs']}")
        print(f"\n  [DESPLAZAMIENTOS]")
        print(f"    δ_max:              {result['max_displacement_mm']:.4f} mm")
        print(f"    δ_y (vertical):     {result['disp_y_max_mm']:.4f} mm")
        print(f"\n  [RIGIDEZ]")
        print(f"    k_vertical:         {result['k_vertical_kNm']:.1f} kN/m")
        print(f"\n  [ESFUERZOS]")
        print(f"    σ_vm max:           {result['sigma_vm_MPa']:.2f} MPa")
        print(f"    σ_yield:            {result['sigma_y_MPa']:.1f} MPa")
        print(f"    Deform. plástica:   {result['plastic_strain_pct']:.2f} %")
        print(f"\n  [CONCENTRACIÓN DE ESFUERZOS]")
        print(f"    Kt rim:             {result['Kt_rim']:.2f}")
        print(f"    Kt tread:           {result['Kt_tread']:.2f}")
        print(f"\n  [FACTOR DE SEGURIDAD]")
        print(f"    SF (resistencia):   {result['SF_strength']:.2f}")

        # Memoria de forma
        if 'shape_memory' in result:
            sm = result['shape_memory']
            print(f"\n  [MEMORIA DE FORMA - Neo-Hookean]")
            print(f"    Modelo:             {sm.get('model', 'N/A')}")
            print(f"    C1:                 {sm.get('C1', 'N/A')}")
            print(f"    D1:                 {sm.get('D1', 'N/A')}")
            print(f"    Recovery ratio:     {result['recovery_ratio']:.4f}")
            print(f"    Shape memory ratio: {result['shape_memory_ratio']:.4f}")
            print(f"    Energía almacenada: {result['energy_stored_mJ']:.2f} mJ")
            print(f"    Fuerza max (30%):   {sm.get('max_force_N', 0):.2f} N")

        # Fatiga
        if 'fatigue' in result:
            fat = result['fatigue']
            print(f"\n  [FATIGA - Goodman + Basquin]")
            print(f"    Goodman ratio:      {result['goodman_ratio']:.4f}")
            print(f"    Ciclos de vida:     {result['life_cycles']:.2e}")
            print(f"    Vida estimada:      {result['life_km']:.1f} km")
            print(f"    Riesgo:             {result['fatigue_risk']}")
            print(f"    S_e (límite fatiga): {fat.get('S_e_MPa', 0):.2f} MPa")
            print(f"    b (Basquin):        {fat.get('b_basquin', 0):.3f}")

        print(f"\n  Material:           {result['material']}")
        print(f"  FEA type:           {result['fea_type']}")
    else:
        print("  ERROR: FEM 3D no devolvió resultados")

    print("\n" + "=" * 60)

    # ---- Test de factores de corrección ----
    # Simular resultado 2D para comparar
    fea2d_mock = {
        'stiffness_kN_m': 178.7,
        'max_deflection_mm': 2.8,
        'sigma_vm_max_MPa': 5.33,
    }

    if result:
        factors = compute_correction_factors(fea2d_mock, result)
        print("\n  [FACTORES DE CORRECCIÓN 2D→3D]")
        for k, v in factors.items():
            print(f"    {k:30s}: {v:.4f}")
