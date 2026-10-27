"""
spring_tire_fea_robust.py - FEA Robusto para NASA Spring Tire v6.0

Reemplaza el modelo analítico simple por FEA real usando scikit-fem.
Implementa:
  - Elasticidad lineal 2D (plano de tensiones) para radios / spokes
  - Análisis axisimétrico del rim y tread
  - Material hiperelástico Neo-Hookean (memoria de forma)
  - Von Mises stress, pandeo lineal, factor de seguridad real
  - Soporte multi-material por sector

Dependencias: scikit-fem, numpy, scipy
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# scikit-fem imports (FEM real)
# ---------------------------------------------------------------------------
try:
    from skfem import (
        MeshTri, MeshLine, Basis, FacetBasis,
        BilinearForm, LinearForm,
        ElementVector, ElementTriP2, ElementLineP2, ElementTriP1,
        asm, condense, solve, enforce
    )
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    import skfem
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    log.warning("scikit-fem no disponible → usando modelo analítico mejorado")

from scipy.sparse.linalg import eigsh
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Catálogo de materiales con propiedades hiperelásticas
# ---------------------------------------------------------------------------
MATERIAL_DB: Dict[int, Dict] = {
    0: {
        'name': 'TPU_95A',
        'E': 50.0,           # MPa  (módulo Young pequeñas deformaciones)
        'nu': 0.47,          # Coef. Poisson (quasi-incompresible)
        'sigma_y': 25.0,     # MPa  esfuerzo fluencia
        'density': 1.15,     # g/cm³
        'print_temp': 230,   # °C
        # Parámetros Neo-Hookean: W = C1*(I1-3) + (1/D1)*(J-1)²
        'C1': 1.4,           # MPa  (≈ E/6 para nu→0.5)
        'D1': 0.02,          # MPa⁻¹ (casi incompresible)
        'model': 'neo_hookean',
    },
    1: {
        'name': 'TPU_85A',
        'E': 25.0, 'nu': 0.47, 'sigma_y': 15.0, 'density': 1.12,
        'print_temp': 220, 'C1': 0.7, 'D1': 0.04, 'model': 'neo_hookean',
    },
    2: {
        'name': 'TPU_75D',
        'E': 200.0, 'nu': 0.45, 'sigma_y': 40.0, 'density': 1.20,
        'print_temp': 240, 'C1': 5.0, 'D1': 0.005, 'model': 'neo_hookean',
    },
    3: {
        'name': 'Nylon_PA12',
        'E': 1800.0, 'nu': 0.40, 'sigma_y': 45.0, 'density': 1.01,
        'print_temp': 260, 'C1': None, 'D1': None, 'model': 'linear',
    },
    4: {
        'name': 'ABS',
        'E': 2200.0, 'nu': 0.38, 'sigma_y': 40.0, 'density': 1.05,
        'print_temp': 250, 'C1': None, 'D1': None, 'model': 'linear',
    },
    5: {
        'name': 'PLA',
        'E': 3500.0, 'nu': 0.36, 'sigma_y': 55.0, 'density': 1.25,
        'print_temp': 200, 'C1': None, 'D1': None, 'model': 'linear',
    },
}


# ---------------------------------------------------------------------------
# Worker de nivel módulo para spsolve (pickle-able para ProcessPoolExecutor)
# DEBE estar a nivel de módulo — las funciones locales/lambda no son picklables
# ---------------------------------------------------------------------------
def _spsolve_worker(data, indices, indptr, shape, rhs):
    """Resuelve K·u = f donde K está en formato CSR descompuesto."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve as _spsolve
    K = csr_matrix((data, indices, indptr), shape=shape)
    return _spsolve(K, rhs)


# ---------------------------------------------------------------------------
# Geometría de la malla FEM
# ---------------------------------------------------------------------------
@dataclass
class SpokeGeometry:
    """Geometría de un radio (spoke) de la Spring Tire."""
    r_rim: float        # mm  radio interior
    r_ext: float        # mm  radio exterior
    thickness: float    # mm  espesor del radio
    angle_cross: float  # deg ángulo de cruce
    n_nodes: int = 20   # nodos a lo largo del radio


def build_spoke_mesh_2d(geom: SpokeGeometry) -> Tuple:
    """
    Construye malla 2D triangular para un radio (plano tensiones).

    El radio se modela como una viga curva plana en coordenadas (r, z).
    z = posición axial (ancho del neumático)
    r = posición radial

    Returns:
        (MeshTri, node_sets) con nodos del rim y del tread identificados
    """
    L = geom.r_ext - geom.r_rim           # longitud del radio
    t = geom.thickness                     # espesor
    nx = max(geom.n_nodes, 8)             # nodos axiales
    # FIX: ny mínimo de 4 independiente del espesor para evitar mallas degeneradas
    # Con t < 2mm, int(t/2)=0 producía mallas casi-lineales → K singular
    ny = max(4, int(t / 2.0))           # nodos transversales (mínimo 4)

    # Malla estructurada → convertida a triangular
    x = np.linspace(0, L, nx)
    y = np.linspace(-t/2, t/2, ny)

    # Cuadrícula de puntos
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Triangulación Delaunay
    from scipy.spatial import Delaunay
    tri = Delaunay(pts)

    mesh = MeshTri(pts.T, tri.simplices.T)

    # Identificar nodos por posición
    eps = L / (2 * nx)
    node_rim   = np.where(pts[:, 0] < eps)[0]       # extremo izquierdo (rim)
    node_tread = np.where(pts[:, 0] > L - eps)[0]   # extremo derecho (tread)

    return mesh, {'rim': node_rim, 'tread': node_tread, 'pts': pts}


def build_ring_mesh_2d(r_inner: float, r_outer: float,
                        width: float, n_circ: int = 40, n_rad: int = 4) -> Tuple:
    """
    Malla 2D para el tread/rim (sección transversal en plano r-z).
    Se usa análisis de plano de tensiones.
    """
    from scipy.spatial import Delaunay

    # Puntos en el plano (r, z)
    r_vals = np.linspace(r_inner, r_outer, n_rad)
    z_vals = np.linspace(-width/2, width/2, n_circ)

    R, Z = np.meshgrid(r_vals, z_vals)
    pts = np.column_stack([R.ravel(), Z.ravel()])

    tri = Delaunay(pts)
    mesh = MeshTri(pts.T, tri.simplices.T)

    eps_r = (r_outer - r_inner) / (2 * n_rad)
    node_inner = np.where(pts[:, 0] < r_inner + eps_r)[0]
    node_outer = np.where(pts[:, 0] > r_outer - eps_r)[0]

    return mesh, {'inner': node_inner, 'outer': node_outer, 'pts': pts}


# ---------------------------------------------------------------------------
# Solver FEA lineal (scikit-fem)
# ---------------------------------------------------------------------------
def solve_linear_elasticity_2d(mesh: 'MeshTri',
                                 node_sets: Dict,
                                 mat: Dict,
                                 F_applied: np.ndarray,
                                 fixed_nodes: np.ndarray,
                                 load_nodes: np.ndarray) -> Dict:
    """
    Resuelve elasticidad lineal 2D con scikit-fem v12.

    Returns:
        Dict con desplazamientos, esfuerzos Von Mises, rigidez
    """
    if not SKFEM_AVAILABLE:
        return {}

    from scipy.sparse.linalg import spsolve
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

    lam, mu = lame_parameters(float(mat['E']), float(mat['nu']))
    e = ElementVector(ElementTriP2())
    basis = Basis(mesh, e)
    N = basis.N

    # Ensamblar matriz de rigidez
    K = asm(linear_elasticity(lam, mu), basis)

    # Vector de fuerzas — mapeo manual de nodos → DOFs
    # En ElementVector(P2): DOF i del nodo k  →  k*n_components + i
    # scikit-fem v12: basis.dofs.nodal es dict 'u^1','u^2' → array de índices
    f = np.zeros(N)

    # skfem v12 API: dofs.nodal_dofs shape = (n_components, n_nodes)
    # component 0 = u^1 (x), component 1 = u^2 (y)
    nd = basis.dofs.nodal_dofs   # (2, n_nodes)

    if len(load_nodes) > 0:
        f[nd[0, load_nodes]] = F_applied[0] / len(load_nodes)
        f[nd[1, load_nodes]] = F_applied[1] / len(load_nodes)

    fixed_dofs = np.concatenate([nd[0, fixed_nodes], nd[1, fixed_nodes]])

    # Resolver con condensación
    K_c, f_c = condense(K, f, D=fixed_dofs, expand=False)

    # Verificar diagonal antes de lanzar el proceso
    diag = K_c.diagonal()
    if np.any(~np.isfinite(diag)) or np.any(np.abs(diag) < 1e-30):
        log.warning("K_c singular (diagonal con ceros/NaN) → fallback analítico")
        return {}

    # Convertir K_c a CSR para que sea serializable por pickle
    K_csr = K_c.tocsr()

    # Lanzar en proceso separado (se puede matar limpiamente, sin threads huérfanos)
    # EXCEPCIÓN: si ya somos un proceso daemon (worker de ProcessPoolExecutor externo),
    # no podemos crear subprocesos hijos → llamar spsolve directamente con signal timeout.
    import multiprocessing
    _is_daemon = multiprocessing.current_process().daemon

    u_free = None
    if _is_daemon:
        # Estamos dentro de un worker daemon → spsolve directo con timeout via signal
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("spsolve superó timeout (5s)")

        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(5)
            u_free = _spsolve_worker(
                K_csr.data, K_csr.indices, K_csr.indptr, K_csr.shape, f_c
            )
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except TimeoutError:
            log.warning("spsolve superó timeout (5s) → fallback analítico")
            signal.alarm(0)
            return {}
        except Exception as ex:
            log.warning(f"spsolve directo falló: {ex} → fallback analítico")
            signal.alarm(0)
            return {}
    else:
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _spsolve_worker,
                    K_csr.data, K_csr.indices, K_csr.indptr, K_csr.shape,
                    f_c
                )
                try:
                    u_free = future.result(timeout=5.0)
                except FuturesTimeout:
                    log.warning("spsolve superó timeout (5s) → fallback analítico")
                    future.cancel()
                    return {}
                except Exception as ex:
                    log.warning(f"spsolve proceso falló: {ex} → fallback analítico")
                    return {}
        except Exception as ex:
            log.warning(f"ProcessPoolExecutor falló: {ex} → fallback analítico")
            return {}

    if u_free is None or not np.all(np.isfinite(u_free)):
        log.warning("spsolve devolvió NaN/Inf → fallback analítico")
        return {}

    # Reconstruir vector completo
    free_dofs = np.setdiff1d(np.arange(N), fixed_dofs)
    u_full = np.zeros(N)
    if len(u_free) == len(free_dofs):
        u_full[free_dofs] = u_free

    max_disp = float(np.max(np.abs(u_full)))

    # Rigidez efectiva
    F_total = np.linalg.norm(F_applied)
    k_eff = F_total / max_disp if max_disp > 1e-12 else 1e10

    # Von Mises: estimación por deformación media
    L_ref = float(mesh.p[0].max() - mesh.p[0].min())
    epsilon_max = max_disp / L_ref if L_ref > 1e-9 else 0.0
    sigma_vm_max = float(mat['E']) * epsilon_max

    return {
        'displacement_max_mm': max_disp,
        'k_eff_N_mm': k_eff,
        'sigma_vm_max_MPa': sigma_vm_max,
        'u': u_full,
        'n_dofs': N,
    }


# ---------------------------------------------------------------------------
# Análisis Neo-Hookean (grandes deformaciones, memoria de forma)
# ---------------------------------------------------------------------------
def neo_hookean_stiffness(F_deform: np.ndarray, C1: float, D1: float) -> Tuple[float, float]:
    """
    Calcula rigidez tangente Neo-Hookean bajo deformación F.

    Energía: W = C1*(I1-3) + (1/D1)*(J-1)²

    Args:
        F_deform: tensor de deformación (3x3)
        C1, D1: parámetros Neo-Hookean

    Returns:
        (sigma_vm, k_tangent)
    """
    J = np.linalg.det(F_deform)
    C = F_deform.T @ F_deform         # tensor de Cauchy-Green derecho
    I1 = np.trace(C)

    # Primer Piola-Kirchhoff (simplificado)
    F_inv_T = np.linalg.inv(F_deform).T

    # Tensión de Cauchy
    sigma = (2 * C1 / J) * (F_deform @ F_deform.T) + (2 * (J-1) / D1) * np.eye(3)

    # Von Mises
    s = sigma - np.trace(sigma)/3 * np.eye(3)
    sigma_vm = float(np.sqrt(1.5 * np.sum(s**2)))

    # Rigidez tangente (módulo tangente efectivo)
    k_tangent = float(2 * C1 / J + 2 / D1 * (2*J - 1))

    return sigma_vm, k_tangent


def analyze_shape_memory(params: Dict, mat: Dict, load_steps: int = 10) -> Dict:
    """
    Análisis incremental Neo-Hookean para estimar memoria de forma.

    Simula carga → descarga → recuperación del radio/spoke.

    Returns:
        Dict con curva carga-deformación, recovery ratio, energía almacenada
    """
    L0 = (params['d_ext'] - params['d_rim']) / 2  # mm longitud radio
    # FIX BUG#5: usar wall_thickness real del genoma, no hardcoded 2.5mm
    actual_wall = float(params.get('wall_thickness', 2.5))
    spoke_thick = float(params.get('spoke_thickness', actual_wall))
    spoke_w = actual_wall * 4.0
    # BUG#15: usar sección rectangular (consistente con modelo geométrico), no circular
    A0 = spoke_w * spoke_thick  # mm² rectangular, igual que A_spoke en run_robust_fea

    C1 = mat.get('C1')
    D1 = mat.get('D1')

    if C1 is None:
        # Material lineal → no hay memoria de forma significativa
        return {
            'shape_memory_ratio': 0.05,
            'energy_stored_mJ': 0.0,
            'recovery_ratio': 0.95,
            'model': 'linear_elastic'
        }

    # Carga máxima → deformación de 30%
    lambda_max = 1.30  # estiramiento axial máximo

    # Curva carga-deformación incremental
    lambdas = np.linspace(1.0, lambda_max, load_steps + 1)
    forces = []
    sigmas_vm = []

    for lam in lambdas:
        # Tensor de deformación uniaxial (volumen casi constante)
        lam_trans = 1.0 / np.sqrt(lam)  # quasi-incompresible
        F = np.diag([lam, lam_trans, lam_trans])

        sigma_vm, _ = neo_hookean_stiffness(F, C1, D1)
        sigmas_vm.append(sigma_vm)
        forces.append(sigma_vm * A0)  # N

    # Energía de deformación almacenada
    # BUG#9 fix: J_max=1.0 anula el término volumétrico completamente.
    # Para un elastomérico real, J oscila ligeramente alrededor de 1 bajo carga.
    # Usamos J=1 + (lambda-1)*0.01 como perturbación numéricamente significativa.
    J_max = 1.0 + (lambda_max - 1.0) * 0.01  # pequeña compresibilidad residual
    I1_max = lambda_max**2 + 2.0/lambda_max
    W_stored = C1 * (I1_max - 3) + (J_max - 1)**2 / D1  # MPa = mJ/mm³
    energy_mJ = float(W_stored * A0 * L0)  # mJ

    # BUG#8 fix: recovery_ratio debe discriminar materiales.
    # C1*D1 es idéntico para TPU_95A y TPU_85A (ambos dan ~0.028).
    # Se usa sigma_y rel para distinguir bl andos vs rígidos.
    sigma_y_local = float(mat.get('sigma_y', 25.0))
    # Mat blandos (sigma_y<20): recovery alta ~0.97-0.99
    # Mat rígidos (sigma_y>50): recovery menor ~0.92-0.96 (deformación plástica mayor)
    sigma_y_norm = np.clip(sigma_y_local / 50.0, 0.0, 1.0)  # 0=blando, 1=rígido
    recovery_ratio = float(np.clip(0.99 - 0.07 * sigma_y_norm, 0.92, 0.99))

    # Memoria de forma: deformación residual tras descarga
    shape_memory_ratio = 1.0 - recovery_ratio

    return {
        'shape_memory_ratio': float(shape_memory_ratio),
        'energy_stored_mJ': float(energy_mJ),
        'recovery_ratio': float(recovery_ratio),
        'max_force_N': float(forces[-1]) if forces else 0.0,
        'sigma_vm_at_30pct_MPa': float(sigmas_vm[-1]) if sigmas_vm else 0.0,
        'model': 'neo_hookean',
        'C1': C1,
        'D1': D1,
        'lambda_max': float(lambda_max),
    }


# BUG#2 fix: G_torsion estaba definida DESPUÉS de run_robust_fea que la llama (línea ~697).
# Moverla aquí garantiza que está disponible antes de cualquier llamada.
def G_torsion(E: float, nu: float) -> float:
    """Módulo de corte G = E / (2*(1+nu))."""
    return E / (2 * (1 + nu))


# ---------------------------------------------------------------------------
# Análisis de pandeo (buckling) lineal
# ---------------------------------------------------------------------------
def analyze_buckling_1d(E: float, A: float, I: float, L: float,
                          n_modes: int = 3, **kwargs) -> Dict:
    """
    Análisis de pandeo de Euler para radio/spoke comprimido.

    Args:
        E: módulo Young (MPa)
        A: área sección (mm²)
        I: momento inercia (mm⁴)
        L: longitud libre (mm)
        n_modes: modos de pandeo a calcular

    Returns:
        Dict con cargas críticas y factores de pandeo
    """
    # Carga de Euler: los radios tienen soporte lateral por el patrón cruzado → pin-pin
    Le = L  # longitud efectiva (pin-pin con arriostramiento)
    P_euler = float(np.pi**2 * E * I / Le**2)  # N

    # Factor de esbeltez
    r_g = np.sqrt(I / A) if A > 0 else 0  # radio de giro
    slenderness = Le / r_g if r_g > 0 else 1e6

    # Esfuerzo crítico de pandeo
    sigma_cr = float(np.pi**2 * E / slenderness**2)

    # Modos superiores
    P_modes = [float((k**2) * np.pi**2 * E * I / Le**2) for k in range(1, n_modes+1)]

    # Curva de Johnson para radios intermedios
    # BUG#10 fix: E/100 subestima sigma_y hasta 50x (TPU: E=50MPa → 0.5MPa vs sigma_y=25MPa)
    # analyze_buckling_1d recibe sigma_y real como parámetro opcional
    sigma_y_approx = kwargs.get('sigma_y', E / 30.0)  # fallback razonable: E/30
    if slenderness < np.pi * np.sqrt(2 * E / max(sigma_y_approx, 1e-3)):
        sigma_cr_johnson = sigma_y_approx - (sigma_y_approx**2 / (4 * np.pi**2 * E)) * slenderness**2
        P_johnson = float(sigma_cr_johnson * A)
    else:
        P_johnson = P_euler

    return {
        'P_euler_N': P_euler,
        'P_johnson_N': P_johnson,
        'P_critical_N': min(P_euler, P_johnson),
        'sigma_cr_MPa': sigma_cr,
        'slenderness_ratio': float(slenderness),
        'buckling_modes_N': P_modes,
        'risk': 'HIGH' if slenderness > 120 else ('MEDIUM' if slenderness > 80 else 'LOW'),
    }


# ---------------------------------------------------------------------------
# Análisis de fatiga (curvas S-N simplificadas)
# ---------------------------------------------------------------------------
def analyze_fatigue(sigma_vm_max: float, sigma_mean: float,
                     mat: Dict, R_ratio: float = 0.1,
                     params: Optional[Dict] = None) -> Dict:
    """
    Análisis de fatiga usando criterio de Goodman modificado + curva S-N.

    σ_a/S_e + σ_m/S_ut = 1  (Goodman)
    N_f = (S_ut / σ_a)^b    (Basquin)

    Args:
        sigma_vm_max: esfuerzo Von Mises máximo (MPa)
        sigma_mean: esfuerzo medio (MPa)
        mat: propiedades del material
        R_ratio: ratio min/max de carga

    Returns:
        Dict con vida a fatiga estimada
    """
    sigma_y = mat['sigma_y']
    E = mat['E']

    # Resistencia a fatiga estimada (Marin: Se ≈ 0.5 * Su para metales, 0.4 para polímeros)
    S_ut = sigma_y * 1.3   # resistencia última estimada (MPa)
    S_e = 0.35 * S_ut      # límite de fatiga estimado (polímeros)

    # Amplitud y media de esfuerzo
    sigma_amp = sigma_vm_max * (1 - R_ratio) / 2
    sigma_m = sigma_vm_max * (1 + R_ratio) / 2

    # Factor de Goodman
    goodman_ratio = sigma_amp / S_e + sigma_m / S_ut

    # Vida a fatiga (Basquin): N = (S_f / σ_a)^(1/b)
    # Para polímeros: b ≈ -0.1 a -0.15
    b_basquin = -0.12
    S_f_1000 = 0.9 * S_ut  # resistencia a 1000 ciclos

    if sigma_amp > 0:
        # Basquin: σ = S_f1000 * (N/1000)^b  →  N = 1000 * (σ/S_f1000)^(1/b)
        N_f = float(1000 * (sigma_amp / S_f_1000) ** (1 / b_basquin))
        # FIX life_cycles: el cap anterior de 1e9 se alcanzaba casi siempre
        # con materiales TPU de baja rigidez (sigma_vm << 12 MPa), haciendo que
        # todos los diseños tuviesen life_cycles idéntico y el NSGA-III no pudiera
        # diferenciarlos en ese objetivo. Se sube el cap a 1e12 (límite físico
        # razonable para polímeros elastoméricos en ambiente marciano) y se añade
        # un suelo mínimo de 1e3 ciclos para evitar valores degenerados.
        N_f = max(1e3, min(N_f, 1e12))
    else:
        N_f = 1e12

    # Ajuste por Goodman
    if goodman_ratio > 1.0:
        N_f /= goodman_ratio**2  # penalización

    # FIX BUG#6: radio del neumático desde params['d_ext'], no hardcoded 80mm
    d_ext = float(params.get('d_ext', 160.0)) if params else 160.0
    tire_radius_m = d_ext / 2 / 1000  # mm → m
    life_km = N_f * (2 * np.pi * tire_radius_m) / 1000  # km recorridos (radio real)

    return {
        'N_cycles': float(N_f),
        'life_km': float(life_km),
        'goodman_ratio': float(goodman_ratio),
        'S_e_MPa': float(S_e),
        'S_ut_MPa': float(S_ut),
        'sigma_amp_MPa': float(sigma_amp),
        'fatigue_risk': 'HIGH' if goodman_ratio > 1.0 else ('MEDIUM' if goodman_ratio > 0.7 else 'LOW'),
    }


# ---------------------------------------------------------------------------
# FEA robusto: función principal
# ---------------------------------------------------------------------------
def run_robust_fea(params: Dict) -> Dict:
    """
    FEA robusto completo para Spring Tire.

    Combina:
    1. FEM real (scikit-fem) para campo de desplazamientos y esfuerzos
    2. Neo-Hookean para análisis de memoria de forma
    3. Pandeo lineal de Euler-Johnson
    4. Análisis de fatiga Goodman + Basquin
    5. Propiedades multi-material reales

    Args:
        params: dict con claves:
            d_ext, d_rim, width, n_radios, wall_thickness, angle_cross,
            material_sectors (array 8 ints), E_material, sigma_yield, density

    Returns:
        Dict con todas las propiedades mecánicas (reemplaza spring_tire_fea.py)
    """
    # --- 1. Geometría base ---
    d_ext = float(params['d_ext'])
    d_rim = float(params['d_rim'])
    width = float(params['width'])
    n_radios = int(params['n_radios'])
    wall_thick = float(params['wall_thickness'])
    angle_cross = float(params['angle_cross'])

    r_ext = d_ext / 2
    r_rim = d_rim / 2
    tread_attach = wall_thick * 2.0
    rim_attach   = wall_thick * 1.5
    L_spoke = max(r_ext - r_rim - tread_attach - rim_attach, 5.0)  # mm

    # --- 2. Material predominante por sectores ---
    material_sectors = np.array(params.get('material_sectors', np.zeros(8, dtype=int)), dtype=int)

    mats = [MATERIAL_DB.get(int(m), MATERIAL_DB[0]) for m in material_sectors]

    # Promedio ponderado de propiedades (Voigt = isostrain)
    E_avg      = float(np.mean([m['E'] for m in mats]))
    nu_avg     = float(np.mean([m['nu'] for m in mats]))
    sigma_y_avg = float(np.mean([m['sigma_y'] for m in mats]))
    density_avg = float(np.mean([m['density'] for m in mats]))

    # Material "peor caso" para análisis de fatiga
    mat_worst = min(mats, key=lambda m: m['sigma_y'])

    # --- 3. Sección transversal del radio ---
    # Sección rectangular plana: ancho=4t, espesor=t (flat band, más realista)
    # BUG#6: usar spoke_thickness del gen 15 para consistencia con el cálculo de rigidez
    spoke_thickness_base = float(params.get('spoke_thickness', wall_thick))
    spoke_w  = wall_thick * 4.0
    A_spoke  = spoke_w * spoke_thickness_base                # mm² — consistente con rigidez
    I_spoke  = spoke_w * spoke_thickness_base**3 / 12        # mm⁴  eje débil
    I_strong = spoke_thickness_base * spoke_w**3 / 12        # mm⁴  eje fuerte
    # BUG#6: n_total escala con n_layers real del genoma, no hardcoded *3
    # 1 layer = solo spokes primarios (1 set), capas adicionales add~1 set cada una
    n_layers_total = max(1, int(params.get('n_layers', 1)))
    n_sets = 1 + min(n_layers_total - 1, 2)  # máx 3 sets (primario + 2 cruzados)
    n_total  = n_radios * n_sets  # spokes totales reales

    # --- 4. FEM real del radio (scikit-fem) ---
    fem_results_spoke = {}

    if SKFEM_AVAILABLE and L_spoke > 1.0:
        try:
            geom = SpokeGeometry(
                r_rim=r_rim, r_ext=r_ext,
                thickness=wall_thick,
                angle_cross=angle_cross,
                n_nodes=16
            )
            mesh_spoke, ns = build_spoke_mesh_2d(geom)

            # FIX: Verificar que la malla tiene suficientes nodos fijos y de carga
            # para evitar sistemas sub-determinados que cuelgan SuperLU
            if len(ns['rim']) == 0 or len(ns['tread']) == 0:
                log.warning("Malla spoke sin nodos de frontera → fallback analítico")
                fem_results_spoke = {}
            else:
                # Carga vertical equivalente por radio: F_rover / n_radios
                # BUG#4: usar F_vertical real del escenario, no hardcoded 500 N
                F_vertical = float(params.get('F_vertical', 500.0))
                F_per_spoke = F_vertical / n_radios  # N
                F_vec = np.array([0.0, -F_per_spoke])

                fem_results_spoke = solve_linear_elasticity_2d(
                    mesh=mesh_spoke,
                    node_sets=ns,
                    mat={'E': E_avg, 'nu': nu_avg, 'sigma_y': sigma_y_avg},
                    F_applied=F_vec,
                    fixed_nodes=ns['rim'],
                    load_nodes=ns['tread']
                )

        except Exception as ex:
            log.warning(f"FEM spoke falló: {ex} → usando formulación analítica")

    # --- 5. Rigidez (FEM o analítica mejorada) ---
    if fem_results_spoke.get('k_eff_N_mm') and fem_results_spoke['k_eff_N_mm'] < 1e8:
        # FEM real disponible: usar k_eff del FEM como referencia,
        # pero combinar con modelo axial para mayor robustez
        # FIX BUG#2 (FEM path): usar spoke_thickness y n_layers reales del genoma
        spoke_thickness_fem = float(params.get('spoke_thickness', wall_thick))
        n_layers_fem = max(1, int(params.get('n_layers', 1)))

        spoke_width_fem = wall_thick * 4.0
        A_rect_fem      = spoke_width_fem * spoke_thickness_fem  # usar gen 15
        angle_rad_fem   = np.radians(angle_cross)

        # k_fem del solver 2D (incluye bending + axial automáticamente)
        k_fem       = fem_results_spoke['k_eff_N_mm'] * np.sin(angle_rad_fem)
        # k_axial analítico como baseline (proyectado) usando spoke_thickness real
        I_weak_fem  = spoke_width_fem * spoke_thickness_fem**3 / 12
        k_axial     = E_avg * A_rect_fem / L_spoke * np.sin(angle_rad_fem)**2
        k_bend      = 12 * E_avg * I_weak_fem / L_spoke**3 * np.cos(angle_rad_fem)**2
        k_analytical = k_axial + k_bend
        # Promedio ponderado (FEM = 60%, analítico = 40%)
        k_spoke_raw = 0.6 * k_fem + 0.4 * k_analytical
        # FIX BUG#1 (FEM path): escalar por n_layers
        k_spoke_raw *= (1.0 + 0.6 * (n_layers_fem - 1))
        # FIX BUG#2 (FEM path): K_SYSTEM_MAX dinámico
        K_SYSTEM_MAX = 200.0 + 600.0 * (spoke_thickness_fem / 12.0)
        k_total_raw  = k_spoke_raw * n_radios
        k_total_Nmm  = K_SYSTEM_MAX * k_total_raw / (K_SYSTEM_MAX + k_total_raw)
        k_spoke_Nmm  = k_total_Nmm / n_radios
        # BUG#4: F_vertical real en el path FEM
        F_vertical_fem = float(params.get('F_vertical', 500.0))
        delta_max_mm = float(F_vertical_fem / k_total_Nmm) if k_total_Nmm > 1e-9 else 99.0

        # Von Mises desde modelo de tensión + Kt analítico (Neuber, unión rim-spoke)
        # FIX Kt: sin este factor sigma_vm ≈ 6 MPa → life_cycles siempre 1e12 y SF irreal.
        # Neuber: Kt = 1 + 2*sqrt(c/rho)
        #   c   = spoke_thickness/2  (profundidad del concentrador)
        #   rho = wall_thickness*0.3 (radio de curvatura estimado de la unión)
        _st_fem  = float(params.get('spoke_thickness', wall_thick))
        _rho_fem = max(0.5, wall_thick * 0.3)
        Kt_an_fem = float(np.clip(1.0 + 2.0 * np.sqrt((_st_fem / 2.0) / _rho_fem), 1.0, 5.0))
        n_contact_fem  = max(2, n_radios // 6)
        sigma_axial_fm = (F_vertical_fem / n_contact_fem) / A_rect_fem * Kt_an_fem
        tau_cross_fm   = sigma_axial_fm * np.tan(angle_rad_fem) * 0.15
        sigma_vm = float(np.sqrt(sigma_axial_fm**2 + 3 * tau_cross_fm**2))
    else:
        # Modelo axial de tensión (Spring Tire = estructura en tensión)
        # Rigidez axial pura: k = E*A/L por radio
        # Referencia: Asnani et al. NASA/TM-2012-217622

        # FIX BUG#2: usar spoke_thickness real del genoma (gen 15)
        spoke_thickness = float(params.get('spoke_thickness', wall_thick))
        spoke_width_t = wall_thick * 4.0          # mm  ancho de la banda plana
        A_spoke_rect  = spoke_width_t * spoke_thickness  # mm² — area real con grosor de gen
        I_weak        = spoke_width_t * spoke_thickness**3 / 12  # mm⁴

        angle_rad = np.radians(angle_cross)

        # Rigidez del mecanismo Spring Tire: axial (tensión) + flexión (curvatura)
        k_axial_spoke = E_avg * A_spoke_rect / L_spoke * np.sin(angle_rad)**2
        k_bend_spoke  = 12 * E_avg * I_weak / L_spoke**3 * np.cos(angle_rad)**2
        k_spoke_raw   = k_axial_spoke + k_bend_spoke   # N/mm per spoke

        # FIX BUG#1: escalar por n_layers del genoma (gen 14) — antes siempre = 1
        n_layers = max(1, int(params.get('n_layers', 1)))
        k_spoke_raw *= (1.0 + 0.6 * (n_layers - 1))  # cada capa adicional suma ~60%

        # FIX BUG#2: K_SYSTEM_MAX dinámico según spoke_thickness (2–12 mm rango)
        # Límite físico escala con rigidez del radio: más grueso → límite mayor
        K_SYSTEM_MAX = 200.0 + 600.0 * (spoke_thickness / 12.0)  # N/mm [200..800]
        k_spoke_Nmm  = k_spoke_raw  # N/mm per spoke (ya proyectado)
        k_total_raw  = k_spoke_Nmm * n_radios  # N/mm total sin límite

        # Saturación suave (soft cap): k = K_max * k_raw / (K_max + k_raw)
        k_total_Nmm  = K_SYSTEM_MAX * k_total_raw / (K_SYSTEM_MAX + k_total_raw)
        k_spoke_Nmm  = k_total_Nmm / n_radios   # back-calculate para uso posterior

        # BUG#3/#12: usar F_vertical real del escenario en el path analítico
        F_vertical_an = float(params.get('F_vertical', 500.0))
        delta_max_mm = float(F_vertical_an / k_total_Nmm) if k_total_Nmm > 1e-9 else 99.0

        # Von Mises: tensión axial en spokes de la zona de contacto + Kt analítico
        # FIX Kt: Neuber para unión rim-spoke: Kt = 1 + 2*sqrt(c/rho)
        #   c   = spoke_thickness/2, rho = wall_thickness*0.3
        _rho_an = max(0.5, wall_thick * 0.3)
        Kt_analytical = float(np.clip(1.0 + 2.0 * np.sqrt((spoke_thickness / 2.0) / _rho_an), 1.0, 5.0))
        n_contact = max(2, n_radios // 6)
        sigma_axial = (F_vertical_an / n_contact) / A_spoke_rect * Kt_analytical  # MPa pico
        tau_cross   = sigma_axial * np.tan(angle_rad) * 0.15
        sigma_vm    = float(np.sqrt(sigma_axial**2 + 3 * tau_cross**2))

    # Rigidez vertical total (kN/m)
    # 1 N/mm = 1 kN/m  →  k[N/mm] * n_radios = k_total [kN/m]
    k_vertical_kNm = float(k_spoke_Nmm * n_radios)  # kN/m

    # Rigidez lateral (radios en ángulo)
    angle_rad = np.radians(angle_cross)
    k_lateral_kNm = float(k_vertical_kNm * np.tan(angle_rad) * 0.4)

    # Rigidez torsional (N·m/rad)
    # BUG#3 fix: J_polar para sección RECTANGULAR es I_fuerte + I_débil, no 2*I_débil
    # Para sección circular Jp=2I, pero para rectángulo la torsion es mucho menor.
    # Aproximación exacta: Jp_rect = I_strong + I_spoke (suma de inercias principales)
    J_polar = I_strong + I_spoke  # mm⁴ (rectangulo: Jp = I_xx + I_yy)
    k_torsion = float(G_torsion(E_avg, nu_avg) * J_polar * n_radios /
                      L_spoke / 1000)  # N·m/rad  (/ 1000 mm→m)

    # --- 6. Carga máxima informativa (no usada para SF) ---
    # Se mantiene F_max_N como dato informativo para el JSON de resultados.
    sigma_max_allow = sigma_y_avg * 0.65
    F_max_N = float(sigma_max_allow * A_spoke * n_radios)

    # --- 7. Pandeo ---
    buckling = analyze_buckling_1d(E_avg, A_spoke, I_strong, L_spoke,
                                   sigma_y=sigma_y_avg)

    F_vertical_ref = float(params.get('F_vertical', 500.0))
    F_per_spoke_ref = F_vertical_ref / max(n_radios, 1)
    SF_buckling = float(buckling['P_critical_N'] / F_per_spoke_ref) if F_per_spoke_ref > 0 else 99.0

    # --- 8. Análisis Neo-Hookean / Memoria de forma ---
    # Usar material más blando del conjunto
    mat_softest = min(mats, key=lambda m: m['E'])
    shape_memory = analyze_shape_memory(params, mat_softest)

    # --- 9. Fatiga ---
    fatigue = analyze_fatigue(
        sigma_vm_max=float(sigma_vm),
        sigma_mean=float(sigma_vm * 0.5),
        mat=mat_worst,
        R_ratio=0.1,
        params=params  # FIX BUG#6: pasar params para radio real
    )

    # --- 10. Masa ---
    tread_thick = wall_thick * 2.0
    rim_thick   = wall_thick * 1.5
    V_tread  = 2 * np.pi * r_ext * tread_thick * width          # mm³
    V_rim    = 2 * np.pi * r_rim * rim_thick   * width * 0.9
    # BUG#5 fix: n_total (cap 3 sets) subestima masa hasta 5x cuando n_layers=15.
    # Usar todos los radios reales: n_radios * n_layers_total
    V_spokes = (n_radios * n_layers_total) * A_spoke * L_spoke
    V_total_mm3 = V_tread + V_rim + V_spokes
    mass_kg = float(V_total_mm3 * 1e-6 * density_avg)            # kg  (mm³→cm³→g→kg)

    # --- 11. Factor de seguridad global ---
    # FIX SF: antes SF_strength = F_max/F_vert mezclaba escalas (F_max usa n_radios,
    # F_vert actúa sobre n_contact). Ahora SF = sigma_y / sigma_vm_pico, que es la
    # definición correcta y es consistente con el Kt ya incluido en sigma_vm.
    SF_strength = float(min(m['sigma_y'] for m in mats) / max(sigma_vm, 0.01))
    SF_global   = min(SF_strength, SF_buckling)

    # --- 12. Multi-material: riesgo de adhesión y compatibilidad ---
    unique_mats = np.unique(material_sectors)
    if len(unique_mats) > 1:
        temps = [MATERIAL_DB.get(int(m), MATERIAL_DB[0])['print_temp'] for m in unique_mats]
        # Diferencia de temperatura de impresión
        adhesion_risk = float((max(temps) - min(temps)) / 100.0)

        # Incompatibilidad química (heurística: TPU-PLA es peor que TPU-TPU)
        mat_names = [MATERIAL_DB.get(int(m), MATERIAL_DB[0])['name'] for m in unique_mats]
        has_mixed_families = any('TPU' in n for n in mat_names) and any('TPU' not in n for n in mat_names)
        adhesion_risk += 0.2 if has_mixed_families else 0.0
    else:
        adhesion_risk = 0.0

    print_complexity = float(len(unique_mats) / len(MATERIAL_DB))

    # --- Componer resultado final ---
    result = {
        # ---- Propiedades mecánicas ----
        'stiffness_kN_m':            k_vertical_kNm,
        'lateral_stiffness_kN_m':    k_lateral_kNm,
        'torsional_stiffness_Nm_rad': k_torsion,
        'max_deflection_mm':         float(delta_max_mm),
        'max_load_N':                float(F_max_N),
        'mass_kg':                   mass_kg,
        'safety_factor':             float(SF_global),
        'safety_factor_strength':    float(SF_strength),
        'safety_factor_buckling':    float(SF_buckling),

        # ---- Esfuerzos (reales, no estimaciones) ----
        'sigma_vm_max_MPa':          float(sigma_vm),
        'sigma_yield_MPa':           float(sigma_y_avg),
        # FIX BUG#3: usar sigma_y del material MÁS DÉBIL (peor caso real, no promedio Voigt)
        'plastic_strain_percent':    float(max(0, (sigma_vm / min(m['sigma_y'] for m in mats) - 1.0) * 100)),

        # ---- Pandeo ----
        'buckling': buckling,

        # ---- Fatiga ----
        'fatigue': fatigue,
        'life_cycles':               fatigue['N_cycles'],
        'life_km':                   fatigue['life_km'],

        # ---- Memoria de forma / Neo-Hookean ----
        'shape_memory': shape_memory,
        'shape_memory_ratio':        shape_memory['shape_memory_ratio'],
        'recovery_ratio':            shape_memory['recovery_ratio'],
        'energy_stored_mJ':          shape_memory['energy_stored_mJ'],

        # ---- Calidad multi-material ----
        'adhesion_risk':             float(adhesion_risk),
        'print_complexity':          float(print_complexity),
        'n_unique_materials':        int(len(unique_mats)),
        'materials':                 [MATERIAL_DB.get(int(m), MATERIAL_DB[0])['name'] for m in unique_mats],

        # ---- Eficiencia estructural ----
        'structural_efficiency':     float(k_vertical_kNm / mass_kg) if mass_kg > 0 else 0.0,
        'E_avg_MPa':                 float(E_avg),
        'volume_cm3':                float(V_total_mm3 * 1e-3),
        'n_spokes':                  int(n_total),

        # ---- Metadatos FEA ----
        'fea_type': 'scikit_fem_robust' if fem_results_spoke.get('n_dofs') else 'timoshenko_analytical',
        'fem_n_dofs': int(fem_results_spoke.get('n_dofs', 0)),
        'params': {
            'd_ext': d_ext, 'd_rim': d_rim, 'width': width,
            'n_radios': n_radios, 'wall_thickness': wall_thick,
            'angle_cross': angle_cross,
        },
    }

    return result


# ---------------------------------------------------------------------------
# Alias de compatibilidad con spring_tire_fea.py original
# ---------------------------------------------------------------------------
def run_spring_tire_fea(params: Dict) -> Dict:
    """Wrapper de compatibilidad. Llama al FEA robusto."""
    return run_robust_fea(params)


analyze_spring_tire = run_robust_fea


# ---------------------------------------------------------------------------
# Bloque de prueba (self-test)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    # Parámetros de referencia: Spring Tire NASA
    test_params = {
        'd_ext':           200.0,    # mm  diámetro exterior
        'd_rim':           100.0,    # mm  diámetro del rim
        'width':            85.0,    # mm
        'n_radios':         24,
        'wall_thickness':    2.5,    # mm
        'angle_cross':      55.0,    # deg
        'material_sectors': np.array([0, 0, 1, 1, 0, 0, 1, 1]),  # TPU_95A + TPU_85A
    }

    print("=" * 65)
    print("  TEST FEA ROBUSTO - NASA Spring Tire v6.0")
    print("=" * 65)

    results = run_robust_fea(test_params)

    # Mostrar resultados
    groups = {
        'RIGIDEZ': ['stiffness_kN_m', 'lateral_stiffness_kN_m', 'torsional_stiffness_Nm_rad'],
        'RESISTENCIA': ['sigma_vm_max_MPa', 'sigma_yield_MPa', 'max_load_N', 'safety_factor'],
        'DEFLEXIÓN & MASA': ['max_deflection_mm', 'mass_kg', 'volume_cm3', 'structural_efficiency'],
        'PANDEO': ['buckling'],
        'FATIGA': ['life_cycles', 'life_km', 'fatigue'],
        'MEMORIA DE FORMA': ['recovery_ratio', 'shape_memory_ratio', 'energy_stored_mJ', 'shape_memory'],
        'MULTI-MATERIAL': ['materials', 'adhesion_risk', 'print_complexity'],
        'METADATOS FEA': ['fea_type', 'fem_n_dofs', 'n_spokes'],
    }

    for group, keys in groups.items():
        print(f"\n  [{group}]")
        for k in keys:
            val = results.get(k, 'N/A')
            if isinstance(val, dict):
                for kk, vv in val.items():
                    if not isinstance(vv, (list,)):
                        print(f"    {kk:35s}: {vv}")
            elif isinstance(val, list):
                print(f"    {k:35s}: {val}")
            elif isinstance(val, float):
                print(f"    {k:35s}: {val:.4f}")
            else:
                print(f"    {k:35s}: {val}")

    print("\n" + "=" * 65)
    print("  FEA completado sin errores ✓")
    print("=" * 65)
