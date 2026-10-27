#!/usr/bin/env python3
"""
validate_pareto_fem3d_fulltire.py — Validación FEM 3D llanta completa v1.0

Toma el JSON de resultados del NSGA-III y valida las mejores soluciones
del frente de Pareto usando FEM 3D de la llanta completa (tread + rim + todos
los radios), no solo un radio individual.

Diferencia clave respecto al FEM 3D del surrogate:
  - Modela la geometría completa: cilindro tread + cilindro rim + N radios
  - Aplica carga vertical real en el tread (dirección -Y, zona de contacto)
  - Fija el rim en su cara interior (condición de rueda montada en eje)
  - Calcula deflexión vertical del tread directamente (comparable con FEA 2D)
  - Calcula rigidez como k = F_vertical / delta_vertical (definición física real)
  - sigma_vm real en todo el campo, incluyendo las uniones rim-spoke y spoke-tread

Uso:
    python validate_pareto_fem3d_fulltire.py \\
        --pareto results/pareto/pareto_spring_tire_TIMESTAMP.json \\
        --top-n 20 \\
        --mesh-size 5.0 \\
        --output results/pareto/validated_fulltire_TIMESTAMP.json

Tiempo estimado:
    mesh=8mm:  ~8s/solución   → 20 soluciones ≈ 3 min
    mesh=5mm:  ~30s/solución  → 20 soluciones ≈ 10 min
    mesh=3mm:  ~120s/solución → 20 soluciones ≈ 40 min

Dependencias: gmsh, meshio, scikit-fem, scipy, numpy
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Imports opcionales
# ---------------------------------------------------------------------------
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    log.error("gmsh no disponible — instalar con: pip install gmsh")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    log.error("meshio no disponible — instalar con: pip install meshio")

try:
    from skfem import MeshTet, Basis, ElementVector, ElementTetP1, asm, condense
    from skfem.models import linear_elasticity, lame_parameters
    from scipy.sparse.linalg import spsolve
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    log.error("scikit-fem no disponible — instalar con: pip install scikit-fem")

FEM3D_AVAILABLE = GMSH_AVAILABLE and MESHIO_AVAILABLE and SKFEM_AVAILABLE

# ---------------------------------------------------------------------------
# Catálogo de materiales
# ---------------------------------------------------------------------------
MATERIAL_DB = {
    0: {'name': 'TPU_95A',    'E': 50.0,   'nu': 0.47, 'sigma_y': 25.0,  'density': 1.15},
    1: {'name': 'TPU_85A',    'E': 25.0,   'nu': 0.47, 'sigma_y': 15.0,  'density': 1.12},
    2: {'name': 'TPU_75D',    'E': 200.0,  'nu': 0.45, 'sigma_y': 40.0,  'density': 1.20},
    3: {'name': 'Nylon_PA12', 'E': 1800.0, 'nu': 0.40, 'sigma_y': 45.0,  'density': 1.01},
    4: {'name': 'ABS',        'E': 2200.0, 'nu': 0.38, 'sigma_y': 40.0,  'density': 1.05},
    5: {'name': 'PLA',        'E': 3500.0, 'nu': 0.36, 'sigma_y': 55.0,  'density': 1.25},
}


# ---------------------------------------------------------------------------
# PASO 1: Geometría 3D completa con gmsh
# ---------------------------------------------------------------------------

def build_full_tire_geometry(params: Dict, mesh_size: float = 5.0) -> str:
    """
    Genera geometría 3D de la Spring Tire COMPLETA con gmsh.

    Componentes:
      - Tread: cilindro anular (r_ext, r_tread_inner, width)
      - Rim:   cilindro anular (r_rim_outer, r_rim_inner, width*0.9)
      - Spokes: N_radios × N_layers cilindros rectangulares extruidos

    Condiciones de contorno físicas:
      - BC_RIM_FIXED:  cara interior del rim (r < r_rim_inner + tol) — empotrado
      - BC_TREAD_LOAD: arco de contacto del tread (zona inferior, |θ| < 30°) — carga -Y
      - Todo lo demás: libre

    Returns:
        path al archivo .msh temporal
    """
    d_ext       = float(params['d_ext'])
    d_rim       = float(params['d_rim'])
    width       = float(params['width'])
    wall_thick  = float(params['wall_thickness'])
    n_radios    = int(params['n_radios'])
    angle_cross = float(params['angle_cross'])
    n_layers    = max(1, int(params.get('n_layers', 1)))
    spoke_thick = float(params.get('spoke_thickness', wall_thick))

    r_ext       = d_ext / 2
    r_rim       = d_rim / 2
    tread_thick = wall_thick * 2.0
    rim_thick   = wall_thick * 1.5
    r_tread_in  = r_ext - tread_thick
    r_rim_out   = r_rim + rim_thick
    r_rim_in    = max(r_rim - rim_thick * 0.5, 10.0)
    spoke_w     = wall_thick * 4.0
    L_spoke     = max(r_tread_in - r_rim_out, 5.0)

    # Silenciar gmsh (fork-safety)
    import os as _os
    try:
        _dn = open(_os.devnull, 'w')
        _fd = _os.dup(2)
        _os.dup2(_dn.fileno(), 2)
        try:
            gmsh.finalize()
        finally:
            _os.dup2(_fd, 2)
            _os.close(_fd)
            _dn.close()
    except Exception:
        pass

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spring_tire_full")
    occ = gmsh.model.occ

    # ------------------------------------------------------------------
    # 1. TREAD — cilindro anular
    # ------------------------------------------------------------------
    # Cilindro exterior macizo
    tread_outer = occ.addCylinder(0, 0, -width/2, 0, 0, width, r_ext)
    # Cilindro a restar (hueco interior)
    tread_inner = occ.addCylinder(0, 0, -width/2 - 1, 0, 0, width + 2, r_tread_in)
    # Diferencia booleana → anillo
    tread_vols, _ = occ.cut([(3, tread_outer)], [(3, tread_inner)], removeObject=True, removeTool=True)
    tread_vol_tags = [t for _, t in tread_vols]

    # ------------------------------------------------------------------
    # 2. RIM — cilindro anular
    # ------------------------------------------------------------------
    rim_h = width * 0.9
    rim_outer = occ.addCylinder(0, 0, -rim_h/2, 0, 0, rim_h, r_rim_out)
    rim_inner = occ.addCylinder(0, 0, -rim_h/2 - 1, 0, 0, rim_h + 2, r_rim_in)
    rim_vols, _ = occ.cut([(3, rim_outer)], [(3, rim_inner)], removeObject=True, removeTool=True)
    rim_vol_tags = [t for _, t in rim_vols]

    # ------------------------------------------------------------------
    # 3. SPOKES — N_radios × N_layers cilindros rectangulares
    # ------------------------------------------------------------------
    # Capas Z distribuidas uniformemente en el ancho de la llanta
    margin = spoke_thick * 1.5
    if n_layers == 1:
        z_layers = [0.0]
    else:
        z_layers = list(np.linspace(-width/2 + margin, width/2 - margin, n_layers))

    spoke_vol_tags = []
    angle_cross_rad = np.radians(angle_cross)

    for layer_idx, z_layer in enumerate(z_layers):
        # Alternar dirección de cruce entre capas (patrón pantográfico)
        sign = 1 if layer_idx % 2 == 0 else -1
        ac = sign * angle_cross_rad

        for i in range(n_radios):
            theta = 2 * np.pi * i / n_radios

            # Punto inicial en el rim_outer (con pequeña penetración)
            pen = spoke_thick * 0.5
            p_start_r = r_rim_out - pen
            p_end_r   = r_tread_in + pen

            # Ángulo de inicio y fin del radio (con cruce)
            theta_start = theta
            theta_end   = theta + ac

            # Coordenadas 3D
            x1 = p_start_r * np.cos(theta_start)
            y1 = p_start_r * np.sin(theta_start)
            x2 = p_end_r   * np.cos(theta_end)
            y2 = p_end_r   * np.sin(theta_end)

            vec = np.array([x2 - x1, y2 - y1, 0.0])
            L   = np.linalg.norm(vec)
            if L < 1.0:
                continue
            vec_u = vec / L

            # Dirección perpendicular en XY (ancho del radio)
            perp = np.array([-vec_u[1], vec_u[0], 0.0])

            # 4 puntos de la sección transversal centrada en p_start
            mid = np.array([x1, y1, z_layer])
            hw  = spoke_w / 2.0
            ht  = spoke_thick / 2.0

            corners = [
                mid + perp * hw + np.array([0, 0,  ht]),
                mid - perp * hw + np.array([0, 0,  ht]),
                mid - perp * hw + np.array([0, 0, -ht]),
                mid + perp * hw + np.array([0, 0, -ht]),
            ]
            pts = [occ.addPoint(*c) for c in corners]
            lines = [
                occ.addLine(pts[0], pts[1]),
                occ.addLine(pts[1], pts[2]),
                occ.addLine(pts[2], pts[3]),
                occ.addLine(pts[3], pts[0]),
            ]
            wire = occ.addCurveLoop(lines)
            surf = occ.addPlaneSurface([wire])

            # Extruir en dirección del radio
            ext = occ.extrude([(2, surf)], vec[0], vec[1], vec[2])
            for dim, tag in ext:
                if dim == 3:
                    spoke_vol_tags.append(tag)

    # ------------------------------------------------------------------
    # 4. Unión booleana (fragmentar para que la malla sea conforme)
    # ------------------------------------------------------------------
    occ.synchronize()

    all_vol_tags = tread_vol_tags + rim_vol_tags + spoke_vol_tags

    if len(all_vol_tags) > 1:
        try:
            # occ.fuse une todos los sólidos en uno garantizando nodos compartidos
            # en las interfaces → malla conforme → K no singular
            base  = [(3, all_vol_tags[0])]
            tools = [(3, t) for t in all_vol_tags[1:]]
            fused, _ = occ.fuse(base, tools, removeObject=True, removeTool=True)
            all_vol_tags = [t for _, t in fused]
            log.info(f"  Boolean fuse OK -> {len(all_vol_tags)} volumen(es)")
        except Exception as ex:
            log.warning(f"  occ.fuse fallo ({ex}) — intentando fragment")
            try:
                occ.synchronize()
                vols_now = [t for _, t in gmsh.model.getEntities(3)]
                if len(vols_now) > 1:
                    frags, _ = occ.fragment(
                        [(3, vols_now[0])],
                        [(3, t) for t in vols_now[1:]],
                        removeObject=True, removeTool=True
                    )
                    all_vol_tags = [t for _, t in frags]
            except Exception as ex2:
                log.warning(f"  fragment tambien fallo ({ex2})")

    occ.synchronize()

    # ------------------------------------------------------------------
    # 5. Grupos físicos para condiciones de contorno
    # ------------------------------------------------------------------
    pts_array = np.array([[
        *gmsh.model.getValue(0, n, [])]
        for n in gmsh.model.getEntities(0)
    ]) if False else None  # no usar — usar posición geométrica en post-proceso

    # Volumen total
    all_vols_now = [t for _, t in gmsh.model.getEntities(3)]
    if all_vols_now:
        gmsh.model.addPhysicalGroup(3, all_vols_now, tag=1, name="tire_volume")

    # Superficies clasificadas por radio
    all_surfs = gmsh.model.getEntities(2)
    bc_rim_fixed = []
    bc_tread_load = []

    for dim, tag in all_surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        xc = (bb[0] + bb[3]) / 2
        yc = (bb[1] + bb[4]) / 2
        r_c = np.sqrt(xc**2 + yc**2)

        # Cara del rim interior → empotrado
        if r_c < r_rim_in + 2.0:
            bc_rim_fixed.append(tag)
        # Cara exterior del tread → zona de carga (arco inferior ≈ ±30°)
        elif r_c > r_ext - 2.0 and yc < -r_ext * 0.4:
            bc_tread_load.append(tag)

    if bc_rim_fixed:
        gmsh.model.addPhysicalGroup(2, bc_rim_fixed,  tag=10, name="bc_rim_fixed")
    if bc_tread_load:
        gmsh.model.addPhysicalGroup(2, bc_tread_load, tag=11, name="bc_tread_load")

    # ------------------------------------------------------------------
    # 6. Mallado
    # ------------------------------------------------------------------
    # Malla más fina en los spokes (zona de interés), más gruesa en tread/rim
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.4)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay 3D
    gmsh.option.setNumber("Mesh.Optimize",    1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)  # más lento, no necesario aquí

    gmsh.model.mesh.generate(3)

    tmp_path = tempfile.mktemp(suffix='.msh')
    gmsh.write(tmp_path)
    gmsh.finalize()

    return tmp_path


# ---------------------------------------------------------------------------
# PASO 2: Solver elasticidad 3D — llanta completa
# ---------------------------------------------------------------------------

def solve_full_tire_fem(msh_path: str,
                        params: Dict,
                        mat: Dict,
                        F_vertical: float = 500.0) -> Dict:
    """
    Resuelve elasticidad lineal 3D para la llanta completa.

    Condiciones de contorno físicas correctas:
      - Rim interior empotrado (rueda sobre eje)
      - Carga -Y en el arco de contacto del tread (suelo marciano)

    Outputs físicamente comparables con FEA 2D:
      - k_vertical_kNm:    F_vertical / delta_Y_tread  (rigidez real del sistema)
      - max_deflection_mm: delta_Y en el punto de contacto
      - sigma_vm_MPa:      Von Mises pico en toda la malla
      - Kt_rim, Kt_spoke:  concentración de esfuerzo en uniones
      - SF_strength:       sigma_y_min / sigma_vm_pico
    """
    d_ext      = float(params['d_ext'])
    d_rim      = float(params['d_rim'])
    r_ext      = d_ext / 2
    r_rim      = d_rim / 2
    r_rim_in   = max(r_rim - float(params['wall_thickness']) * 1.5 * 0.5, 10.0)
    n_radios   = int(params['n_radios'])

    E       = float(mat['E'])
    nu      = float(mat['nu'])
    sigma_y = float(mat['sigma_y'])

    # ---- Leer malla ----
    msh = meshio.read(msh_path)
    tet_cells = None
    for cb in msh.cells:
        if cb.type == "tetra":
            tet_cells = cb.data
            break
    if tet_cells is None:
        raise ValueError("No hay tetraedros en la malla")

    pts = msh.points  # (N_nodes, 3)
    mesh3d = MeshTet(pts.T, tet_cells.T)
    log.info(f"  Malla: {mesh3d.nvertices:,} nodos, {mesh3d.nelements:,} tets")

    # ---- Ensamblaje con TetP1 (más rápido para validación) ----
    lam, mu = lame_parameters(E, nu)
    e3d     = ElementVector(ElementTetP1())
    basis   = Basis(mesh3d, e3d)
    N_dofs  = basis.N
    log.info(f"  DOFs: {N_dofs:,}")

    K = asm(linear_elasticity(lam, mu), basis)
    f = np.zeros(N_dofs)

    # FIX: basis.dofs.nodal_dofs tiene shape (3, n_dof_nodes).
    # n_dof_nodes puede diferir de mesh3d.nvertices en algunas versiones de skfem
    # porque incluye nodos de borde/faceta. Usamos basis.doflocs (coordenadas de
    # los puntos DOF) para identificar zonas geométricas, NO pts de meshio.
    dof_nd  = basis.dofs.nodal_dofs       # (3, n_dof_nodes)
    n_dof_nodes = dof_nd.shape[1]

    # Coordenadas de los nodos DOF (pueden diferir de pts si hay nodos extra)
    # basis.doflocs shape: (3, n_dof_nodes)
    dof_coords = basis.doflocs.T          # (n_dof_nodes, 3)
    r_vals_dof    = np.sqrt(dof_coords[:, 0]**2 + dof_coords[:, 1]**2)
    theta_vals_dof = np.arctan2(dof_coords[:, 1], dof_coords[:, 0])

    # Rim interior: r < r_rim_in + tolerancia ESTRECHA
    # Solo la cara interior del rim (el eje), no todo el rim
    # Con geometría fused, r_rim_in es el radio interior real del rim
    tol_rim = max(float(params['wall_thickness']) * 0.3, 1.0)
    rim_fixed_nodes = np.where(r_vals_dof < r_rim_in + tol_rim)[0]

    # Tread zona de contacto: r ≈ r_ext, parte inferior (θ ≈ -π/2 ± 35°)
    tol_tread = float(params['wall_thickness']) * 1.0
    contact_angle = np.radians(35)
    theta_from_bottom = np.abs(theta_vals_dof - (-np.pi / 2))
    theta_from_bottom = np.minimum(theta_from_bottom, 2*np.pi - theta_from_bottom)

    tread_load_nodes = np.where(
        (r_vals_dof > r_ext - tol_tread) &
        (theta_from_bottom < contact_angle)
    )[0]

    if len(tread_load_nodes) == 0:
        log.warning("  No hay nodos en zona de contacto, usando todos los del tread exterior")
        tread_load_nodes = np.where(r_vals_dof > r_ext - tol_tread)[0]

    if len(rim_fixed_nodes) == 0:
        log.warning("  No hay nodos en rim interior, usando todos los del rim")
        rim_fixed_nodes = np.where(r_vals_dof < r_rim_in * 1.5)[0]

    # Asegurar que los índices no salgan de rango
    rim_fixed_nodes   = rim_fixed_nodes[rim_fixed_nodes < n_dof_nodes]
    tread_load_nodes  = tread_load_nodes[tread_load_nodes < n_dof_nodes]

    log.info(f"  Nodos DOF: {n_dof_nodes}, rim fijos: {len(rim_fixed_nodes)} ({len(rim_fixed_nodes)/n_dof_nodes*100:.0f}%), carga tread: {len(tread_load_nodes)}")

    # Guardia: si más del 40% de los nodos están fijados, la geometría
    # probablemente no está bien fusionada (rim y radios son cuerpos separados)
    pct_fixed = len(rim_fixed_nodes) / n_dof_nodes
    if pct_fixed > 0.40:
        log.warning(f"  ADVERTENCIA: {pct_fixed*100:.0f}% de nodos fijados — geometría posiblemente no fusionada")
        log.warning(f"  Ampliando tolerancia de rim a r < {r_rim_in:.1f}mm (solo cara interior)")
        # Forzar tolerancia mínima: solo nodos en la superficie interior real
        tol_rim_strict = 1.0
        rim_fixed_nodes = np.where(r_vals_dof < r_rim_in + tol_rim_strict)[0]
        log.info(f"  Rim fijos (corregido): {len(rim_fixed_nodes)}")

    if len(rim_fixed_nodes) == 0:
        log.error("  Sin nodos de rim para fijar — geometría inválida")
        return {}

    # ---- Vector de fuerzas: carga vertical -Y en zona de contacto ----
    if len(tread_load_nodes) > 0:
        f[dof_nd[1, tread_load_nodes]] -= F_vertical / len(tread_load_nodes)  # -Y

    # ---- Condiciones de contorno: rim empotrado ----
    fixed_dofs = np.concatenate([
        dof_nd[0, rim_fixed_nodes],
        dof_nd[1, rim_fixed_nodes],
        dof_nd[2, rim_fixed_nodes],
    ])

    # ---- Resolver ----
    K_c, f_c = condense(K, f, D=fixed_dofs, expand=False)
    free_dofs = np.setdiff1d(np.arange(N_dofs), fixed_dofs)

    log.info(f"  Resolviendo sistema {len(free_dofs):,} DOFs libres...")
    try:
        u_free = spsolve(K_c, f_c)
    except Exception as ex:
        log.error(f"  spsolve falló: {ex}")
        return {}

    if not np.all(np.isfinite(u_free)):
        log.error("  Solución contiene NaN/Inf")
        return {}

    # ---- Reconstruir campo completo ----
    u = np.zeros(N_dofs)
    u[free_dofs] = u_free

    u_x = u[dof_nd[0, :]]
    u_y = u[dof_nd[1, :]]
    u_z = u[dof_nd[2, :]]
    disp_mag = np.sqrt(u_x**2 + u_y**2 + u_z**2)

    # ---- Deflexión vertical (comparación directa con FEA 2D) ----
    if len(tread_load_nodes) > 0:
        delta_Y = float(np.abs(u_y[tread_load_nodes]).mean())
    else:
        delta_Y = float(np.abs(u_y).max())

    max_disp_mm = float(disp_mag.max())

    # ---- Rigidez vertical: k = F / delta_Y ----
    # Esta es la definición física correcta, comparable con FEA 2D
    k_vertical_kNm = float(F_vertical / delta_Y) if delta_Y > 1e-10 else 1e6

    # ---- Esfuerzos Von Mises (nodal, por gradiente de desplazamientos) ----
    # TetP1: deformación constante por elemento → acumular en nodos de la malla
    # Importante: u_x/u_y/u_z tienen n_dof_nodes entradas (de dof_nd).
    # mesh3d.t contiene índices de nodos de la malla (0..nvertices-1).
    # Para TetP1, los DOF nodales coinciden 1:1 con los nodos de la malla,
    # así que podemos usar node_ids directamente como índices en u_x/u_y/u_z
    # SIEMPRE QUE n_dof_nodes >= nvertices (que es el caso para ElementTetP1).
    n_mesh_nodes   = mesh3d.nvertices
    sigma_vm_nodal = np.zeros(n_mesh_nodes)
    count_nodal    = np.zeros(n_mesh_nodes, dtype=int)

    conn    = mesh3d.t.T        # (n_elements, 4) — índices en malla
    pts_arr = mesh3d.p.T        # (n_mesh_nodes, 3)

    # u_x/u_y/u_z son arrays de n_dof_nodes. Para TetP1 puro, los primeros
    # n_mesh_nodes entradas corresponden a los nodos de la malla.
    ux_mesh = u_x[:n_mesh_nodes]
    uy_mesh = u_y[:n_mesh_nodes]
    uz_mesh = u_z[:n_mesh_nodes]

    lam_v = E * nu / ((1 + nu) * (1 - 2*nu))
    mu_v  = E / (2 * (1 + nu))
    dN_dxi = np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]], dtype=float)

    for elem_idx in range(min(mesh3d.nelements, 50000)):
        node_ids = conn[elem_idx]
        # Guardia: todos los ids deben ser válidos
        if node_ids.max() >= n_mesh_nodes:
            continue
        ux_e = ux_mesh[node_ids]
        uy_e = uy_mesh[node_ids]
        uz_e = uz_mesh[node_ids]
        coords = pts_arr[node_ids]

        J = (coords[1:] - coords[0]).T
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            continue
        if abs(np.linalg.det(J)) < 1e-15:
            continue

        dN_dx  = dN_dxi @ J_inv.T
        eps_xx = float(dN_dx[:, 0] @ ux_e)
        eps_yy = float(dN_dx[:, 1] @ uy_e)
        eps_zz = float(dN_dx[:, 2] @ uz_e)
        gam_xy = float(dN_dx[:, 1] @ ux_e + dN_dx[:, 0] @ uy_e)
        gam_xz = float(dN_dx[:, 2] @ ux_e + dN_dx[:, 0] @ uz_e)
        gam_yz = float(dN_dx[:, 2] @ uy_e + dN_dx[:, 1] @ uz_e)

        tr_eps = eps_xx + eps_yy + eps_zz
        sx = lam_v * tr_eps + 2*mu_v * eps_xx
        sy = lam_v * tr_eps + 2*mu_v * eps_yy
        sz = lam_v * tr_eps + 2*mu_v * eps_zz
        txy = mu_v * gam_xy; txz = mu_v * gam_xz; tyz = mu_v * gam_yz

        sv = float(np.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                                   + 6*(txy**2 + txz**2 + tyz**2))))
        for nid in node_ids:
            sigma_vm_nodal[nid] += sv
            count_nodal[nid]    += 1

    mask = count_nodal > 0
    sigma_vm_nodal[mask] /= count_nodal[mask]
    sigma_vm_max = float(sigma_vm_nodal.max())
    sigma_vm_max = min(sigma_vm_max, sigma_y * 10.0)

    # ---- Kt en zonas de unión — usar coords de malla (pts_arr) ----
    r_vals_mesh   = np.sqrt(pts_arr[:, 0]**2 + pts_arr[:, 1]**2)
    r_rim_out_kt  = r_rim + float(params['wall_thickness']) * 1.5
    r_tread_in_kt = r_ext - float(params['wall_thickness']) * 2.0
    tol_kt        = float(params['wall_thickness']) * 1.5

    junction_rim_mask   = (r_vals_mesh > r_rim_out_kt - tol_kt) & (r_vals_mesh < r_rim_out_kt + tol_kt)
    junction_tread_mask = (r_vals_mesh > r_tread_in_kt - tol_kt) & (r_vals_mesh < r_tread_in_kt + tol_kt)

    sv_mean = float(sigma_vm_nodal[sigma_vm_nodal > 0].mean()) if sigma_vm_nodal.max() > 0 else 1.0
    Kt_rim   = float(sigma_vm_nodal[junction_rim_mask].max()   / max(sv_mean, 0.001)) if junction_rim_mask.any()   else 1.0
    Kt_tread = float(sigma_vm_nodal[junction_tread_mask].max() / max(sv_mean, 0.001)) if junction_tread_mask.any() else 1.0
    Kt_max   = max(Kt_rim, Kt_tread)

    # ---- Factor de seguridad ----
    sigma_y_min = float(mat.get('sigma_y_min', sigma_y))
    SF_strength = float(sigma_y_min / max(sigma_vm_max, 0.001))

    # ---- Deformación plástica ----
    plastic_strain_pct = float(max(0.0, (sigma_vm_max / sigma_y_min - 1.0) * 100.0))

    return {
        # Comparables directamente con FEA 2D
        'max_deflection_mm':    delta_Y,
        'k_vertical_kNm':      k_vertical_kNm,
        'sigma_vm_MPa':        sigma_vm_max,
        'SF_strength':         SF_strength,
        'plastic_strain_pct':  plastic_strain_pct,
        # Concentración de esfuerzo
        'Kt_rim':   Kt_rim,
        'Kt_tread': Kt_tread,
        'Kt_max':   Kt_max,
        # Desplazamientos
        'max_disp_total_mm': max_disp_mm,
        'disp_Y_contact_mm': delta_Y,
        # Metadatos
        'n_nodes':    mesh3d.nvertices,
        'n_elements': mesh3d.nelements,
        'n_dofs':     N_dofs,
        'fea_type':   'fem3d_fulltire_TetP1',
    }


# ---------------------------------------------------------------------------
# Función principal de validación
# ---------------------------------------------------------------------------

def run_fem3d_fulltire(params: Dict,
                       F_vertical: float = 500.0,
                       mesh_size: float = 5.0) -> Dict:
    """
    FEM 3D llanta completa para una solución del Pareto.

    Args:
        params:     parámetros decodificados de la Spring Tire
        F_vertical: carga vertical total (N)
        mesh_size:  tamaño de elemento gmsh (mm). Si None, se calcula automáticamente.

    Returns:
        Dict con resultados físicamente comparables con FEA 2D, o {} si falla
    """
    if not FEM3D_AVAILABLE:
        log.error("Dependencias FEM 3D no disponibles")
        return {}

    d_ext      = float(params['d_ext'])
    d_rim      = float(params['d_rim'])
    n_radios   = int(params['n_radios'])
    wall_thick = float(params['wall_thickness'])

    r_ext       = d_ext / 2
    r_rim       = d_rim / 2
    tread_thick = wall_thick * 2.0
    rim_thick   = wall_thick * 1.5
    r_tread_in  = r_ext - tread_thick
    r_rim_out   = r_rim + rim_thick
    L_spoke     = r_tread_in - r_rim_out  # espacio libre para los radios

    # --- Validación geométrica ---
    if L_spoke < 5.0:
        log.error(
            f"  Geometría inválida para FEM 3D: gap rim-tread = {L_spoke:.1f}mm < 5mm. "
            f"(d_ext={d_ext:.0f}, d_rim={d_rim:.0f}, wall={wall_thick:.1f}). "
            f"El FEA 2D maneja esto analíticamente pero gmsh no puede mallar correctamente."
        )
        return {'error': f'gap_invalido_{L_spoke:.1f}mm', 'L_spoke_mm': L_spoke}

    # --- Mesh size adaptativo ---
    # Necesitamos al menos 6 elementos a lo largo del spoke para resultados válidos.
    # mesh_size_max = L_spoke / 6
    # También respetar el límite inferior por espesor de radio.
    spoke_thick = float(params.get('spoke_thickness', wall_thick))
    ms_min_thickness = min(wall_thick, spoke_thick) / 3.0
    ms_max_gap       = L_spoke / 6.0
    ms_auto          = float(np.clip(min(ms_min_thickness, ms_max_gap), 0.5, mesh_size))

    if ms_auto < mesh_size:
        log.info(f"  Mesh adaptativo: {ms_auto:.2f}mm (gap={L_spoke:.1f}mm, solicitado={mesh_size:.1f}mm)")
        mesh_size_used = ms_auto
    else:
        mesh_size_used = mesh_size

    # Material predominante
    material_sectors = np.array(params.get('material_sectors', [0]*8), dtype=int)
    mat_ids, counts = np.unique(material_sectors, return_counts=True)
    dominant_id = int(mat_ids[np.argmax(counts)])
    mat = dict(MATERIAL_DB.get(dominant_id, MATERIAL_DB[0]))
    mat['sigma_y_min'] = float(min(MATERIAL_DB[int(m)]['sigma_y'] for m in material_sectors))

    log.info(f"  Llanta: d_ext={d_ext:.0f}mm d_rim={d_rim:.0f}mm n_radios={n_radios} gap={L_spoke:.1f}mm")
    log.info(f"  Material: {mat['name']} (E={mat['E']}MPa, sigma_y={mat['sigma_y']}MPa)")
    log.info(f"  Mesh size: {mesh_size_used:.2f}mm")

    t0 = time.time()
    try:
        msh_path = build_full_tire_geometry(params, mesh_size=mesh_size_used)
    except Exception as ex:
        log.error(f"  gmsh falló: {ex}")
        return {}
    t_mesh = time.time() - t0
    log.info(f"  Malla generada en {t_mesh:.1f}s")

    try:
        result = solve_full_tire_fem(msh_path, params, mat, F_vertical=F_vertical)
    except Exception as ex:
        log.error(f"  Solver falló: {ex}")
        result = {}
    finally:
        if os.path.exists(msh_path):
            os.unlink(msh_path)

    if not result:
        return {}

    t_total = time.time() - t0
    result['fem3d_time_s']  = t_total
    result['material']      = mat['name']
    result['F_vertical']    = F_vertical
    result['mesh_size_mm']  = mesh_size_used
    result['L_spoke_mm']    = L_spoke

    log.info(f"  FEM 3D completado en {t_total:.1f}s")
    log.info(f"    k_vertical:   {result.get('k_vertical_kNm', 0):.1f} kN/m")
    log.info(f"    deflexion:    {result.get('max_deflection_mm', 0):.4f} mm")
    log.info(f"    sigma_vm:     {result.get('sigma_vm_MPa', 0):.2f} MPa")
    log.info(f"    SF:           {result.get('SF_strength', 0):.2f}")
    log.info(f"    Kt_max:       {result.get('Kt_max', 0):.2f}")

    return result


# ---------------------------------------------------------------------------
# Script de validación de Pareto
# ---------------------------------------------------------------------------

def validate_pareto(pareto_json_path: str,
                    top_n: int = 20,
                    mesh_size: float = 5.0,
                    output_path: str = None,
                    sort_by: str = 'stiffness_kN_m') -> Dict:
    """
    Valida las top-N soluciones del frente de Pareto con FEM 3D llanta completa.

    Args:
        pareto_json_path: path al JSON del NSGA-III
        top_n:           número de soluciones a validar
        mesh_size:       tamaño de elemento gmsh
        output_path:     donde guardar los resultados (None = auto)
        sort_by:         objetivo por el que ordenar para seleccionar top-N

    Returns:
        Dict con resultados de validación
    """
    log.info("=" * 65)
    log.info("  VALIDACIÓN FEM 3D LLANTA COMPLETA")
    log.info("=" * 65)

    with open(pareto_json_path) as f:
        pareto_data = json.load(f)

    sols = pareto_data.get('soluciones', [])
    log.info(f"  Soluciones en Pareto: {len(sols)}")
    log.info(f"  Validando top-{top_n} por '{sort_by}'")

    # Escenario de carga (Perseverance por defecto)
    F_vertical = 1025.0 * 3.71 / 6 * 2.5  # ≈ 1584 N
    log.info(f"  Carga de diseño: {F_vertical:.0f} N (Perseverance × SF 2.5)")
    log.info(f"  Mesh size: {mesh_size:.1f} mm")
    log.info("=" * 65)

    # Ordenar y seleccionar top-N
    sorted_sols = sorted(
        sols,
        key=lambda s: float(s['objectives'].get(sort_by, 0)),
        reverse=True
    )
    to_validate = sorted_sols[:top_n]

    results_validated = []
    t0_total = time.time()

    for rank, sol in enumerate(to_validate):
        sol_id  = sol['id']
        params  = sol['params']
        fea2d   = sol.get('fea_results', {})
        obj     = sol['objectives']

        log.info(f"\n  [{rank+1}/{len(to_validate)}] Solución id={sol_id}")
        log.info(f"    FEA2D:  k={obj.get('stiffness_kN_m',0):.1f}kN/m  "
                 f"SF={obj.get('safety_factor',0):.2f}  "
                 f"mass={obj.get('mass_kg',0):.2f}kg")

        t_sol = time.time()
        fem3d_result = run_fem3d_fulltire(params, F_vertical=F_vertical, mesh_size=mesh_size)
        t_sol_elapsed = time.time() - t_sol

        # Calcular factores de corrección 2D→3D (ahora físicamente válidos)
        correction_factors = {}
        if fem3d_result and fea2d:
            k2d = float(fea2d.get('stiffness_kN_m', 1.0))
            k3d = float(fem3d_result.get('k_vertical_kNm', 1.0))
            correction_factors['k_correction'] = k3d / k2d if k2d > 0 else 1.0

            d2d = float(fea2d.get('max_deflection_mm', 1.0))
            d3d = float(fem3d_result.get('max_deflection_mm', 1.0))
            correction_factors['deflection_correction'] = d3d / d2d if d2d > 0 else 1.0

            s2d = float(fea2d.get('sigma_vm_max_MPa', 1.0))
            s3d = float(fem3d_result.get('sigma_vm_MPa', 1.0))
            correction_factors['sigma_correction'] = s3d / s2d if s2d > 0 else 1.0

            correction_factors['Kt_3d'] = float(fem3d_result.get('Kt_max', 1.0))

            devs = [abs(correction_factors[k] - 1.0) for k in
                    ('k_correction', 'deflection_correction', 'sigma_correction')]
            correction_factors['model_confidence'] = float(max(0.0, 1.0 - np.mean(devs)))

            if fem3d_result:
                log.info(f"    FEM3D:  k={fem3d_result.get('k_vertical_kNm',0):.1f}kN/m  "
                         f"SF={fem3d_result.get('SF_strength',0):.2f}  "
                         f"sigma={fem3d_result.get('sigma_vm_MPa',0):.2f}MPa  "
                         f"Kt={fem3d_result.get('Kt_max',0):.2f}")
                log.info(f"    Factores: k_corr={correction_factors['k_correction']:.3f}  "
                         f"d_corr={correction_factors['deflection_correction']:.3f}  "
                         f"conf={correction_factors['model_confidence']:.2f}")

        validated_sol = dict(sol)
        validated_sol['fem3d_fulltire'] = fem3d_result if fem3d_result else {'error': 'FEM 3D falló'}
        validated_sol['fem3d_fulltire_correction_factors'] = correction_factors
        validated_sol['fem3d_fulltire_rank'] = rank + 1
        results_validated.append(validated_sol)

    # Soluciones no validadas: copiar sin FEM 3D
    validated_ids = {s['id'] for s in to_validate}
    for sol in sols:
        if sol['id'] not in validated_ids:
            sol_copy = dict(sol)
            sol_copy['fem3d_fulltire'] = None
            results_validated.append(sol_copy)

    # Ordenar por id original
    results_validated.sort(key=lambda s: s['id'])

    t_total = time.time() - t0_total

    # ------------------------------------------------------------------
    # Resumen estadístico
    # ------------------------------------------------------------------
    validated_ok = [s for s in results_validated
                    if s.get('fem3d_fulltire') and 'error' not in str(s.get('fem3d_fulltire', {}))]

    log.info(f"\n{'='*65}")
    log.info(f"  RESUMEN VALIDACIÓN FEM 3D LLANTA COMPLETA")
    log.info(f"{'='*65}")
    log.info(f"  Validadas OK:   {len(validated_ok)}/{len(to_validate)}")
    log.info(f"  Tiempo total:   {t_total:.1f}s ({t_total/60:.1f} min)")
    if validated_ok:
        ks   = [s['fem3d_fulltire']['k_vertical_kNm'] for s in validated_ok]
        sfs  = [s['fem3d_fulltire']['SF_strength']    for s in validated_ok]
        sigs = [s['fem3d_fulltire']['sigma_vm_MPa']   for s in validated_ok]
        defs = [s['fem3d_fulltire']['max_deflection_mm'] for s in validated_ok]
        log.info(f"  k_vertical:    {min(ks):.1f} – {max(ks):.1f} kN/m")
        log.info(f"  deflexión:     {min(defs):.4f} – {max(defs):.4f} mm")
        log.info(f"  sigma_vm:      {min(sigs):.2f} – {max(sigs):.2f} MPa")
        log.info(f"  SF_strength:   {min(sfs):.2f} – {max(sfs):.2f}")

    # ------------------------------------------------------------------
    # Guardar JSON
    # ------------------------------------------------------------------
    output_data = dict(pareto_data)
    output_data['soluciones'] = results_validated
    output_data['fem3d_fulltire_validation'] = {
        'timestamp':      datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_validated':    len(validated_ok),
        'top_n':          top_n,
        'sort_by':        sort_by,
        'mesh_size_mm':   mesh_size,
        'F_vertical_N':   F_vertical,
        'total_time_s':   t_total,
        'fea_type':       'fem3d_fulltire_TetP1',
    }

    if output_path is None:
        base = os.path.splitext(pareto_json_path)[0]
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{base}_validated_fulltire_{ts}.json"

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):  return obj.tolist()
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            return super().default(obj)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=_NpEncoder)

    log.info(f"\n  Resultado guardado: {output_path}")
    log.info("=" * 65)

    return output_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validación FEM 3D llanta completa para soluciones del Pareto'
    )
    parser.add_argument('--pareto',     type=str, required=True,
                        help='Path al JSON del Pareto del NSGA-III')
    parser.add_argument('--top-n',      type=int, default=20,
                        help='Número de soluciones a validar (default: 20)')
    parser.add_argument('--mesh-size',  type=float, default=5.0,
                        help='Tamaño de elemento gmsh en mm (default: 5.0). '
                             'Mayor = más rápido, menos preciso.')
    parser.add_argument('--output',     type=str, default=None,
                        help='Path de salida para el JSON validado (default: auto)')
    parser.add_argument('--sort-by',    type=str, default='stiffness_kN_m',
                        choices=['stiffness_kN_m', 'safety_factor', 'life_cycles',
                                 'recovery_ratio', 'mass_kg', 'energy_stored_mJ'],
                        help='Objetivo para seleccionar top-N (default: stiffness_kN_m)')

    args = parser.parse_args()

    if not FEM3D_AVAILABLE:
        log.error("Dependencias no disponibles. Instalar: gmsh meshio scikit-fem scipy")
        sys.exit(1)

    if not os.path.exists(args.pareto):
        log.error(f"Archivo no encontrado: {args.pareto}")
        sys.exit(1)

    validate_pareto(
        pareto_json_path=args.pareto,
        top_n=args.top_n,
        mesh_size=args.mesh_size,
        output_path=args.output,
        sort_by=args.sort_by,
    )


if __name__ == '__main__':
    main()
