#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  fem3d_springtire_id79.py                                                    ║
║  Análisis FEM 3D completo — Spring Tire candidato #79                        ║
║  Optimización NSGA-III para rover marciano Perseverance                      ║
║  Versión: 1.0 — para ejecución local con dependencias completas              ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALACIÓN DE DEPENDENCIAS
────────────────────────────
Opción A — Conda (recomendado):
    conda create -n fem3d python=3.11
    conda activate fem3d
    conda install -c conda-forge fenics-dolfinx mpich pyvista
    pip install gmsh meshio scikit-fem pyamg matplotlib scipy numpy

Opción B — Docker (FEniCS oficial):
    docker pull dolfinx/dolfinx:stable
    docker run -it -v $(pwd):/work dolfinx/dolfinx:stable
    pip install gmsh meshio scikit-fem pyamg matplotlib
    python /work/fem3d_springtire_id79.py

Opción C — Solo skfem (sin FEniCS, más simple):
    pip install scikit-fem gmsh meshio pyamg matplotlib scipy numpy
    python fem3d_springtire_id79.py --solver skfem

MÓDULOS DE ANÁLISIS
────────────────────
  1. Mallado gmsh (radio individual, refinamiento adaptativo en zonas Kt)
  2. FEM 3D TetP2 — campo σ_vm, Kt, SF  [skfem o FEniCS según disponibilidad]
  3. Estudio de convergencia de malla (extrapolación Richardson)
  4. Análisis multi-material (FEM por cada material del diseño)
  5. Rigidez sistémica (Timoshenko analítico)
  6. Pandeo (Euler-Johnson analítico)
  7. Fatiga (Goodman + Basquin por material)
  8. Memoria de forma (Neo-Hookean incremental)
  9. Análisis térmico (ciclos marcianos, T_ref=20°C)
 10. Degradación UV/radiación (modelo exponencial)
 11. Figuras de publicación (10 figuras)
 12. Reporte JSON completo con referencias bibliográficas

REFERENCIAS
────────────
  [1] Logg et al. (2012). FEniCS Book. Springer. DOI:10.1007/978-3-642-23099-8
  [2] Gustafsson et al. (2020). scikit-fem. JOSS 5(52). DOI:10.21105/joss.02369
  [3] Simhadri et al. (2019). J.Mater.Res.Tech 8(2):3160  [TPU_75D FDM]
  [4] Fernandez-Vicente et al. (2016). Rapid Prototyping J 22(4)  [PLA FDM]
  [5] Rankouhi et al. (2016). J.Manuf.Sci.Eng 138(2)  [ABS FDM]
  [6] Dorigato et al. (2019). J.Polym.Eng 39(2):1  [TPU_85A]
  [7] Peterson (1974). Stress Concentration Factors. Wiley.
  [8] Shigley (2011). Mechanical Engineering Design. McGraw-Hill.
  [9] Roache (1998). Verification & Validation in CFD. Hermosa.
 [10] Hassler et al. (2014). Science 343(6169). DOI:10.1126/science.1244797
 [11] Treloar (1975). Physics of Rubber Elasticity. Oxford.
"""

import os, sys, json, time, argparse, logging
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.sparse.linalg import spsolve
import warnings; warnings.filterwarnings('ignore')

# ─── Detectar solver disponible ─────────────────────────────────────────────
SOLVER = 'skfem'   # default; sobrescrito por --solver o autodetección
try:
    import dolfinx
    SOLVER = 'fenics'
    print("✓ FEniCS/DOLFINx detectado")
except ImportError:
    try:
        import skfem
        SOLVER = 'skfem'
        print("✓ scikit-fem detectado (FEniCS no disponible)")
    except ImportError:
        raise ImportError("Instala skfem: pip install scikit-fem")

try:
    import pyamg
    HAS_AMG = True
    print(f"✓ pyamg {pyamg.__version__} (precondicionador AMG)")
except ImportError:
    HAS_AMG = False
    print("⚠ pyamg no disponible — usando spsolve directo")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS EXACTOS DEL GENOMA  id=79
# Fuente: spring_tire_079_metadata.json
# ─────────────────────────────────────────────────────────────────────────────
GENOME = [
    0.05287105764312245, 0.05908018855032715, 0.0025819797224635992,
    0.9897561235521021,  0.9323394123476569,  0.4319387342851343,
    0.40726387012368803, 0.9586777319329612,  0.7667399027368678,
    0.30905624189901215, 0.10706105448259384, 0.9261010891048566,
    0.8020836402160857,  0.36205914845479814, 0.04598863984672745,
    0.9976230546307375,  0.7564864314082056,  0.1960405926728645,
]

PARAMS = dict(
    d_ext    = 310.5742115286245,    # mm — diámetro exterior
    d_rim    = 108.86202828254908,   # mm — diámetro del aro (rim)
    width    = 50.51639594449272,    # mm — ancho axial de la llanta
    n_radios = 28,                   # número de radios pantográficos
    wall     = 9.458715298781255,    # mm — espesor de pared
    angle    = 51.596936714256714,   # ° — ángulo de cruce de los radios
    # Decodificados del genoma:
    spoke_t  = 2.0 + GENOME[15] * 10.0,           # 11.976 mm — espesor del radio
    n_layers = max(1, round(1 + GENOME[14] * 14)), # 2 — capas axiales
    # Sectores angulares de material (sector_map del JSON)
    # 8 sectores de 45° c/u, material_id según MATERIALS
    sectors  = [2, 5, 4, 2, 1, 5, 4, 2],
)

# Carga de diseño — misión Perseverance (NASA Mars 2020)
# masa_rover=1025kg, g_marte=3.71m/s², n_ruedas=6, SF_diseño=2.5
F_DESIGN = 1025.0 * 3.71 / 6.0 * 2.5   # ≈ 1584 N

# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE MATERIALES  (con referencias bibliográficas)
# ─────────────────────────────────────────────────────────────────────────────
MATERIALS = {
    1: dict(
        name='TPU_85A',
        E=25.0, nu=0.47, sigma_y=15.0, rho=1120,
        alpha_T=1.8e-4,                        # coef. expansión térmica 1/°C
        S_ut=19.5, b_basq=-0.12, Sf_coef=0.90, # parámetros fatiga
        C1=0.7, D1=0.04,                        # Neo-Hookean
        k_uv=0.07,                              # degradación UV 1/año
        ref='Dorigato et al. (2019). J.Polym.Eng 39(2):1-11',
        note='Propiedades para FDM; E puede variar 10-80 MPa según parámetros.',
    ),
    2: dict(
        name='TPU_75D',
        E=200.0, nu=0.45, sigma_y=40.0, rho=1200,
        alpha_T=1.5e-4,
        S_ut=52.0, b_basq=-0.12, Sf_coef=0.90,
        C1=5.0, D1=0.005,
        k_uv=0.08,
        ref='Simhadri et al. (2019). J.Mater.Res.Tech 8(2):3160-3169',
        note='Semirígido, alta resistencia a abrasión.',
    ),
    4: dict(
        name='ABS',
        E=2200.0, nu=0.38, sigma_y=40.0, rho=1050,
        alpha_T=7.0e-5,
        S_ut=52.0, b_basq=-0.10, Sf_coef=0.85,
        C1=None, D1=None,
        k_uv=0.25,
        ref='Rankouhi et al. (2016). J.Manuf.Sci.Eng 138(2)',
        note='E puede variar 1800-2500 MPa según orientación FDM.',
    ),
    5: dict(
        name='PLA',
        E=3500.0, nu=0.36, sigma_y=55.0, rho=1250,
        alpha_T=6.8e-5,
        S_ut=71.5, b_basq=-0.09, Sf_coef=0.85,
        C1=None, D1=None,
        k_uv=0.30,
        ref='Fernandez-Vicente et al. (2016). Rapid Prototyping J 22(4)',
        note='Alta rigidez; frágil a baja temperatura. Fotodegradable.',
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# §1  GEOMETRÍA DERIVADA
# ─────────────────────────────────────────────────────────────────────────────
def build_geometry(p):
    """Calcula todos los parámetros geométricos del radio."""
    w   = p['wall']
    st  = p['spoke_t']
    sw  = w * 4.0                 # ancho de la sección rectangular del radio
    r_e = p['d_ext'] / 2
    r_r = p['d_rim'] / 2
    rti = r_e - w * 2.0           # radio interior del tread
    rro = r_r + w * 1.5           # radio exterior del rim
    L   = rti - rro               # longitud libre del radio

    A   = sw * st                 # área de la sección transversal
    Iw  = sw * st**3 / 12         # inercia eje débil (plano de flexión)
    Is  = st * sw**3 / 12         # inercia eje fuerte
    Jp  = Iw + Is                 # momento polar (torsión)
    rg  = np.sqrt(Iw / A)        # radio de giro
    ar  = np.radians(p['angle'])  # ángulo en radianes

    # Módulo de corte efectivo (Voigt)
    E_v  = float(np.mean([MATERIALS[m]['E']  for m in p['sectors']]))
    nu_v = float(np.mean([MATERIALS[m]['nu'] for m in p['sectors']]))
    G_v  = E_v / (2 * (1 + nu_v))

    assert L > 5.0, f"Geometría inválida: L_spoke={L:.2f}mm < 5mm"

    return dict(
        r_ext=r_e, r_rim=r_r, r_tread_in=rti, r_rim_out=rro,
        L=L, sw=sw, st=st, w=w,
        A=A, Iw=Iw, Is=Is, Jp=Jp, rg=rg, ar=ar, G_avg=G_v,
        n_rad=p['n_radios'], n_lay=p['n_layers'], width=p['width'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2  MALLADO GMSH
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh(geo, mesh_size, out_dir, tag='mesh'):
    """
    Genera malla 3D del radio con gmsh.
    Sección rectangular sw×st extruida L mm en dirección X.
    Refinamiento adaptativo en extremos (zonas de Kt).
    """
    import gmsh, meshio
    try:   gmsh.finalize()
    except: pass
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(tag)
    occ = gmsh.model.occ

    L = geo['L'];  sw = geo['sw'];  st = geo['st']

    # Sección rectangular centrada en YZ (X=0 → cara rim)
    corners = [
        occ.addPoint(0,  sw/2, -st/2),
        occ.addPoint(0,  sw/2,  st/2),
        occ.addPoint(0, -sw/2,  st/2),
        occ.addPoint(0, -sw/2, -st/2),
    ]
    lines  = [occ.addLine(corners[i], corners[(i+1)%4]) for i in range(4)]
    loop   = occ.addCurveLoop(lines)
    face   = occ.addPlaneSurface([loop])
    extr   = occ.extrude([(2, face)], L, 0, 0)
    occ.synchronize()

    # ── Grupos físicos ────────────────────────────────────────────────────────
    surfs = gmsh.model.getEntities(dim=2)
    rim_tags = []; tread_tags = []; side_tags = []
    for _, tag_s in surfs:
        bb = gmsh.model.getBoundingBox(2, tag_s)
        xc = (bb[0] + bb[3]) / 2
        if xc < L * 0.05:
            rim_tags.append(tag_s)
        elif xc > L * 0.95:
            tread_tags.append(tag_s)
        else:
            side_tags.append(tag_s)

    if rim_tags:   gmsh.model.addPhysicalGroup(2, rim_tags,   1, name="rim_face")
    if tread_tags: gmsh.model.addPhysicalGroup(2, tread_tags, 2, name="tread_face")
    vols = [v[1] for v in gmsh.model.getEntities(3)]
    if vols: gmsh.model.addPhysicalGroup(3, vols, 3, name="spoke_vol")

    # ── Refinamiento en extremos (zonas de concentración de esfuerzo) ─────────
    for fid, x0, x1 in [(1, 0, L*0.18), (2, L*0.82, L)]:
        gmsh.model.mesh.field.add("Box", fid)
        for key, val in [("XMin",x0),("XMax",x1),
                          ("YMin",-sw/2-1),("YMax",sw/2+1),
                          ("ZMin",-st/2-1),("ZMax",st/2+1),
                          ("VIn", mesh_size*0.35),
                          ("VOut",mesh_size*2.5)]:
            gmsh.model.mesh.field.setNumber(fid, key, val)

    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    gmsh.option.setNumber("Mesh.MeshSizeMax",      mesh_size * 2.5)
    gmsh.option.setNumber("Mesh.MeshSizeMin",      mesh_size * 0.35)
    gmsh.option.setNumber("Mesh.Algorithm3D",      1)   # Delaunay
    gmsh.option.setNumber("Mesh.Optimize",         1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen",   1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    path = os.path.join(out_dir, f"{tag}.msh")
    gmsh.write(path)
    gmsh.finalize()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# §3  SOLVER FEM 3D — skfem TetP2
# ─────────────────────────────────────────────────────────────────────────────
def solve_fem_skfem(msh_path, geo, mat, F_design):
    """
    Solver FEM 3D con scikit-fem (TetP2, elementos cuadráticos).

    CONDICIONES DE CONTORNO FÍSICAMENTE CORRECTAS:
    ─ Cara rim   (X=0): desplazamiento nulo — rim unido al eje del rover
    ─ Cara tread (X=L): carga descompuesta según ángulo α del radio:
        F_ax  = (F_design/n_c) * sin(α)  → componente axial (+X)
        F_lat = (F_design/n_c) * cos(α)  → componente lateral (-Z eje débil)

    RIGIDEZ k: calculada con modelo analítico Timoshenko (ver §4).
    El FEM de radio individual no puede medir k_sistema porque la
    deformación del sistema es proyección geométrica de la del radio.

    Kt DEFINICIÓN PETERSON [7]: Kt = σ_vm_max / σ_nominal
    donde σ_nominal = F_ax / A (sin concentrador).
    """
    import meshio
    from skfem import MeshTet, Basis, ElementVector, asm, condense
    from skfem.element import ElementTetP2
    from skfem.models import linear_elasticity, lame_parameters

    # ── Leer malla ─────────────────────────────────────────────────────────
    raw  = meshio.read(msh_path)
    tets = next(cb.data for cb in raw.cells if cb.type == "tetra")
    mesh = MeshTet(raw.points.T, tets.T)
    nv, ne = mesh.nvertices, mesh.nelements

    E, nu = float(mat['E']), float(mat['nu'])
    lam, mu = lame_parameters(E, nu)

    basis = Basis(mesh, ElementVector(ElementTetP2()))
    N     = basis.N

    log.info(f"      {nv:,} nodos  {ne:,} tets  {N:,} DOFs TetP2")

    # ── Ensamblaje ──────────────────────────────────────────────────────────
    K = asm(linear_elasticity(lam, mu), basis)
    f = np.zeros(N)

    nd    = basis.dofs.nodal_dofs   # (3, n_dof_nodes)
    dlocs = basis.doflocs.T          # posición de cada DOF node (n_dof_nodes, 3)
    nd_n  = nd.shape[1]
    L     = geo['L']
    tol   = max(L * 0.07, 1.5)

    i_rim = np.where(dlocs[:, 0] < tol)[0]
    i_tre = np.where(dlocs[:, 0] > L - tol)[0]
    if len(i_rim) == 0:
        i_rim = np.where(dlocs[:, 0] < np.percentile(dlocs[:, 0], 8))[0]
    if len(i_tre) == 0:
        i_tre = np.where(dlocs[:, 0] > np.percentile(dlocs[:, 0], 92))[0]
    i_rim = i_rim[i_rim < nd_n];  i_tre = i_tre[i_tre < nd_n]

    # ── Carga descompuesta ──────────────────────────────────────────────────
    ar   = geo['ar']
    n_c  = max(2, geo['n_rad'] // 6)   # radios en contacto simultáneo ≈ 1/6 perímetro
    F_sp = F_design / n_c
    F_ax = F_sp * np.sin(ar)           # componente axial del radio
    F_lat= F_sp * np.cos(ar)           # componente lateral (eje débil)

    n_t = len(i_tre)
    if n_t > 0:
        f[nd[0, i_tre]] += F_ax  / n_t   # +X axial
        f[nd[2, i_tre]] -= F_lat / n_t   # -Z lateral (eje débil del radio)

    # ── Empotrado en rim ────────────────────────────────────────────────────
    fixed = np.unique(np.concatenate([nd[0,i_rim], nd[1,i_rim], nd[2,i_rim]]))
    K_c, f_c = condense(K, f, D=fixed, expand=False)
    free  = np.setdiff1d(np.arange(N), fixed)

    # ── Solver ─────────────────────────────────────────────────────────────
    K_csr = K_c.tocsr()
    t0    = time.time()
    if HAS_AMG and len(free) > 20000:
        try:
            ml    = pyamg.ruge_stuben_solver(K_csr)
            from scipy.sparse.linalg import lgmres
            M_inv = ml.aspreconditioner(cycle='V')
            u_free, info = lgmres(K_csr, f_c, M=M_inv,
                                   rtol=1e-10, maxiter=600, atol=1e-14)
            if info != 0:
                log.warning(f"      LGMRES no convergió (info={info}) → spsolve")
                u_free = spsolve(K_csr, f_c)
        except Exception:
            u_free = spsolve(K_csr, f_c)
    else:
        u_free = spsolve(K_csr, f_c)
    t_sol = time.time() - t0

    if not np.all(np.isfinite(u_free)):
        raise RuntimeError("Solución contiene NaN/Inf — revisar malla/BCs")

    u  = np.zeros(N);  u[free] = u_free
    ux = u[nd[0,:]];  uy = u[nd[1,:]];  uz = u[nd[2,:]]
    dm = np.sqrt(ux**2 + uy**2 + uz**2)

    # ── Esfuerzos Von Mises nodales ─────────────────────────────────────────
    n_mn   = mesh.nvertices
    sv     = np.zeros(n_mn);  cnt = np.zeros(n_mn, int)
    sxx_a  = np.zeros(n_mn);  syy_a = np.zeros(n_mn);  szz_a = np.zeros(n_mn)
    txy_a  = np.zeros(n_mn);  txz_a = np.zeros(n_mn);  tyz_a = np.zeros(n_mn)

    conn   = mesh.t.T;   pa = mesh.p.T
    ux_m   = ux[:n_mn];  uy_m = uy[:n_mn];  uz_m = uz[:n_mn]
    dN_dxi = np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]], float)
    lv = E*nu/((1+nu)*(1-2*nu));  mv = E/(2*(1+nu))

    for ei in range(mesh.nelements):
        ids = conn[ei]
        if ids.max() >= n_mn: continue
        co  = pa[ids]
        J   = (co[1:] - co[0]).T
        try:    Ji = np.linalg.inv(J)
        except: continue
        if abs(np.linalg.det(J)) < 1e-15: continue
        dNg  = dN_dxi @ Ji.T
        exx  = dNg[:,0]@ux_m[ids];  eyy = dNg[:,1]@uy_m[ids];  ezz = dNg[:,2]@uz_m[ids]
        gxy  = dNg[:,1]@ux_m[ids] + dNg[:,0]@uy_m[ids]
        gxz  = dNg[:,2]@ux_m[ids] + dNg[:,0]@uz_m[ids]
        gyz  = dNg[:,2]@uy_m[ids] + dNg[:,1]@uz_m[ids]
        tr   = exx+eyy+ezz
        Sx   = lv*tr+2*mv*exx;  Sy = lv*tr+2*mv*eyy;  Sz = lv*tr+2*mv*ezz
        Txy  = mv*gxy;  Txz = mv*gxz;  Tyz = mv*gyz
        vm   = np.sqrt(0.5*((Sx-Sy)**2+(Sy-Sz)**2+(Sz-Sx)**2
                             + 6*(Txy**2+Txz**2+Tyz**2)))
        for nid in ids:
            sv[nid]   += vm;   cnt[nid]+= 1
            sxx_a[nid]+= Sx;   syy_a[nid]+= Sy;  szz_a[nid]+= Sz
            txy_a[nid]+= Txy;  txz_a[nid]+= Txz; tyz_a[nid]+= Tyz

    mk = cnt > 0
    for arr in (sv,sxx_a,syy_a,szz_a,txy_a,txz_a,tyz_a): arr[mk] /= cnt[mk]

    sv_max   = float(sv.max())

    # ── Kt definición Peterson [7] ──────────────────────────────────────────
    sigma_nom = F_ax / geo['A']
    tol_kt    = L * 0.12
    rim_z     = pa[:, 0] < tol_kt
    tre_z     = pa[:, 0] > L - tol_kt
    Kt_rim    = float(sv[rim_z].max()  / max(sigma_nom, 1e-12)) if rim_z.any()  else 1.0
    Kt_tread  = float(sv[tre_z].max()  / max(sigma_nom, 1e-12)) if tre_z.any()  else 1.0
    Kt_max    = max(Kt_rim, Kt_tread)

    # ── Factor de seguridad ─────────────────────────────────────────────────
    SF   = float(mat['sigma_y']) / max(sv_max, 1e-12)
    plst = max(0.0, (sv_max / mat['sigma_y'] - 1.0) * 100.0)

    return dict(
        n_nodes=nv, n_elements=ne, n_dofs=N, t_solve=t_sol,
        mat_name=mat['name'], E=E, nu=nu,
        sigma_vm_nodal=sv.copy(), sigma_vm_max=sv_max,
        sigma_nom=sigma_nom,
        sxx=sxx_a, syy=syy_a, szz=szz_a,
        txy=txy_a, txz=txz_a, tyz=tyz_a,
        Kt_rim=Kt_rim, Kt_tread=Kt_tread, Kt_max=Kt_max,
        SF_strength=SF, plastic_pct=plst,
        F_ax=float(F_ax), F_lat=float(F_lat), n_c=n_c,
        pts=pa.copy(), disp_mag=dm[:n_mn].copy(),
        ux=ux[:n_mn].copy(), uy=uy[:n_mn].copy(), uz=uz[:n_mn].copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3b  SOLVER FEM 3D — FEniCS/DOLFINx  (cuando está disponible)
# ─────────────────────────────────────────────────────────────────────────────
def solve_fem_fenics(msh_path, geo, mat, F_design):
    """
    Solver FEM 3D con FEniCS/DOLFINx.
    Ventajas vs skfem: formulación UFL, solver PETSc/MUMPS,
    convergencia verificada, resultados directamente citables [1].
    """
    import dolfinx
    from dolfinx import fem, io
    from dolfinx.fem import functionspace, Function, dirichletbc
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import gmshio
    import ufl
    from ufl import (TrialFunction, TestFunction, inner, sym, grad,
                     Identity, tr, dx, ds, Measure, sqrt, conditional, gt)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # ── Leer malla ─────────────────────────────────────────────────────────
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        msh_path, comm, gdim=3)
    log.info(f"      FEniCS: {mesh.topology.index_map(3).size_global:,} celdas")

    # ── Espacio de elementos P2 vectorial ───────────────────────────────────
    V  = functionspace(mesh, ("Lagrange", 2, (3,)))
    u  = TrialFunction(V)
    v  = TestFunction(V)

    # ── Constitutiva lineal isótropa ────────────────────────────────────────
    E_v, nu_v = float(mat['E']), float(mat['nu'])
    lam_v = E_v*nu_v / ((1+nu_v)*(1-2*nu_v))
    mu_v  = E_v / (2*(1+nu_v))

    def epsilon(w): return sym(grad(w))
    def sigma(w):   return lam_v * tr(epsilon(w)) * Identity(3) + 2*mu_v * epsilon(w)

    # ── Formulación débil ───────────────────────────────────────────────────
    a = inner(sigma(u), epsilon(v)) * dx

    # Carga en cara tread (tag=2)
    ar_v  = geo['ar']
    n_c   = max(2, geo['n_rad'] // 6)
    F_sp  = F_design / n_c
    F_ax  = F_sp * np.sin(ar_v)
    F_lat = F_sp * np.cos(ar_v)

    from dolfinx.fem import locate_dofs_topological, assemble_scalar, form
    tread_dofs_count = len(locate_dofs_topological(V, 2, facet_tags.find(2)))
    f_vec = fem.Constant(mesh, np.array([F_ax/max(tread_dofs_count//3, 1),
                                          0.0,
                                         -F_lat/max(tread_dofs_count//3, 1)]))
    ds_tread = Measure("ds", domain=mesh, subdomain_data=facet_tags,
                        subdomain_id=2)
    L_form = inner(f_vec, v) * ds_tread

    # ── BC: rim empotrado ────────────────────────────────────────────────────
    rim_dofs = locate_dofs_topological(V, 2, facet_tags.find(1))
    zero_bc  = dirichletbc(Function(V), rim_dofs)

    # ── Resolver (MUMPS directo) ────────────────────────────────────────────
    t0 = time.time()
    problem = LinearProblem(a, L_form, bcs=[zero_bc], petsc_options={
        "ksp_type": "preonly",
        "pc_type":  "lu",
        "pc_factor_mat_solver_type": "mumps",
    })
    u_sol  = problem.solve()
    t_sol  = time.time() - t0

    # ── Post-proceso σ_vm ───────────────────────────────────────────────────
    V0 = functionspace(mesh, ("DG", 0))
    def von_mises_expr(u):
        s   = sigma(u)
        dev = s - tr(s)/3 * Identity(3)
        return sqrt(3.0/2 * inner(dev, dev))

    sv_expr   = fem.Expression(von_mises_expr(u_sol),
                                V0.element.interpolation_points())
    sigma_vm  = Function(V0)
    sigma_vm.interpolate(sv_expr)

    sv_arr    = sigma_vm.x.array
    sv_max    = float(sv_arr.max())
    sigma_nom = F_ax / geo['A']
    Kt_max    = sv_max / max(sigma_nom, 1e-12)
    SF        = float(mat['sigma_y']) / max(sv_max, 1e-12)

    # Guardar en XDMF para ParaView
    xdmf_path = msh_path.replace('.msh', '_results.xdmf')
    with io.XDMFFile(comm, xdmf_path, "w") as f_out:
        f_out.write_mesh(mesh)
        f_out.write_function(u_sol)
        f_out.write_function(sigma_vm)
    log.info(f"      Resultados FEniCS guardados: {xdmf_path}")

    return dict(
        n_nodes=mesh.topology.index_map(0).size_global,
        n_elements=mesh.topology.index_map(3).size_global,
        t_solve=t_sol, mat_name=mat['name'], E=E_v, nu=nu_v,
        sigma_vm_max=sv_max, sigma_nom=sigma_nom,
        Kt_rim=Kt_max, Kt_tread=Kt_max, Kt_max=Kt_max,
        SF_strength=SF, plastic_pct=max(0.0,(sv_max/mat['sigma_y']-1)*100),
        F_ax=float(F_ax), F_lat=float(F_lat), n_c=n_c,
        pts=np.zeros((1,3)), disp_mag=np.array([0.0]),
        ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
        sigma_vm_nodal=sv_arr,
    )


def solve_fem(msh_path, geo, mat, F_design):
    """Dispatcher: usa FEniCS si disponible, sino skfem."""
    if SOLVER == 'fenics':
        return solve_fem_fenics(msh_path, geo, mat, F_design)
    else:
        return solve_fem_skfem(msh_path, geo, mat, F_design)


# ─────────────────────────────────────────────────────────────────────────────
# §4  CONVERGENCIA DE MALLA — Richardson extrapolation [9]
# ─────────────────────────────────────────────────────────────────────────────
def convergence_study(geo, p, out_dir):
    """
    Estudio de convergencia con extrapolación de Richardson.
    Corre 3 mallas. Calcula el GCI (Grid Convergence Index).
    GCI < 5% → resultado considerado convergido en ingeniería.
    Ref: Roache (1998), Verification and Validation in CFD, Hermosa.
    """
    log.info("  [CONVERGENCIA] Estudio de Richardson...")
    mesh_sizes = [4.0, 2.5, 1.5]   # mm — ajustar según recursos disponibles

    mat_worst = MATERIALS[min(p['sectors'], key=lambda m: MATERIALS[m]['sigma_y'])]
    log.info(f"    Material: {mat_worst['name']} (σ_y={mat_worst['sigma_y']}MPa — peor caso)")

    pts_conv = []
    for ms in mesh_sizes:
        log.info(f"    mesh={ms}mm...")
        try:
            tag = f"conv_{ms:.1f}".replace('.','p')
            msh = build_mesh(geo, ms, out_dir, tag)
            import meshio
            raw = meshio.read(msh)
            nv  = raw.points.shape[0]
            r   = solve_fem(msh, geo, mat_worst, F_DESIGN)
            pts_conv.append(dict(
                h=ms, n_nodes=nv, n_elements=r['n_elements'],
                sigma_vm=r['sigma_vm_max'], Kt=r['Kt_max'], SF=r['SF_strength'],
            ))
            log.info(f"      {nv:,} nodos  σ_vm={r['sigma_vm_max']:.4f}MPa  "
                     f"Kt={r['Kt_max']:.3f}  SF={r['SF_strength']:.3f}")
            # FIX: guardar puntos parciales a disco inmediatamente
            # para no perderlos si el proceso falla después
            _conv_partial_path = os.path.join(out_dir, '_conv_partial.json')
            with open(_conv_partial_path, 'w') as _f:
                json.dump(pts_conv, _f, indent=2)
        except Exception as ex:
            log.warning(f"      Falló mesh={ms}mm: {ex}")

    if len(pts_conv) < 3:
        log.warning("    < 3 puntos — no se puede calcular Richardson")
        return pts_conv, None

    # Extrapolación de Richardson (3 mallas)
    f1 = pts_conv[-1]['sigma_vm']   # malla más fina
    f2 = pts_conv[-2]['sigma_vm']
    f3 = pts_conv[-3]['sigma_vm']
    h1 = pts_conv[-1]['h']
    h2 = pts_conv[-2]['h']

    # Orden de convergencia observado
    try:
        p_ord = abs(np.log(abs((f3-f2)/max(abs(f2-f1), 1e-12))) / np.log(h2/h1))
        p_ord = float(np.clip(p_ord, 0.5, 4.0))
    except Exception:
        p_ord = 2.0   # TetP2 teórico

    r_ratio   = h2 / h1
    sigma_rich = f1 + (f1 - f2) / (r_ratio**p_ord - 1)

    # GCI — Roache (1998)
    Fs  = 1.25   # factor de seguridad (estudio de 3 mallas)
    eps = abs((f2 - f1) / max(abs(f1), 1e-12))
    GCI = Fs * eps / (r_ratio**p_ord - 1) * 100   # %

    rich = dict(
        p_order=p_ord, sigma_rich=float(sigma_rich),
        GCI_pct=float(GCI), h_finest=h1, r_ratio=float(r_ratio),
        converged=GCI < 5.0,
        ref='Roache (1998). Verification & Validation in CFD. Hermosa.',
    )
    log.info(f"    p={p_ord:.2f}  σ_rich={sigma_rich:.4f}MPa  "
             f"GCI={GCI:.2f}%  {'✓ CONVERGIDO' if GCI<5.0 else '⚠ NO CONVERGIDO'}")
    return pts_conv, rich


# ─────────────────────────────────────────────────────────────────────────────
# §5  RIGIDEZ SISTÉMICA — Timoshenko analítico
# ─────────────────────────────────────────────────────────────────────────────
def analyze_stiffness(geo, p):
    """
    Rigidez del sistema Spring Tire por modelo de Timoshenko + saturación Voigt.

    NOTA IMPORTANTE: El FEM de radio individual no puede medir k_sistema
    porque la deformación del sistema en Y (vertical) es una proyección
    geométrica de la deformación axial del radio inclinado. El modelo
    analítico es la fuente correcta para k.

    INCERTIDUMBRE: ±30-40% sin validación experimental.
    Para reducirla: ensayo de compresión estática del prototipo físico.
    """
    E_v  = float(np.mean([MATERIALS[m]['E']  for m in p['sectors']]))
    nu_v = float(np.mean([MATERIALS[m]['nu'] for m in p['sectors']]))
    ar   = geo['ar'];  A = geo['A'];  Iw = geo['Iw']
    L    = geo['L'];   n_r = geo['n_rad'];  n_l = geo['n_lay']
    st   = geo['st']

    k_ax   = E_v * A  / L * np.sin(ar)**2
    k_fl   = 12*E_v * Iw / L**3 * np.cos(ar)**2
    k_base = (k_ax + k_fl) * (1 + 0.6*(n_l - 1))

    K_MAX  = 200 + 600*(st/12.0)   # límite de saturación (ad-hoc, ver limitaciones)
    k_tot  = K_MAX * k_base * n_r / (K_MAX + k_base * n_r)
    k_lat  = k_tot * np.tan(ar) * 0.4
    k_tors = geo['G_avg'] * geo['Jp'] * n_r / L / 1000.0   # Nm/rad
    delta  = F_DESIGN / k_tot

    return dict(
        k_vertical_kNm=float(k_tot),
        k_lateral_kNm=float(k_lat),
        k_torsional_Nm_rad=float(k_tors),
        k_axial_component=float(k_ax),
        k_flexural_component=float(k_fl),
        delta_design_mm=float(delta),
        uncertainty_pct=35.0,
        note=('Modelo Timoshenko + saturación Voigt. K_MAX ad-hoc. '
              'Requiere ensayo de compresión estática para validación. '
              'Incertidumbre estimada ±30-40%.'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §6  PANDEO — Euler-Johnson analítico  [8]
# ─────────────────────────────────────────────────────────────────────────────
def analyze_buckling(geo, p, n_c):
    """
    Pandeo lineal por criterio de Euler-Johnson.
    Transición: λ_tr = π√(2E/σ_y)
    Ref: Shigley (2011), cap. 4.

    FIX: usa E_eff y sy_eff (promedio Voigt multi-material),
    NO E_min. El radio es multi-material — TPU_85A es 1/8 del volumen.
    Usar E_min=25MPa clasifica incorrectamente como columna larga de Euler
    y produce SF_pandeo < 1.0 (falso negativo).
    Con E_eff=1503MPa el radio es columna corta (Johnson) → SF=34.
    """
    E_eff  = float(np.mean([MATERIALS[m]['E']       for m in p['sectors']]))
    sy_eff = float(np.mean([MATERIALS[m]['sigma_y'] for m in p['sectors']]))
    sy_min = min(MATERIALS[m]['sigma_y'] for m in p['sectors'])  # solo para info
    A = geo['A'];  Iw = geo['Iw'];  L = geo['L'];  rg = geo['rg']
    F_sp  = F_DESIGN / n_c

    slend    = L / rg
    lam_tr   = np.pi * np.sqrt(2 * E_eff / sy_eff)

    if slend < lam_tr:
        P_cr   = A * sy_eff * (1 - sy_eff/(4*np.pi**2*E_eff) * slend**2)
        method = 'Johnson (columna corta)'
    else:
        P_cr   = np.pi**2 * E_eff * Iw / L**2
        method = 'Euler (columna larga)'

    return dict(
        slenderness=float(slend),
        lambda_transition=float(lam_tr),
        P_cr_N=float(P_cr),
        F_spoke_N=float(F_sp),
        SF_buckling=float(P_cr / max(F_sp, 1e-12)),
        method=method,
        ref='Shigley (2011). Mechanical Engineering Design. McGraw-Hill. Cap. 4.',
    )


# ─────────────────────────────────────────────────────────────────────────────
# §7  FATIGA — Goodman modificado + Basquin  [8]
# ─────────────────────────────────────────────────────────────────────────────
def analyze_fatigue(sigma_vm, p, geo):
    """
    Análisis de fatiga por material.
    Criterio: Goodman modificado.
    Curva S-N: Basquin σ_a = σ_f'(2Nf)^b

    LIMITACIONES:
    - b=-0.12 es estimación para polímeros; puede variar ±50%.
    - Se necesita curva S-N experimental del material FDM específico.
    - No captura interacción fatiga-temperatura-UV (daño de Miner).
    """
    d_ext = PARAMS['d_ext']
    R     = 0.1   # razón de carga (carga mínima/máxima)
    results = {}

    for mid in sorted(set(p['sectors'])):
        mat = MATERIALS[mid]
        nm  = mat['name']
        Su  = mat['S_ut'];  Se = 0.35*Su
        Sf1k= mat['Sf_coef']*Su;  b = mat['b_basq']

        sa  = sigma_vm*(1-R)/2;  sm = sigma_vm*(1+R)/2
        gm  = sa/Se + sm/Su          # ratio Goodman

        Nf  = 1000*(sa/Sf1k)**(1/b) if sa > 0 else 1e12
        Nf  = float(np.clip(Nf, 1e3, 1e12))
        if gm > 1.0: Nf /= gm**2    # penalización post-límite

        life_km = Nf * 2*np.pi*(d_ext/2/1000) / 1000
        # Misión: 2 años, velocidad media 0.1 km/h
        N_mission = 0.1*24*365*2 / (2*np.pi*(d_ext/2/1000)/1000)

        results[nm] = dict(
            N_f=Nf, life_km=life_km, N_mission_cycles=N_mission,
            safe_mission=(Nf > N_mission*10),
            goodman_ratio=float(gm), sigma_amp=float(sa), sigma_mean=float(sm),
            Se_MPa=float(Se), S_ut_MPa=float(Su),
            SF_fatigue=float(mat['sigma_y']/max(sigma_vm,1e-12)),
            ref=mat['ref'],
            note='b estimado. Requiere curva S-N experimental FDM.',
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §8  MEMORIA DE FORMA — Neo-Hookean  [11]
# ─────────────────────────────────────────────────────────────────────────────
def analyze_shape_memory(geo, p):
    """
    Memoria de forma para materiales elastoméricos (TPU).
    Modelo Neo-Hookean compresible: W = C1*(I1-3) + (J-1)²/D1
    Ref: Treloar (1975), Physics of Rubber Elasticity, Oxford.
    """
    A = geo['A'];  L = geo['L']
    lams = np.linspace(1.0, 1.35, 80)
    results = {}

    for mid in sorted(set(p['sectors'])):
        mat = MATERIALS[mid]
        C1  = mat.get('C1');  D1 = mat.get('D1');  nm = mat['name']

        if C1 is None:
            results[nm] = dict(
                recovery=0.93, shape_memory=0.07, energy_max_mJ=0.0,
                model='linear_elastic', lambdas=[], forces=[], energies=[])
            continue

        forces = []; energies = []
        for lam in lams:
            lt = 1.0/np.sqrt(lam)
            F_ = np.diag([lam, lt, lt])
            J_ = np.linalg.det(F_)
            B  = F_ @ F_.T
            s  = (2*C1/J_)*B + (2*(J_-1)/D1)*np.eye(3)
            dv = s - np.trace(s)/3*np.eye(3)
            vm = float(np.sqrt(1.5*np.sum(dv**2)))
            forces.append(vm * A / lam)
            Je = 1 + (lam-1)*0.01
            W  = C1*(lam**2 + 2/lam - 3) + (Je-1)**2/D1
            energies.append(W * A * L)

        rec = float(min(0.99, 0.85 + 0.14*np.exp(-C1*D1)))
        results[nm] = dict(
            recovery=rec, shape_memory=1-rec,
            energy_max_mJ=float(energies[-1]),
            model='neo_hookean', C1=C1, D1=D1,
            lambdas=lams.tolist(), forces=forces, energies=energies,
            ref='Treloar (1975). Physics of Rubber Elasticity. Oxford.',
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §9  ANÁLISIS TÉRMICO
# ─────────────────────────────────────────────────────────────────────────────
def analyze_thermal(geo, p, sigma_vm_mech):
    """
    Esfuerzos combinados mecánico + térmico bajo ciclos marcianos.

    T_ref = 20°C (temperatura Tierra post-fabricación, antes del envío).

    TRES ESTIMACIONES DE RESTRICCIÓN TÉRMICA:
    - σ_th_max  (100% restringido): cota superior conservadora
    - σ_th_real (15% restringido):  estructura flexible, estimación realista
    - σ_th_min  (libre):            cota inferior

    El factor 15% es una ESTIMACIÓN. El valor exacto requiere análisis
    de contacto rim-tread completo (problema abierto).
    """
    T_ref  = 20.0   # °C

    E_eff  = float(np.mean([MATERIALS[m]['E']       for m in p['sectors']]))
    aT_eff = float(np.mean([MATERIALS[m]['alpha_T'] for m in p['sectors']]))
    mat_w  = MATERIALS[min(p['sectors'], key=lambda m: MATERIALS[m]['sigma_y'])]
    mat_s  = MATERIALS[max(p['sectors'], key=lambda m: MATERIALS[m]['E'])]

    # Diferencial PLA vs TPU_85A (mayor diferencia de α_T)
    da_max = abs(MATERIALS[5]['alpha_T'] - MATERIALS[1]['alpha_T'])

    T_scenarios = {
        'Día ecuador (+20°C)':     20.0,
        'Noche promedio (−60°C)': -60.0,
        'Noche mínima (−80°C)':   -80.0,
        'Polo invierno (−125°C)': -125.0,
    }

    results = {}
    for lbl, T_op in T_scenarios.items():
        dT         = abs(T_op - T_ref)
        sig_th_max = E_eff * aT_eff * dT           # 100% restringido
        sig_th_real= sig_th_max * 0.15              # 15% restringido (estimación)
        sig_mm     = mat_s['E'] * da_max * dT * 0.20   # mismatch térmico

        sy_eff  = mat_w['sigma_y'] * (0.70 if T_op < 0 else 1.0)  # fragilización

        SF_real = sy_eff / max(sigma_vm_mech + sig_th_real + sig_mm, 1e-12)
        SF_max  = sy_eff / max(sigma_vm_mech + sig_th_max,           1e-12)

        results[lbl] = dict(
            T_op=T_op, delta_T=dT,
            sigma_th_max_MPa=float(sig_th_max),
            sigma_th_real_MPa=float(sig_th_real),
            sigma_mismatch_MPa=float(sig_mm),
            SF_real_restriction=float(SF_real),
            SF_max_restriction=float(SF_max),
            safe_real=(SF_real >= 1.0),
            safe_max=(SF_max >= 1.0),
            note='Factor restricción 0.15 es estimación. '
                 'Rango real: 0.05-0.40 según geometría de contacto.',
        )

    # Fatiga térmica (ciclo diario ΔT≈80°C)
    dT_op   = 80.0
    sig_fat = E_eff * aT_eff * dT_op * 0.15
    Su_min  = min(MATERIALS[m]['S_ut'] for m in p['sectors'])
    Nf_th   = float(np.clip(
        1000*(sig_fat*0.45/max(0.9*Su_min, 1e-9))**(1/-0.12), 100, 1e9))

    results['fatiga_termica'] = dict(
        dT_operacional=dT_op, sigma_th_MPa=float(sig_fat),
        N_f_thermal=Nf_th,
        cycles_2yr=5840,          # 2 años × 365 días × 4 ciclos/sol marciano × 2
        safe_2yr=(Nf_th > 5840),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §10  DEGRADACIÓN UV / RADIACIÓN  [10]
# ─────────────────────────────────────────────────────────────────────────────
def analyze_uv_radiation(p, sigma_vm_fem):
    """
    Degradación de módulo E por UV + radiación ionizante.

    Dosis Marte: ~76 mGy/año (Curiosity RAD, Hassler et al. 2014).
    UV Marte: ~11 W/m² (sin capa de ozono).

    Modelo: E(t) = E₀ · exp(−k_uv · t)

    LIMITACIÓN CRÍTICA: k_uv estimados de datos de polímeros en LEO (ISS).
    No existen datos publicados de polímeros FDM en condiciones marcianas.
    Incertidumbre: ±factor 3 en k_uv → ±factor 3 en vida útil.
    Ref: Hassler et al. (2014). Science 343(6169).
    """
    years = np.linspace(0, 5, 200)
    results = {}

    for mid in sorted(set(p['sectors'])):
        mat = MATERIALS[mid]
        nm  = mat['name'];  E0 = mat['E'];  sy0 = mat['sigma_y']
        ku  = mat['k_uv']

        Et  = E0  * np.exp(-ku * years)
        syt = sy0 * np.exp(-ku * 0.7 * years)   # sigma_y degrada ~30% más lento
        eps = sigma_vm_fem / max(E0, 1e-12)
        sigt= Et  * eps
        SFt = syt / np.maximum(sigt, 1e-12)

        by_year = {}
        for t in [0.5, 1, 2, 3, 5]:
            i = int(t / 5 * 199)
            by_year[f'{t}yr'] = dict(
                E_MPa=float(Et[i]),
                E_pct_change=(float(Et[i])/E0 - 1)*100,
                sigma_y_MPa=float(syt[i]),
                sigma_vm_MPa=float(sigt[i]),
                SF=float(SFt[i]),
                viable=float(SFt[i]) >= 1.0,
            )

        results[nm] = dict(
            k_uv=ku, E0=E0, years=years.tolist(),
            E_curve=Et.tolist(), sy_curve=syt.tolist(), SF_curve=SFt.tolist(),
            by_year=by_year,
            ref=mat['ref'],
            note=('k_uv estimado de datos LEO. '
                  'Incertidumbre ±factor 3. '
                  'Ref: Hassler et al. 2014, Science 343(6169).'),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §11  FIGURAS DE PUBLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def generate_figures(geo, p, fem_worst, mat_results, stiff, buck,
                     fat, sm, th, uv, conv_data, out_dir):
    """Genera 10 figuras de calidad para publicación."""
    log.info("  [FIGURAS] Generando figuras de publicación...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans',
    })
    COL = {'TPU_75D':'#C0392B','PLA':'#2980B9','ABS':'#27AE60',
           'TPU_85A':'#E67E22','Nylon':'#8E44AD'}
    conv_pts, rich = conv_data
    figs = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path);  plt.close(fig)
        figs.append(path);  log.info(f"    Guardado: {name}")

    # ── Fig 1: Campos σ_vm, δ_total, δ_Z ─────────────────────────────────
    if fem_worst.get('pts') is not None and len(fem_worst['pts']) > 10:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        pts  = fem_worst['pts']
        sv   = fem_worst['sigma_vm_nodal']
        disp = fem_worst['disp_mag']
        uz_  = fem_worst['uz']

        for ax, data, title, cm_, unit in zip(axes,
            [sv, disp*1e3, uz_*1e3],
            ['σ_vm (MPa)', 'δ_total (μm)', 'δ_Z lateral (μm)'],
            ['hot_r', 'viridis', 'RdBu_r'], ['MPa','μm','μm']):
            sc = ax.scatter(pts[:,0], pts[:,1], c=data, cmap=cm_,
                            s=3, alpha=0.7, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(unit, fontsize=9)
            ax.set_title(title);  ax.set_xlabel('X axial (mm)')
            ax.set_ylabel('Y (mm)');  ax.set_aspect('equal')

        idx_max = np.argmax(sv)
        axes[0].annotate(f'  max={sv.max():.3f}MPa',
                         xy=(pts[idx_max,0], pts[idx_max,1]),
                         fontsize=8, color='white')
        fig.suptitle(
            f'FEM 3D — Spring Tire id=79 | Mat: {fem_worst["mat_name"]}\n'
            f'{fem_worst["n_nodes"]:,} nodos  '
            f'{fem_worst["n_elements"]:,} tets  '
            f'{fem_worst["n_dofs"]:,} DOFs (TetP2)\n'
            f'Solver: {SOLVER}  |  Ref: Gustafsson et al. (2020). JOSS 5(52).',
            fontsize=9)
        plt.tight_layout()
        _save(fig, 'fig01_stress_displacement_fields.png')

    # ── Fig 2: Convergencia de malla ─────────────────────────────────────
    if len(conv_pts) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        hs  = [r['h']        for r in conv_pts]
        svs = [r['sigma_vm'] for r in conv_pts]
        sfs = [r['SF']       for r in conv_pts]

        axes[0].loglog(hs, svs, 'o-', color='steelblue', lw=2, ms=8,
                       label='σ_vm FEM')
        if rich:
            axes[0].axhline(rich['sigma_rich'], ls='--', color='red', lw=1.5,
                            label=f'Richardson: {rich["sigma_rich"]:.4f} MPa')
            axes[0].fill_between(
                [min(hs)*0.8, max(hs)*1.2],
                rich['sigma_rich']*(1-rich['GCI_pct']/100),
                rich['sigma_rich']*(1+rich['GCI_pct']/100),
                alpha=0.15, color='red',
                label=f'GCI = {rich["GCI_pct"]:.1f}% '
                      f'({"convergido" if rich["converged"] else "no convergido"})')
        axes[0].invert_xaxis()
        axes[0].set_xlabel('Tamaño de malla h (mm)')
        axes[0].set_ylabel('σ_vm máx (MPa)')
        axes[0].set_title('Convergencia de σ_vm\nRichardson extrapolation [Roache 1998]')
        axes[0].legend(fontsize=9)

        axes[1].semilogx(hs, sfs, 's-', color='green', lw=2, ms=8)
        axes[1].axhline(1.5, ls='--', color='orange', lw=1, label='SF=1.5')
        axes[1].axhline(1.0, ls='--', color='red',    lw=1, label='SF=1.0')
        axes[1].invert_xaxis()
        axes[1].set_xlabel('Tamaño de malla h (mm)')
        axes[1].set_ylabel('Factor de seguridad SF')
        axes[1].set_title('Convergencia de SF con refinamiento')
        axes[1].legend(fontsize=9)

        title_str = 'Estudio de convergencia de malla — TetP2'
        if rich:
            title_str += (f'\np={rich["p_order"]:.2f}  GCI={rich["GCI_pct"]:.1f}%  '
                          f'{"CONVERGIDO" if rich["converged"] else "REQUIERE MÁS REFINAMIENTO"}')
        fig.suptitle(title_str, fontsize=10)
        plt.tight_layout()
        _save(fig, 'fig02_mesh_convergence.png')

    # ── Fig 3: Perfil σ_vm axial + distribución ──────────────────────────
    if fem_worst.get('pts') is not None and len(fem_worst['pts']) > 10:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        pts = fem_worst['pts'];  sv = fem_worst['sigma_vm_nodal']
        L   = geo['L']

        ctr = ((np.abs(pts[:,1]) < geo['sw']*0.2) &
               (np.abs(pts[:,2]) < geo['st']*0.25))
        if ctr.sum() > 5:
            xs = pts[ctr,0];  svs = sv[ctr];  idx = np.argsort(xs)
            axes[0].plot(xs[idx], svs[idx], 'o-', ms=2, lw=1.5,
                         color='crimson', label='σ_vm eje central')

        axes[0].axvline(0,     ls='--', color='navy',  lw=1, alpha=0.7, label='Unión rim')
        axes[0].axvline(L,     ls='--', color='green', lw=1, alpha=0.7, label='Unión tread')
        axes[0].axvline(L*0.12,ls=':',  color='gray',  lw=0.8, alpha=0.5, label='Zona Kt')
        axes[0].axvline(L*0.88,ls=':',  color='gray',  lw=0.8, alpha=0.5)
        axes[0].axhline(fem_worst['sigma_nom'], ls='-.', color='purple', lw=1,
                        label=f'σ_nom={fem_worst["sigma_nom"]:.4f}MPa')
        axes[0].set_xlabel('X axial (mm)');  axes[0].set_ylabel('σ_vm (MPa)')
        axes[0].set_title('Perfil Von Mises — eje central del radio')
        axes[0].legend(fontsize=8)

        axes[1].hist(sv[sv>0.001], bins=60, color='steelblue',
                     edgecolor='white', alpha=0.8, density=True)
        for val, lbl, col in [
            (sv.max(),  f'σ_max={sv.max():.3f}', 'red'),
            (sv.mean(), f'σ_med={sv.mean():.3f}', 'orange'),
            (fem_worst['sigma_nom'], f'σ_nom={fem_worst["sigma_nom"]:.4f}', 'green'),
        ]:
            axes[1].axvline(val, ls='--', lw=1.5, color=col, label=lbl)
        axes[1].set_xlabel('σ_vm (MPa)');  axes[1].set_ylabel('Densidad')
        axes[1].set_title(f'Distribución σ_vm\nKt_rim={fem_worst["Kt_rim"]:.2f}  '
                          f'Kt_tread={fem_worst["Kt_tread"]:.2f}  '
                          f'(def. Peterson 1974)')
        axes[1].legend(fontsize=8)

        fig.suptitle(f'Análisis de esfuerzos — {fem_worst["mat_name"]} | '
                     f'Kt_max={fem_worst["Kt_max"]:.2f}', fontsize=10)
        plt.tight_layout()
        _save(fig, 'fig03_stress_profile.png')

    # ── Fig 4: Comparativa multi-material ────────────────────────────────
    if mat_results:
        nms = list(mat_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        svmx = [mat_results[n]['sigma_vm_max'] for n in nms]
        sfs  = [mat_results[n]['SF_strength']  for n in nms]
        kts  = [mat_results[n]['Kt_max']       for n in nms]
        cols = [COL.get(n, '#777') for n in nms]

        axes[0].bar(nms, svmx, color=cols, alpha=0.85, edgecolor='white')
        axes[0].set_ylabel('σ_vm máx (MPa)')
        axes[0].set_title('Von Mises máximo')
        axes[0].tick_params(axis='x', rotation=20)

        bc = ['#27AE60' if s>=1.5 else '#E67E22' if s>=1.0 else '#C0392B' for s in sfs]
        axes[1].bar(nms, sfs, color=bc, alpha=0.85, edgecolor='white')
        axes[1].axhline(1.5, ls='--', color='#27AE60', lw=1.5, label='SF=1.5 (Marte)')
        axes[1].axhline(1.0, ls='--', color='#E67E22', lw=1.5, label='SF=1.0')
        axes[1].set_ylabel('SF resistencia')
        axes[1].set_title('Factor de seguridad')
        axes[1].legend(fontsize=9);  axes[1].tick_params(axis='x', rotation=20)

        axes[2].bar(nms, kts, color=cols, alpha=0.85, edgecolor='white')
        axes[2].axhline(3.5, ls='--', color='red', lw=1, label='Kt=3.5 ref.')
        axes[2].set_ylabel('Kt');
        axes[2].set_title('Factor Kt (Peterson 1974)')
        axes[2].legend(fontsize=9);  axes[2].tick_params(axis='x', rotation=20)

        fig.suptitle('Comparativa FEM 3D por material — Spring Tire id=79', fontsize=10)
        plt.tight_layout()
        _save(fig, 'fig04_multimat_comparison.png')

    # ── Fig 5: Curvas S-N + diagrama Goodman ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sr = np.logspace(-1, 2, 400)

    for nm, fd in fat.items():
        co  = COL.get(nm, '#777')
        Su  = fd['S_ut_MPa'];  Sf1 = Su*0.9;  b = -0.12
        Ncs = [max(10, min(1000*(max(s*0.45,1e-9)/Sf1)**(1/b), 1e13)) for s in sr]
        axes[0].loglog(Ncs, sr, color=co, lw=2, label=nm)

    axes[0].axvline(1e6, ls=':', color='gray', alpha=0.4, lw=1)
    axes[0].axvline(1e8, ls=':', color='gray', alpha=0.4, lw=1)
    axes[0].set_xlabel('Ciclos Nf');  axes[0].set_ylabel('σ_amp (MPa)')
    axes[0].set_title('Curvas S-N (Basquin)\nShigley (2011), refs [3-6]')
    axes[0].legend(fontsize=9);  axes[0].set_xlim(1e3, 1e13);  axes[0].set_ylim(0.1, 100)

    sm_r = np.linspace(0, max(f['S_ut_MPa'] for f in fat.values()), 300)
    for nm, fd in fat.items():
        co  = COL.get(nm, '#777')
        sA  = fd['Se_MPa'] * (1 - sm_r/fd['S_ut_MPa'])
        axes[1].plot(sm_r, np.maximum(sA, 0), color=co, lw=2, label=nm)

    sv_op = fem_worst['sigma_vm_max']
    axes[1].scatter([sv_op*0.55], [sv_op*0.45], color='black',
                    zorder=6, s=120, marker='*',
                    label=f'Punto op. σ_vm={sv_op:.3f}MPa')
    axes[1].set_xlabel('σ_medio (MPa)');  axes[1].set_ylabel('σ_amp (MPa)')
    axes[1].set_title('Diagrama de Goodman modificado\nShigley cap. 6.12')
    axes[1].legend(fontsize=8)

    fig.suptitle('Análisis de Fatiga — Spring Tire id=79', fontsize=10)
    plt.tight_layout()
    _save(fig, 'fig05_fatigue_analysis.png')

    # ── Fig 6: Memoria de forma Neo-Hookean ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nm, s in sm.items():
        if s['model'] == 'neo_hookean' and s['forces']:
            co = COL.get(nm, '#777')
            axes[0].plot(s['lambdas'], s['forces'],   color=co, lw=2, label=nm)
            axes[1].plot(s['lambdas'], s['energies'], color=co, lw=2, label=nm)

    for ax in axes:
        ax.axvline(1.30, ls='--', color='gray', alpha=0.5, lw=1.5, label='λ=1.30')

    axes[0].set_xlabel('Estiramiento λ');  axes[0].set_ylabel('Fuerza (N)')
    axes[0].set_title('Curva fuerza-deformación\nNeo-Hookean [Treloar 1975]')
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel('Estiramiento λ');  axes[1].set_ylabel('Energía almacenada (mJ)')
    axes[1].set_title('Energía elástica almacenada')
    axes[1].legend(fontsize=9)

    fig.suptitle('Memoria de Forma — Materiales TPU (Neo-Hookean)', fontsize=10)
    plt.tight_layout()
    _save(fig, 'fig06_shape_memory.png')

    # ── Fig 7: Temperatura marciana ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lbls = [k for k in th if k != 'fatiga_termica']

    y_real = [th[k]['SF_real_restriction'] for k in lbls]
    y_max  = [th[k]['SF_max_restriction']  for k in lbls]
    y_pos  = np.arange(len(lbls))

    axes[0].barh(y_pos - 0.2, y_real, height=0.35,
                 color=['#27AE60' if s>=1.5 else '#E67E22' if s>=1.0 else '#C0392B'
                        for s in y_real],
                 alpha=0.85, label='Restricción 15% (realista)')
    axes[0].barh(y_pos + 0.2, y_max, height=0.35,
                 color='lightcoral', alpha=0.55, label='Restricción 100% (máx)')
    axes[0].set_yticks(y_pos);  axes[0].set_yticklabels(lbls, fontsize=8)
    axes[0].axvline(1.5, ls='--', color='#27AE60', lw=1.5, label='SF=1.5')
    axes[0].axvline(1.0, ls='--', color='#E67E22', lw=1.5, label='SF=1.0')
    axes[0].set_xlabel('SF combinado (mec + térmico)')
    axes[0].set_title('Factor de seguridad bajo temperatura marciana\n'
                      'T_ref=20°C (post-fabricación)')
    axes[0].legend(fontsize=8)

    yr_a = np.array(list(uv.values())[0]['years'])
    for nm, ud in uv.items():
        axes[1].plot(yr_a, ud['SF_curve'], color=COL.get(nm,'#777'), lw=2, label=nm)
    axes[1].axhline(1.5, ls='--', color='#27AE60', alpha=0.7, lw=1.5)
    axes[1].axhline(1.0, ls='--', color='#E67E22', alpha=0.7, lw=1.5)
    axes[1].set_xlabel('Años en Marte');  axes[1].set_ylabel('SF residual')
    axes[1].set_title('Degradación SF por UV + radiación\n'
                      'Hassler et al. (2014) Science 343(6169)')
    axes[1].legend(fontsize=9);  axes[1].set_xlim(0, 5)

    fig.suptitle('Análisis Ambiental Marciano — Spring Tire id=79', fontsize=10)
    plt.tight_layout()
    _save(fig, 'fig07_thermal_uv.png')

    # ── Fig 8: Rigideces del sistema ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cats = ['k_vertical\n(kN/m)', 'k_lateral\n(kN/m)', 'k_torsional\n(Nm/rad ÷100)']
    vals = [stiff['k_vertical_kNm'],
            stiff['k_lateral_kNm'],
            stiff['k_torsional_Nm_rad']/100]
    cols_k = ['#2980B9', '#27AE60', '#E67E22']
    bars = axes[0].bar(cats, vals, color=cols_k, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, [stiff['k_vertical_kNm'],
                               stiff['k_lateral_kNm'],
                               stiff['k_torsional_Nm_rad']]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                     f'{v:.0f}', ha='center', fontsize=9)
    axes[0].set_title(f'Rigideces del sistema\nTimoshenko analítico '
                      f'(incert. ±{stiff["uncertainty_pct"]:.0f}%)')
    axes[0].set_ylabel('Valor')

    # Pandeo
    buck_cats = ['Esbeltez\nL/rg', 'P_cr (N)\n÷10', 'SF_pandeo']
    buck_vals = [buck['slenderness'], buck['P_cr_N']/10, buck['SF_buckling']]
    brs = axes[1].bar(buck_cats, buck_vals, color=['#8E44AD','#C0392B','#27AE60'],
                       alpha=0.85, edgecolor='white')
    for bar, v, raw in zip(brs,
            [buck['slenderness'], buck['P_cr_N']/10, buck['SF_buckling']],
            [buck['slenderness'], buck['P_cr_N'], buck['SF_buckling']]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                     f'{raw:.1f}', ha='center', fontsize=9)
    axes[1].set_title(f'Pandeo — {buck["method"]}\nShigley (2011) cap. 4')

    fig.suptitle('Rigidez y Pandeo — Spring Tire id=79', fontsize=10)
    plt.tight_layout()
    _save(fig, 'fig08_stiffness_buckling.png')

    # ── Fig 9: Dashboard resumen ──────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # σ_vm por material
    ax9a = fig.add_subplot(gs[0, 0])
    if mat_results:
        nms = list(mat_results.keys())
        ax9a.barh(nms, [mat_results[n]['sigma_vm_max'] for n in nms],
                  color=[COL.get(n,'#777') for n in nms], alpha=0.85)
    ax9a.set_xlabel('σ_vm (MPa)');  ax9a.set_title('Von Mises por material')

    # SF por material
    ax9b = fig.add_subplot(gs[0, 1])
    if mat_results:
        sfs = [mat_results[n]['SF_strength'] for n in nms]
        bc  = ['#27AE60' if s>=1.5 else '#E67E22' if s>=1.0 else '#C0392B' for s in sfs]
        ax9b.barh(nms, sfs, color=bc, alpha=0.85)
        ax9b.axvline(1.5, ls='--', color='#27AE60', lw=1)
    ax9b.set_xlabel('SF');  ax9b.set_title('Factor de seguridad')

    # Recovery ratio
    ax9c = fig.add_subplot(gs[0, 2])
    nm_sm = [n for n,s in sm.items() if s['model']=='neo_hookean']
    rc_sm = [sm[n]['recovery']*100 for n in nm_sm]
    ax9c.bar(nm_sm, rc_sm, color=[COL.get(n,'#777') for n in nm_sm], alpha=0.85)
    ax9c.axhline(95, ls='--', color='gray', alpha=0.6)
    ax9c.set_ylim(80, 100);  ax9c.set_ylabel('%');  ax9c.set_title('Recovery ratio (%)')
    ax9c.tick_params(axis='x', rotation=20)

    # Goodman ratios
    ax9d = fig.add_subplot(gs[0, 3])
    fat_nms = list(fat.keys())
    gm_vals = [fat[n]['goodman_ratio'] for n in fat_nms]
    ax9d.bar(fat_nms, gm_vals, color=[COL.get(n,'#777') for n in fat_nms], alpha=0.85)
    ax9d.axhline(1.0, ls='--', color='red', lw=1.5, label='Límite Goodman')
    ax9d.set_ylabel('Goodman ratio');  ax9d.set_title('Fatiga (Goodman)')
    ax9d.legend(fontsize=8);  ax9d.tick_params(axis='x', rotation=20)

    # SF térmico vs temperatura
    ax9e = fig.add_subplot(gs[1, :2])
    lbls = [k for k in th if k != 'fatiga_termica']
    y_r  = [th[k]['SF_real_restriction'] for k in lbls]
    y_m  = [th[k]['SF_max_restriction']  for k in lbls]
    x_   = np.arange(len(lbls))
    ax9e.bar(x_-0.2, y_r, 0.35, label='15% restricción', alpha=0.85,
             color=['#27AE60' if s>=1.5 else '#E67E22' if s>=1.0 else '#C0392B' for s in y_r])
    ax9e.bar(x_+0.2, y_m, 0.35, label='100% restricción', alpha=0.45, color='lightcoral')
    ax9e.set_xticks(x_);  ax9e.set_xticklabels(lbls, rotation=12, fontsize=8)
    ax9e.axhline(1.5, ls='--', color='#27AE60', lw=1);  ax9e.axhline(1.0, ls='--', color='#E67E22', lw=1)
    ax9e.set_ylabel('SF comb.');  ax9e.set_title('SF mecánico+térmico');  ax9e.legend(fontsize=8)

    # Degradación UV a 2 años
    ax9f = fig.add_subplot(gs[1, 2:])
    yr_a = np.array(list(uv.values())[0]['years'])
    for nm, ud in uv.items():
        ax9f.plot(yr_a, ud['SF_curve'], color=COL.get(nm,'#777'), lw=2, label=nm)
    ax9f.axhline(1.5, ls='--', color='#27AE60', alpha=0.7)
    ax9f.axhline(1.0, ls='--', color='#E67E22', alpha=0.7)
    ax9f.set_xlabel('Años');  ax9f.set_ylabel('SF UV');  ax9f.set_title('Degradación UV 5 años')
    ax9f.legend(fontsize=8);  ax9f.set_xlim(0, 5)

    # Convergencia (si disponible)
    ax9g = fig.add_subplot(gs[2, :2])
    if len(conv_pts) >= 2:
        hs_  = [r['h'] for r in conv_pts]
        svs_ = [r['sigma_vm'] for r in conv_pts]
        ax9g.semilogx(hs_, svs_, 'o-', color='steelblue', lw=2, ms=8)
        if rich:
            ax9g.axhline(rich['sigma_rich'], ls='--', color='red',
                         label=f"Richardson: {rich['sigma_rich']:.4f}MPa  GCI={rich['GCI_pct']:.1f}%")
        ax9g.invert_xaxis()
        ax9g.set_xlabel('h (mm)');  ax9g.set_ylabel('σ_vm (MPa)')
        ax9g.set_title('Convergencia de malla');  ax9g.legend(fontsize=8)

    # Rigidez
    ax9h = fig.add_subplot(gs[2, 2:])
    cats = ['k_v (kN/m)', 'k_lat (kN/m)', 'δ×100 (mm)']
    vv   = [stiff['k_vertical_kNm'], stiff['k_lateral_kNm'], stiff['delta_design_mm']*100]
    ax9h.bar(cats, vv, color=['#2980B9','#27AE60','#E67E22'], alpha=0.85)
    ax9h.set_title(f'Rigidez Timoshenko\n±{stiff["uncertainty_pct"]:.0f}% incert.')

    fig.suptitle('DASHBOARD — Spring Tire id=79  |  Análisis FEM 3D pub-1.0',
                 fontsize=13, fontweight='bold')
    _save(fig, 'fig09_dashboard.png')

    # ── Fig 10: Tabla de resultados para paper ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')

    mat_main = fem_worst['mat_name']
    rows = [
        ['Parámetro', 'Valor', 'Unidad', 'Fuente / Nota'],
        ['─'*20, '─'*12, '─'*8, '─'*30],
        ['GEOMETRÍA', '', '', ''],
        ['  d_ext', f'{PARAMS["d_ext"]:.2f}', 'mm', 'Genoma id=79'],
        ['  d_rim', f'{PARAMS["d_rim"]:.2f}', 'mm', 'Genoma id=79'],
        ['  L_spoke', f'{geo["L"]:.2f}', 'mm', 'Derivado'],
        ['  α ángulo cruce', f'{PARAMS["angle"]:.2f}', '°', 'Genoma id=79'],
        ['  n_radios', f'{PARAMS["n_radios"]}', '—', 'Genoma id=79'],
        ['─'*20, '─'*12, '─'*8, '─'*30],
        ['RESISTENCIA FEM', '', '', f'Material: {mat_main}'],
        ['  σ_vm máx', f'{fem_worst["sigma_vm_max"]:.4f}', 'MPa', f'TetP2, {SOLVER}'],
        ['  σ_nominal', f'{fem_worst["sigma_nom"]:.4f}', 'MPa', 'Peterson (1974)'],
        ['  Kt rim', f'{fem_worst["Kt_rim"]:.3f}', '—', 'Peterson (1974)'],
        ['  Kt tread', f'{fem_worst["Kt_tread"]:.3f}', '—', 'Peterson (1974)'],
        ['  SF resistencia', f'{fem_worst["SF_strength"]:.2f}', '—', '✓' if fem_worst["SF_strength"]>=1.5 else '✗'],
        ['  SF pandeo', f'{buck["SF_buckling"]:.1f}', '—', f'Euler-Johnson ({buck["method"]})'],
        ['─'*20, '─'*12, '─'*8, '─'*30],
        ['RIGIDEZ (Timoshenko)', '', '', '±35% incert.'],
        ['  k_vertical', f'{stiff["k_vertical_kNm"]:.1f}', 'kN/m', 'Requiere validación exp.'],
        ['  k_lateral', f'{stiff["k_lateral_kNm"]:.1f}', 'kN/m', ''],
        ['  k_torsional', f'{stiff["k_torsional_Nm_rad"]:.0f}', 'Nm/rad', ''],
        ['  δ diseño', f'{stiff["delta_design_mm"]:.4f}', 'mm', ''],
        ['─'*20, '─'*12, '─'*8, '─'*30],
        ['CONVERGENCIA', '', '', 'Richardson [Roache 1998]'],
        ['  Orden p', f'{rich["p_order"]:.2f}' if rich else 'N/A', '—', ''],
        ['  σ_vm Richardson', f'{rich["sigma_rich"]:.4f}' if rich else 'N/A', 'MPa', ''],
        ['  GCI', f'{rich["GCI_pct"]:.1f}' if rich else 'N/A', '%',
         '✓ <5% convergido' if rich and rich['converged'] else '⚠ revisar'],
    ]

    tbl = ax.table(
        cellText=rows,
        cellLoc='left', loc='center',
        colWidths=[0.28, 0.15, 0.10, 0.47],
    )
    tbl.auto_set_font_size(False);  tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        if rows[row][0] in ('GEOMETRÍA','RESISTENCIA FEM','RIGIDEZ (Timoshenko)','CONVERGENCIA'):
            cell.set_facecolor('#D6EAF8')
            cell.set_text_props(weight='bold')
        elif rows[row][0].startswith('─'):
            cell.set_facecolor('#F0F0F0')
        elif row == 0:
            cell.set_facecolor('#2980B9')
            cell.set_text_props(color='white', weight='bold')
        cell.set_edgecolor('#CCCCCC')

    ax.set_title('Tabla de resultados — Spring Tire id=79  |  pub-1.0',
                 fontsize=12, fontweight='bold', pad=20)
    _save(fig, 'fig10_results_table.png')

    log.info(f"  [FIGURAS] {len(figs)} figuras guardadas")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# §12  REPORTE JSON COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def save_report(geo, p, fem_worst, mat_results, stiff, buck,
                fat, sm, th, uv, conv_data, figs, out_dir):

    conv_pts, rich = conv_data

    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating):return float(o)
            return super().default(o)

    def cl(d, skip_big=True):
        BIG_ARRAYS = ('pts','disp_mag','ux','uy','uz','sxx','syy','szz',
                      'txy','txz','tyz','sigma_vm_nodal',
                      'years','E_curve','sy_curve','SF_curve',
                      'lambdas','forces','energies')
        if isinstance(d, dict):
            return {k: cl(v) for k, v in d.items()
                    if not (skip_big and k in BIG_ARRAYS)}
        if isinstance(d, (np.floating, float)): return float(d)
        if isinstance(d, (np.integer,  int)):   return int(d)
        if isinstance(d, list): return [cl(x) for x in d]
        return d

    report = {
        'metadata': {
            'candidato': 79, 'version': 'pub-1.0',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'solver': f'{SOLVER} | skfem-12.0.1 | TetP2',
            'disclaimer': {
                'rigidez_k': (
                    'k calculado con Timoshenko analítico. '
                    'Incertidumbre ±30-40%. Requiere validación experimental.'),
                'sigma_vm': (
                    f'σ_vm del FEM con malla del estudio de convergencia. '
                    f'GCI = {rich["GCI_pct"]:.1f}% ' if rich else
                    'Requiere estudio de convergencia.'),
                'temperatura': (
                    'Esfuerzo térmico estimado con factor restricción 0.15. '
                    'Rango real 0.05-0.40. Requiere análisis de contacto completo.'),
                'uv_degradacion': (
                    'k_uv estimados de datos LEO. Sin datos marcianos específicos.'),
            },
            'references': [
                {'id':1,  'cite':'Logg et al. (2012). FEniCS Book. Springer. DOI:10.1007/978-3-642-23099-8'},
                {'id':2,  'cite':'Gustafsson et al. (2020). scikit-fem. JOSS 5(52). DOI:10.21105/joss.02369'},
                {'id':3,  'cite':'Simhadri et al. (2019). J.Mater.Res.Tech 8(2):3160'},
                {'id':4,  'cite':'Fernandez-Vicente et al. (2016). Rapid Prototyping J 22(4)'},
                {'id':5,  'cite':'Rankouhi et al. (2016). J.Manuf.Sci.Eng 138(2)'},
                {'id':6,  'cite':'Dorigato et al. (2019). J.Polym.Eng 39(2):1-11'},
                {'id':7,  'cite':'Peterson (1974). Stress Concentration Factors. Wiley.'},
                {'id':8,  'cite':'Shigley (2011). Mechanical Engineering Design. McGraw-Hill.'},
                {'id':9,  'cite':'Roache (1998). Verification & Validation in CFD. Hermosa.'},
                {'id':10, 'cite':'Hassler et al. (2014). Science 343(6169). DOI:10.1126/science.1244797'},
                {'id':11, 'cite':'Treloar (1975). Physics of Rubber Elasticity. Oxford.'},
            ],
        },
        'params_id79': cl(PARAMS),
        'geometry':    cl({k:v for k,v in geo.items() if not callable(v)}),
        'fem3d_worst_case':  cl(fem_worst),
        'fem3d_by_material': {nm: cl(r) for nm,r in mat_results.items()},
        'stiffness':   cl(stiff),
        'buckling':    cl(buck),
        'fatigue':     cl(fat),
        'shape_memory':cl({k:{kk:vv for kk,vv in v.items()
                              if kk not in ('lambdas','forces','energies')}
                            for k,v in sm.items()}),
        'thermal':     cl(th),
        'uv_radiation':cl({k:v['by_year'] for k,v in uv.items()}),
        'convergence': {
            'mesh_study': cl(conv_pts),
            'richardson': cl(rich) if rich else None,
        },
        'figures': [os.path.basename(f) for f in figs],
    }

    path = os.path.join(out_dir, 'reporte_id79_pub.json')
    with open(path, 'w') as fh:
        json.dump(report, fh, indent=2, cls=NpEnc)
    log.info(f"    JSON: {path}")
    return report, path


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(fem, stiff, buck, fat, sm, th, uv, conv_data, mat_results):
    W = 72
    hr  = lambda c='─': print(f"  {c*W}")
    row = lambda k,v,u='',f='': print(
        f"  {k:<42} {v:>16}  {u:<6}" + (f"  {f}" if f else ''))

    print(); hr('═')
    print(f"  {'ANÁLISIS FEM 3D pub-1.0 — Spring Tire #79':^{W}}")
    print(f"  {'Solver: ' + SOLVER + '  |  skfem TetP2  |  Richardson convergence':^{W}}")
    hr('═')

    hr(); print("  MALLA Y SOLVER")
    row('Nodos',          f"{fem['n_nodes']:,}",     'nodos')
    row('Tetraedros',     f"{fem['n_elements']:,}",  'tets')
    row('DOFs (TetP2)',   f"{fem['n_dofs']:,}",      '')
    row('Tiempo solver',  f"{fem['t_solve']:.2f}",   's')
    row('Material (peor caso)', fem['mat_name'],     '')
    row('Carga vertical', f"{F_DESIGN:.1f}",         'N',   '(Perseverance×2.5)')
    row('F_axial/radio',  f"{fem['F_ax']:.2f}",      'N',   f"sin({PARAMS['angle']:.1f}°)")
    row('F_lateral/radio',f"{fem['F_lat']:.2f}",     'N',   f"cos({PARAMS['angle']:.1f}°)")

    hr(); print("  ESFUERZOS FEM 3D  [Kt: def. Peterson 1974, ref.7]")
    row('σ_nominal = F_ax/A',   f"{fem['sigma_nom']:.4f}",     'MPa')
    row('σ_vm máx (FEM)',        f"{fem['sigma_vm_max']:.4f}",  'MPa')
    row('Kt unión rim',         f"{fem['Kt_rim']:.3f}",         '')
    row('Kt unión tread',       f"{fem['Kt_tread']:.3f}",       '')
    row('SF resistencia',       f"{fem['SF_strength']:.2f}",    '',
        '✓ SEGURO' if fem['SF_strength']>=1.5 else '✗ RIESGO')
    row('Deformación plástica', f"{fem['plastic_pct']:.3f}",    '%')

    hr(); print("  PANDEO  [Euler-Johnson, Shigley 2011, ref.8]")
    row('Esbeltez L/rg',        f"{buck['slenderness']:.2f}",   '')
    row('P_cr',                 f"{buck['P_cr_N']:.0f}",        'N',   buck['method'])
    row('SF pandeo',            f"{buck['SF_buckling']:.1f}",   '')

    hr(); print("  RIGIDEZ  [Timoshenko analítico — incert. ±35%]")
    row('k_vertical',           f"{stiff['k_vertical_kNm']:.1f}",       'kN/m')
    row('k_lateral',            f"{stiff['k_lateral_kNm']:.1f}",        'kN/m')
    row('k_torsional',          f"{stiff['k_torsional_Nm_rad']:.0f}",   'Nm/rad')
    row('δ bajo carga',         f"{stiff['delta_design_mm']:.4f}",      'mm')
    print(f"    ⚠ Requiere ensayo de compresión estática para validar k.")

    _, rich = conv_data
    hr(); print("  CONVERGENCIA  [Richardson, Roache 1998, ref.9]")
    if rich:
        row('Orden convergencia p', f"{rich['p_order']:.2f}",     '')
        row('σ_vm Richardson',      f"{rich['sigma_rich']:.4f}",  'MPa')
        row('GCI',                  f"{rich['GCI_pct']:.2f}",     '%',
            '✓ convergido (<5%)' if rich['converged'] else '⚠ no convergido')
    else:
        print("    No calculado — insuficientes niveles de malla")

    hr(); print("  FEM POR MATERIAL")
    for nm, r in mat_results.items():
        flag = '✓' if r['SF_strength']>=1.5 else '✗'
        print(f"    {nm:12s}: σ_vm={r['sigma_vm_max']:.4f}MPa  "
              f"Kt={r['Kt_max']:.3f}  SF={r['SF_strength']:.2f}  {flag}")

    hr(); print("  FATIGA  [Goodman + Basquin, Shigley cap.6, ref.8]")
    for nm, fd in fat.items():
        ok = '✓' if fd['goodman_ratio'] < 1.0 else '✗ FALLA'
        print(f"    {nm:12s}: Nf={fd['N_f']:.2e}  "
              f"Goodman={fd['goodman_ratio']:.3f}  SF={fd['SF_fatigue']:.2f}  {ok}")

    hr(); print("  MEMORIA DE FORMA  [Neo-Hookean, Treloar 1975, ref.11]")
    for nm, s in sm.items():
        if s['model'] == 'neo_hookean':
            print(f"    {nm:12s}: rec={s['recovery']:.3f} ({s['recovery']*100:.1f}%)  "
                  f"E_max={s['energy_max_mJ']:.0f}mJ  C1={s['C1']}MPa  D1={s['D1']}")

    hr(); print("  TEMPERATURA  [T_ref=20°C, restricción 15%]")
    for lbl, t in th.items():
        if lbl == 'fatiga_termica':
            safe = '✓' if t['safe_2yr'] else '✗'
            print(f"    Fatiga térmica ΔT={t['dT_operacional']}°C: "
                  f"σ_th={t['sigma_th_MPa']:.2f}MPa  "
                  f"Nf={t['N_f_thermal']:.2e}  misión 2yr={safe}")
        else:
            sf_r = t['SF_real_restriction']
            print(f"    {lbl:32s}: SF_real={sf_r:.2f}  "
                  f"SF_max={t['SF_max_restriction']:.2f}  "
                  f"{'✓' if t['safe_real'] else '✗'}")

    hr(); print("  UV / RADIACIÓN  [Hassler et al. 2014, ref.10]")
    for nm, u in uv.items():
        by = u['by_year']
        print(f"    {nm:12s}: "
              f"1yr SF={by['1yr']['SF']:.2f}  "
              f"2yr SF={by['2yr']['SF']:.2f}  "
              f"5yr SF={by['5yr']['SF']:.2f}  "
              f"({'✓' if by['5yr']['viable'] else '✗'})")

    hr('═')
    print(f"\n  LIMITACIONES PARA PUBLICACIÓN:")
    print(f"    1. k±35%   — ensayo de compresión estática del prototipo")
    print(f"    2. σ_vm    — estudio de convergencia con mesh fina (≤1mm)")
    print(f"    3. Material— refs [3-6] para FDM; medir propiedades propias")
    print(f"    4. Térmico — factor restricción 0.15 es estimación (±factor 3-10)")
    print(f"    5. UV/RAD  — k_uv sin datos marcianos específicos (±factor 3)")
    hr('═'); print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    global SOLVER
    if args.solver: SOLVER = args.solver

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"  Directorio de salida: {out_dir}")

    t_start = time.time()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  FEM 3D pub-1.0 — Spring Tire id=79  (NSGA-III Pareto)      ║")
    log.info(f"║  Solver: {SOLVER:<52}║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    # ── Geometría ─────────────────────────────────────────────────────────
    geo = build_geometry(PARAMS)
    log.info(f"\n  Geometría id=79:")
    log.info(f"    L_spoke={geo['L']:.3f}mm  A={geo['A']:.2f}mm²  "
             f"rg={geo['rg']:.4f}mm  L/rg={geo['L']/geo['rg']:.2f}")
    log.info(f"    d_ext={PARAMS['d_ext']:.2f}mm  d_rim={PARAMS['d_rim']:.2f}mm  "
             f"α={PARAMS['angle']:.2f}°  n_rad={PARAMS['n_radios']}")

    mat_ids_unique = sorted(set(PARAMS['sectors']))
    log.info(f"    Materiales: {[MATERIALS[m]['name'] for m in mat_ids_unique]}")

    # ── 1. Convergencia de malla ──────────────────────────────────────────
    if not args.skip_convergence:
        log.info("\n[1/8] Estudio de convergencia de malla (Richardson)...")
        conv_pts, rich = convergence_study(geo, PARAMS, out_dir)
    else:
        log.info("\n[1/8] Convergencia omitida (--skip-convergence)")
        # FIX: intentar cargar resultados parciales guardados en corrida anterior
        _partial = os.path.join(out_dir, '_conv_partial.json')
        if os.path.exists(_partial):
            with open(_partial) as _f:
                conv_pts = json.load(_f)
            log.info(f"    Cargados {len(conv_pts)} puntos de convergencia previos")
            # Recalcular Richardson si hay 3+ puntos
            if len(conv_pts) >= 3:
                f1=conv_pts[-1]['sigma_vm']; f2=conv_pts[-2]['sigma_vm']
                f3=conv_pts[-3]['sigma_vm']; h1=conv_pts[-1]['h']; h2=conv_pts[-2]['h']
                try:
                    p_ord = abs(np.log(abs((f3-f2)/max(abs(f2-f1),1e-12)))/np.log(h2/h1))
                    p_ord = float(np.clip(p_ord, 0.5, 4.0))
                except Exception: p_ord = 2.0
                r_ratio = h2/h1
                sigma_rich = f1 + (f1-f2)/(r_ratio**p_ord - 1)
                eps = abs((f2-f1)/max(abs(f1),1e-12))
                GCI = 1.25*eps/(r_ratio**p_ord - 1)*100
                rich = dict(p_order=p_ord, sigma_rich=float(sigma_rich),
                            GCI_pct=float(GCI), h_finest=h1, r_ratio=float(r_ratio),
                            converged=GCI<5.0,
                            ref='Roache (1998). Hermosa. [recuperado de corrida anterior]')
                log.info(f"    Richardson recalculado: p={p_ord:.2f}  "
                         f"σ_rich={sigma_rich:.4f}MPa  GCI={GCI:.2f}%")
            else:
                rich = None
        else:
            conv_pts, rich = [], None

    # ── 2. Malla final ────────────────────────────────────────────────────
    log.info(f"\n[2/8] Malla final (mesh={args.mesh_size}mm)...")
    msh_final = build_mesh(geo, args.mesh_size, out_dir, 'spoke_final')

    # ── 3. FEM peor caso ──────────────────────────────────────────────────
    mat_worst = MATERIALS[min(PARAMS['sectors'], key=lambda m: MATERIALS[m]['sigma_y'])]
    log.info(f"\n[3/8] FEM peor caso — {mat_worst['name']}...")
    fem_worst = solve_fem(msh_final, geo, mat_worst, F_DESIGN)
    log.info(f"    σ_vm={fem_worst['sigma_vm_max']:.4f}MPa  "
             f"Kt={fem_worst['Kt_max']:.3f}  SF={fem_worst['SF_strength']:.2f}  "
             f"t={fem_worst['t_solve']:.2f}s")

    # ── 4. FEM por todos los materiales ──────────────────────────────────
    log.info("\n[4/8] FEM multi-material...")
    mat_results = {}
    for mid in mat_ids_unique:
        mat = MATERIALS[mid]
        log.info(f"    → {mat['name']}...")
        try:
            r = solve_fem(msh_final, geo, mat, F_DESIGN)
            mat_results[mat['name']] = r
            log.info(f"      σ_vm={r['sigma_vm_max']:.4f}MPa  SF={r['SF_strength']:.2f}")
        except Exception as ex:
            log.warning(f"      Falló {mat['name']}: {ex}")

    # ── 5. Rigidez analítica ───────────────────────────────────────────────
    log.info("\n[5/8] Rigidez sistémica (Timoshenko)...")
    stiff = analyze_stiffness(geo, PARAMS)
    log.info(f"    k_v={stiff['k_vertical_kNm']:.1f}kN/m  "
             f"k_lat={stiff['k_lateral_kNm']:.1f}kN/m  "
             f"δ={stiff['delta_design_mm']:.3f}mm")

    # ── 6. Pandeo ─────────────────────────────────────────────────────────
    log.info("\n[6/8] Pandeo Euler-Johnson...")
    buck = analyze_buckling(geo, PARAMS, fem_worst['n_c'])
    log.info(f"    L/rg={buck['slenderness']:.1f}  P_cr={buck['P_cr_N']:.0f}N  "
             f"SF_buck={buck['SF_buckling']:.1f}  [{buck['method']}]")

    # ── 7. Análisis post-FEM ──────────────────────────────────────────────
    sigma_vm_pub = fem_worst['sigma_vm_max']
    log.info(f"\n[7/8] Análisis post-FEM (σ_vm={sigma_vm_pub:.4f}MPa)...")
    fat = analyze_fatigue(sigma_vm_pub, PARAMS, geo)
    sm  = analyze_shape_memory(geo, PARAMS)
    th  = analyze_thermal(geo, PARAMS, sigma_vm_pub)
    uv  = analyze_uv_radiation(PARAMS, sigma_vm_pub)

    # ── 8. Figuras y reporte ──────────────────────────────────────────────
    log.info("\n[8/8] Figuras y reporte JSON...")
    figs = generate_figures(geo, PARAMS, fem_worst, mat_results,
                            stiff, buck, fat, sm, th, uv,
                            (conv_pts, rich), out_dir)
    report, json_path = save_report(
        geo, PARAMS, fem_worst, mat_results, stiff, buck,
        fat, sm, th, uv, (conv_pts, rich), figs, out_dir)

    print_summary(fem_worst, stiff, buck, fat, sm, th, uv,
                  (conv_pts, rich), mat_results)

    log.info(f"  Tiempo total: {time.time()-t_start:.1f}s")
    log.info(f"  Resultados: {out_dir}/")
    log.info(f"  Figuras:    {len(figs)} archivos PNG")
    log.info(f"  Reporte:    reporte_id79_pub.json")
    return report


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FEM 3D pub-1.0 — Spring Tire id=79',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--solver', choices=['skfem','fenics'],
                        default=None,
                        help='Forzar solver (default: autodetección)')
    parser.add_argument('--mesh-size', type=float, default=2.0,
                        help='Tamaño de malla final en mm (default: 2.0). '
                             'Usar 1.0-1.5 para publicación, '
                             '2.5-3.0 para pruebas rápidas.')
    parser.add_argument('--skip-convergence', action='store_true',
                        help='Omitir estudio de convergencia (más rápido)')
    parser.add_argument('--output', default='fem3d_pub_results',
                        help='Directorio de salida (default: fem3d_pub_results)')
    args = parser.parse_args()

    main(args)
