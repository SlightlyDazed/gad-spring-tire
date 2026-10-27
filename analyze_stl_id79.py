#!/usr/bin/env python3
"""
analyze_stl_id79.py - Análisis visual de STL Spring Tire — Candidato #79
Genera figuras de publicación para el paper.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import trimesh
except ImportError:
    print("Error: trimesh no disponible"); sys.exit(1)

from utils.spring_tire_genome import decode_genome
from utils.spring_tire_geometry_v5 import SpringTireGeometryV5

# ── Genoma exacto del candidato #79 ─────────────────────────────────────────
GENOME_ID79 = np.array([
    0.05287105764312245, 0.05908018855032715, 0.0025819797224635992,
    0.9897561235521021,  0.9323394123476569,  0.4319387342851343,
    0.40726387012368803, 0.9586777319329612,  0.7667399027368678,
    0.30905624189901215, 0.10706105448259384, 0.9261010891048566,
    0.8020836402160857,  0.36205914845479814, 0.04598863984672745,
    0.9976230546307375,  0.7564864314082056,  0.1960405926728645,
])

OUTPUT_DIR = "results/analysis_id79"


def render_mesh_views(mesh, title, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vertices = mesh.vertices
    faces = mesh.faces

    views = [
        ("Front", 0, 0), ("Side", 90, 0), ("Top", 0, 90),
        ("Isometric", 45, 45), ("Isometric 2", -45, 45), ("Bottom", 0, -90),
    ]

    fig = plt.figure(figsize=(20, 12))
    for idx, (name, azim, elev) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, alpha=0.7, color='steelblue', edgecolor='none')
        ax.view_init(elev=elev, azim=azim)
        ax.plot([0, 200], [0, 0], [0, 0], 'r-', lw=2, label='X')
        ax.plot([0, 0], [0, 200], [0, 0], 'g-', lw=2, label='Y')
        ax.plot([0, 0], [0, 0], [0, 100], 'b-', lw=2, label='Z')
        ax.set_title(f"{name} (azim={azim}, elev={elev})")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        max_range = np.ptp(vertices, axis=0).max() / 2
        mid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    path = os.path.join(output_dir, "id79_views.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ Guardado: {path}")
    return path


def analyze_cross_section(mesh, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vertices = mesh.vertices

    tolerance = 5.0
    section_mask = np.abs(vertices[:, 2]) < tolerance  # corte en Z≈0 (eje del cilindro)
    section_verts = vertices[section_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    r_ext = params['d_ext'] / 2
    r_rim = params['d_rim'] / 2
    wall = params['wall_thickness']
    tread_thick = wall * 2
    rim_thick = wall * 1.5

    ax1 = axes[0]
    if len(section_verts) > 0:
        ax1.scatter(section_verts[:, 0], section_verts[:, 1], c='blue', s=10, alpha=0.5)
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(r_ext*np.cos(theta), r_ext*np.sin(theta), 'r--', alpha=0.5, label=f'tread outer ({r_ext:.1f})')
    ax1.plot((r_ext-tread_thick)*np.cos(theta), (r_ext-tread_thick)*np.sin(theta), 'r:', alpha=0.5, label=f'tread inner ({r_ext-tread_thick:.1f})')
    ax1.plot((r_rim+rim_thick)*np.cos(theta), (r_rim+rim_thick)*np.sin(theta), 'g--', alpha=0.5, label=f'rim outer ({r_rim+rim_thick:.1f})')
    ax1.plot(r_rim*np.cos(theta), r_rim*np.sin(theta), 'g:', alpha=0.5, label=f'rim inner ({r_rim:.1f})')
    ax1.set_aspect('equal'); ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)')
    ax1.set_title(f'Cross Section (Z≈0, ±{tolerance}mm) — ID79')
    ax1.legend(loc='upper right', fontsize=7); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    ax2.hist(r, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(r_ext, color='r', ls='--', label=f'tread outer')
    ax2.axvline(r_ext-tread_thick, color='r', ls=':', label=f'tread inner')
    ax2.axvline(r_rim+rim_thick, color='g', ls='--', label=f'rim outer')
    ax2.axvline(r_rim, color='g', ls=':', label=f'rim inner')
    ax2.set_xlabel('Radius (mm)'); ax2.set_ylabel('Vertex count')
    ax2.set_title('Vertex Distribution by Radius — ID79')
    ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "id79_cross_section.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ Guardado: {path}")
    return path


def analyze_connections(mesh, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vertices = mesh.vertices
    edges = mesh.edges_unique

    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    r_ext = params['d_ext'] / 2
    r_rim = params['d_rim'] / 2
    wall = params['wall_thickness']
    tread_thick = wall * 2; rim_thick = wall * 1.5
    r_tread_inner = r_ext - tread_thick
    r_rim_outer = r_rim + rim_thick
    tol = 5.0

    tread_outer_mask = (r >= r_ext-tol) & (r <= r_ext+tol)
    tread_inner_mask = (r >= r_tread_inner-tol) & (r <= r_tread_inner+tol)
    rim_outer_mask   = (r >= r_rim_outer-tol) & (r <= r_rim_outer+tol)
    rim_inner_mask   = (r >= r_rim-tol) & (r <= r_rim+tol)
    spokes_mask      = (r > r_rim_outer+tol) & (r < r_tread_inner-tol)

    conn = {'rim→spokes': 0, 'spokes→tread': 0, 'rim→tread(direct)': 0, 'total_edges': len(edges)}
    for e in edges:
        v1, v2 = e
        if (rim_outer_mask[v1] and spokes_mask[v2]) or (rim_outer_mask[v2] and spokes_mask[v1]):
            conn['rim→spokes'] += 1
        if (spokes_mask[v1] and tread_inner_mask[v2]) or (spokes_mask[v2] and tread_inner_mask[v1]):
            conn['spokes→tread'] += 1
        if (rim_outer_mask[v1] and tread_inner_mask[v2]) or (rim_outer_mask[v2] and tread_inner_mask[v1]):
            conn['rim→tread(direct)'] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    cats = list(conn.keys())[:3]
    vals = [conn[c] for c in cats]
    colors = ['green' if v > 0 else 'red' for v in vals]
    ax1.bar(cats, vals, color=colors, alpha=0.7, edgecolor='black')
    for i, v in enumerate(vals):
        ax1.text(i, v + max(max(vals),1)*0.02, str(v), ha='center', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Connecting edges'); ax1.set_title('Component Connections — ID79')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    sample = min(5000, len(vertices))
    idx = np.random.choice(len(vertices), sample, replace=False)
    cmap = {'tread_outer':'red','tread_inner':'orange','rim_outer':'green',
            'rim_inner':'lime','spokes':'blue','other':'gray'}
    for i in idx:
        x, y = vertices[i, 0], vertices[i, 1]
        if tread_outer_mask[i]: c = cmap['tread_outer']
        elif tread_inner_mask[i]: c = cmap['tread_inner']
        elif rim_outer_mask[i]: c = cmap['rim_outer']
        elif rim_inner_mask[i]: c = cmap['rim_inner']
        elif spokes_mask[i]: c = cmap['spokes']
        else: c = cmap['other']
        ax2.scatter(x, y, c=c, s=1, alpha=0.5)
    for name, color in cmap.items():
        ax2.scatter([], [], c=color, s=20, label=name)
    ax2.set_aspect('equal'); ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View — Components — ID79')
    ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "id79_connections.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ Guardado: {path}")
    return conn, path


def main():
    print("="*70)
    print("  ANÁLISIS VISUAL — SPRING TIRE CANDIDATO #79")
    print("="*70)

    # Decodificar genoma ID79
    params = decode_genome(GENOME_ID79)

    print(f"\n  Parámetros ID79:")
    print(f"    d_ext          : {params['d_ext']:.2f} mm")
    print(f"    d_rim          : {params['d_rim']:.2f} mm")
    print(f"    width          : {params['width']:.2f} mm")
    print(f"    wall_thickness : {params['wall_thickness']:.2f} mm")
    print(f"    n_radios       : {params['n_radios']}")
    print(f"    angle_cross    : {params['angle_cross']:.2f}°")
    print(f"    spoke_thickness: {params['spoke_thickness']:.2f} mm")
    print(f"    n_layers       : {params['n_layers']}")
    print(f"    materials      : {params['material_names']}")

    # Generar malla
    print("\n  Generando malla 3D...")
    tire = SpringTireGeometryV5(params)
    mesh = tire.generate()

    if mesh is None:
        print("  ERROR: No se pudo generar la malla"); return

    print(f"    Vértices : {len(mesh.vertices):,}")
    print(f"    Caras    : {len(mesh.faces):,}")
    print(f"    Watertight: {mesh.is_watertight}")

    # Exportar STL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stl_path = os.path.join(OUTPUT_DIR, "spring_tire_id79.stl")
    mesh.export(stl_path)
    print(f"    STL: {stl_path}")

    # Generar figuras
    print("\n  [1/3] Vistas múltiples...")
    render_mesh_views(mesh, "Spring Tire — Candidate #79 (NSGA-III Optimized)", OUTPUT_DIR)

    print("  [2/3] Corte transversal...")
    analyze_cross_section(mesh, params, OUTPUT_DIR)

    print("  [3/3] Análisis de conexiones...")
    conn, _ = analyze_connections(mesh, params, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("  CONEXIONES")
    print(f"{'='*70}")
    for k, v in conn.items():
        print(f"    {k}: {v}")
    is_ok = conn['rim→spokes'] > 0 and conn['spokes→tread'] > 0
    print(f"\n    {'✓' if is_ok else '✗'} Conexión completa: {'SÍ' if is_ok else 'NO'}")
    print(f"\n  Figuras guardadas en: {OUTPUT_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
