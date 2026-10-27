#!/usr/bin/env python3
"""
analyze_stl_visual.py - Análisis visual de STL Spring Tire

Genera imágenes desde diferentes vistas para analizar conexiones.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Error: trimesh no disponible")
    sys.exit(1)

from utils.spring_tire_genome import create_baseline_genome, decode_genome
from utils.spring_tire_geometry_v4 import SpringTireGeometry


def render_mesh_views(mesh, title="Spring Tire", output_dir="results/analysis"):
    """Genera imágenes del mesh desde diferentes vistas."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Centro de la malla
    center = vertices.mean(axis=0)
    
    # Vistas a generar
    views = [
        ("front", 0, 0),      # Vista frontal
        ("side", 90, 0),      # Vista lateral
        ("top", 0, 90),       # Vista superior
        ("iso", 45, 45),      # Vista isométrica
        ("iso2", -45, 45),    # Otra isométrica
        ("bottom", 0, -90),   # Vista inferior
    ]
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, (name, azim, elev) in enumerate(views):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Plot de la malla
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, alpha=0.7, color='steelblue', edgecolor='none')
        
        # Configurar vista
        ax.view_init(elev=elev, azim=azim)
        
        # Líneas de referencia
        ax.plot([0, 200], [0, 0], [0, 0], 'r-', linewidth=2, label='X')
        ax.plot([0, 0], [0, 200], [0, 0], 'g-', linewidth=2, label='Y')
        ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2, label='Z')
        
        ax.set_title(f"{name} (azim={azim}, elev={elev})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Ejes iguales
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "spring_tire_views.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Imagen guardada: {output_path}")
    plt.close()
    
    return output_path


def analyze_cross_section(mesh, params, output_dir="results/analysis"):
    """Genera vista de corte transversal para ver conexiones."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    vertices = mesh.vertices
    
    # Seleccionar vértices cercanos al plano X=0 (corte transversal)
    tolerance = 5.0
    section_mask = np.abs(vertices[:, 0]) < tolerance
    section_vertices = vertices[section_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vista 2D del corte
    ax1 = axes[0]
    ax1.scatter(section_vertices[:, 1], section_vertices[:, 2], 
                c='blue', s=10, alpha=0.5)
    
    # Líneas de referencia
    r_ext = params['d_ext'] / 2
    r_rim = params['d_rim'] / 2
    wall = params['wall_thickness']
    tread_thick = wall * 2
    rim_thick = wall * 1.5
    
    # Círculos de referencia
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Tread outer
    ax1.plot(r_ext * np.cos(theta), r_ext * np.sin(theta), 'r--', alpha=0.5, label='tread_outer')
    # Tread inner
    ax1.plot((r_ext - tread_thick) * np.cos(theta), (r_ext - tread_thick) * np.sin(theta), 
             'r:', alpha=0.5, label='tread_inner')
    # Rim outer
    ax1.plot((r_rim + rim_thick) * np.cos(theta), (r_rim + rim_thick) * np.sin(theta), 
             'g--', alpha=0.5, label='rim_outer')
    # Rim inner
    ax1.plot(r_rim * np.cos(theta), r_rim * np.sin(theta), 'g:', alpha=0.5, label='rim_inner')
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title(f'Corte Transversal (X≈0, ±{tolerance}mm)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Histograma de radios
    ax2 = axes[1]
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    
    bins = np.linspace(0, r.max() + 10, 30)
    ax2.hist(r, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Líneas de referencia
    ax2.axvline(r_ext, color='r', linestyle='--', label='tread_outer')
    ax2.axvline(r_ext - tread_thick, color='r', linestyle=':', label='tread_inner')
    ax2.axvline(r_rim + rim_thick, color='g', linestyle='--', label='rim_outer')
    ax2.axvline(r_rim, color='g', linestyle=':', label='rim_inner')
    
    ax2.set_xlabel('Radio (mm)')
    ax2.set_ylabel('Número de vértices')
    ax2.set_title('Distribución de Vértices por Radio')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "spring_tire_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Análisis guardado: {output_path}")
    plt.close()
    
    return output_path


def analyze_connections(mesh, params, output_dir="results/analysis"):
    """Analiza conexiones entre componentes."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    vertices = mesh.vertices
    edges = mesh.edges_unique
    
    # Calcular radios
    r = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    
    # Rangos de componentes
    r_ext = params['d_ext'] / 2
    r_rim = params['d_rim'] / 2
    wall = params['wall_thickness']
    tread_thick = wall * 2
    rim_thick = wall * 1.5
    
    r_tread_inner = r_ext - tread_thick
    r_rim_outer = r_rim + rim_thick
    
    tolerance = 5.0
    
    # Máscaras
    tread_outer_mask = (r >= r_ext - tolerance) & (r <= r_ext + tolerance)
    tread_inner_mask = (r >= r_tread_inner - tolerance) & (r <= r_tread_inner + tolerance)
    rim_outer_mask = (r >= r_rim_outer - tolerance) & (r <= r_rim_outer + tolerance)
    rim_inner_mask = (r >= r_rim - tolerance) & (r <= r_rim + tolerance)
    spokes_mask = (r > r_rim_outer + tolerance) & (r < r_tread_inner - tolerance)
    
    # Contar conexiones
    connections = {
        'rim_outer_to_spokes': 0,
        'spokes_to_tread_inner': 0,
        'rim_to_tread_direct': 0,
        'total_edges': len(edges)
    }
    
    for e in edges:
        v1_idx, v2_idx = e[0], e[1]
        
        v1_in_rim = rim_outer_mask[v1_idx]
        v2_in_rim = rim_outer_mask[v2_idx]
        v1_in_spokes = spokes_mask[v1_idx]
        v2_in_spokes = spokes_mask[v2_idx]
        v1_in_tread = tread_inner_mask[v1_idx]
        v2_in_tread = tread_inner_mask[v2_idx]
        
        if (v1_in_rim and v2_in_spokes) or (v2_in_rim and v1_in_spokes):
            connections['rim_outer_to_spokes'] += 1
        
        if (v1_in_spokes and v2_in_tread) or (v2_in_spokes and v1_in_tread):
            connections['spokes_to_tread_inner'] += 1
        
        if (v1_in_rim and v2_in_tread) or (v2_in_rim and v1_in_tread):
            connections['rim_to_tread_direct'] += 1
    
    # Visualizar conexiones
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras de conexiones
    ax1 = axes[0]
    categories = ['rim→spokes', 'spokes→tread', 'rim→tread\n(directo)']
    values = [connections['rim_outer_to_spokes'], 
              connections['spokes_to_tread_inner'],
              connections['rim_to_tread_direct']]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Número de aristas conectoras')
    ax1.set_title('Conexiones entre Componentes')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores sobre las barras
    for i, v in enumerate(values):
        ax1.text(i, v + max(values)*0.02, str(v), ha='center', fontsize=12, fontweight='bold')
    
    # Mapa de vértices por componente (vista superior)
    ax2 = axes[1]
    
    # Muestra de vértices para visualización
    sample_size = min(5000, len(vertices))
    indices = np.random.choice(len(vertices), sample_size, replace=False)
    
    colors_map = {
        'tread_outer': 'red',
        'tread_inner': 'orange',
        'rim_outer': 'green',
        'rim_inner': 'lime',
        'spokes': 'blue',
        'other': 'gray'
    }
    
    for idx in indices:
        x, y = vertices[idx, 0], vertices[idx, 1]
        
        if tread_outer_mask[idx]:
            color = colors_map['tread_outer']
        elif tread_inner_mask[idx]:
            color = colors_map['tread_inner']
        elif rim_outer_mask[idx]:
            color = colors_map['rim_outer']
        elif rim_inner_mask[idx]:
            color = colors_map['rim_inner']
        elif spokes_mask[idx]:
            color = colors_map['spokes']
        else:
            color = colors_map['other']
        
        ax2.scatter(x, y, c=color, s=1, alpha=0.5)
    
    # Leyenda
    for name, color in colors_map.items():
        ax2.scatter([], [], c=color, s=20, label=name)
    
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Vista Superior - Componentes')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "spring_tire_connections.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Conexiones guardadas: {output_path}")
    plt.close()
    
    return connections, output_path


def main():
    print("="*70)
    print("ANÁLISIS VISUAL DE SPRING TIRE")
    print("="*70)
    
    # Crear genoma
    genome = create_baseline_genome('tpu_homogeneous')
    params = decode_genome(genome)
    
    print(f"\nParámetros:")
    print(f"  d_ext: {params['d_ext']:.1f} mm")
    print(f"  d_rim: {params['d_rim']:.1f} mm")
    print(f"  wall_thickness: {params['wall_thickness']:.1f} mm")
    print(f"  n_radios: {params['n_radios']}")
    
    # Generar malla
    print("\nGenerando malla...")
    tire = SpringTireGeometry(params)
    mesh = tire.generate()
    
    if mesh is None:
        print("Error: No se pudo generar la malla")
        return
    
    print(f"  Vértices: {len(mesh.vertices):,}")
    print(f"  Caras: {len(mesh.faces):,}")
    
    # Generar imágenes
    print("\nGenerando imágenes...")
    
    print("\n[1] Vistas múltiples...")
    render_mesh_views(mesh, "Spring Tire v5.2", "results/analysis")
    
    print("\n[2] Corte transversal...")
    analyze_cross_section(mesh, params, "results/analysis")
    
    print("\n[3] Análisis de conexiones...")
    connections, _ = analyze_connections(mesh, params, "results/analysis")
    
    print("\n" + "="*70)
    print("RESULTADOS DE CONEXIONES")
    print("="*70)
    print(f"  rim→spokes: {connections['rim_outer_to_spokes']}")
    print(f"  spokes→tread: {connections['spokes_to_tread_inner']}")
    print(f"  rim→tread (directo): {connections['rim_to_tread_direct']}")
    print(f"  Total aristas: {connections['total_edges']}")
    
    is_connected = (connections['rim_outer_to_spokes'] > 0 and 
                    connections['spokes_to_tread_inner'] > 0)
    print(f"\n  {'✓' if is_connected else '✗'} Conexión completa: {'SÍ' if is_connected else 'NO'}")
    
    print("\n" + "="*70)
    print("Imágenes guardadas en: results/analysis/")
    print("="*70)


if __name__ == '__main__':
    main()
