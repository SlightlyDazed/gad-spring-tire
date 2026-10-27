#!/usr/bin/env python3
"""
test_spring_tire.py - Test del sistema Spring Tire v5.0

Prueba las funcionalidades básicas del sistema.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("TEST: NASA Spring Tire System v5.0")
print("="*70)

# Test 1: Genoma
print("\n[1] Test Genoma Spring Tire")
print("-"*50)

from utils.spring_tire_genome import (
    N_GENES, decode_genome, create_baseline_genome, validate_genome
)

# Crear genoma baseline
genome = create_baseline_genome('tpu_homogeneous')
print(f"  Genoma creado: {len(genome)} genes")
print(f"  Valores: {genome[:6].round(3)} ... (geométricos)")

# Decodificar
params = decode_genome(genome)
print(f"\n  Parámetros decodificados:")
print(f"    d_ext: {params['d_ext']:.1f} mm")
print(f"    d_rim: {params['d_rim']:.1f} mm")
print(f"    width: {params['width']:.1f} mm")
print(f"    n_radios: {params['n_radios']}")
print(f"    wall_thickness: {params['wall_thickness']:.1f} mm")
print(f"    angle_cross: {params['angle_cross']:.1f}°")
print(f"    materiales: {params['material_names']}")

# Validar
is_valid, msg = validate_genome(genome)
print(f"\n  Validación: {'✓' if is_valid else '✗'} {msg}")

# Test 2: Geometría
print("\n[2] Test Geometría Spring Tire")
print("-"*50)

try:
    import trimesh
    from utils.spring_tire_geometry_v4 import SpringTireGeometry
    
    tire = SpringTireGeometry(params)
    mesh = tire.generate()
    
    if mesh:
        print(f"  ✓ Geometría generada")
        print(f"    Vértices: {len(mesh.vertices):,}")
        print(f"    Caras: {len(mesh.faces):,}")
        print(f"    Watertight: {mesh.is_watertight}")
        
        # Guardar STL de prueba
        os.makedirs('results/test', exist_ok=True)
        mesh.export('results/test/test_spring_tire.stl')
        print(f"    STL guardado: results/test/test_spring_tire.stl")
    else:
        print(f"  ✗ Error generando geometría")
except ImportError as e:
    print(f"  ⚠ trimesh no disponible: {e}")

# Test 3: FEA
print("\n[3] Test FEA Spring Tire")
print("-"*50)

from utils.spring_tire_fea import run_spring_tire_fea

fea_results = run_spring_tire_fea(params)
print(f"  ✓ FEA completado")
print(f"\n  Propiedades mecánicas:")
print(f"    stiffness_kN_m: {fea_results['stiffness_kN_m']:.2f}")
print(f"    lateral_stiffness_kN_m: {fea_results['lateral_stiffness_kN_m']:.2f}")
print(f"    torsional_stiffness_Nm_rad: {fea_results['torsional_stiffness_Nm_rad']:.2f}")
print(f"    max_load_N: {fea_results['max_load_N']:.2f}")
print(f"    mass_kg: {fea_results['mass_kg']:.4f}")
print(f"    max_deflection_mm: {fea_results['max_deflection_mm']:.2f}")
print(f"    safety_factor: {fea_results['safety_factor']:.2f}")

# Test 4: Diferentes estrategias
print("\n[4] Test Estrategias de Material")
print("-"*50)

strategies = ['tpu_homogeneous', 'flexible_rigid', 'graded', 'nylon_core']

for strategy in strategies:
    genome = create_baseline_genome(strategy)
    params = decode_genome(genome)
    fea = run_spring_tire_fea(params)
    
    print(f"\n  {strategy}:")
    print(f"    materiales: {params['material_names']}")
    print(f"    stiffness: {fea['stiffness_kN_m']:.1f} kN/m")
    print(f"    mass: {fea['mass_kg']:.3f} kg")

# Test 5: Grafo para GNN
print("\n[5] Test Grafo para GNN")
print("-"*50)

try:
    from fase2_spring_tire_gnn import SpringTireGraphBuilder
    
    builder = SpringTireGraphBuilder(params)
    graph = builder.build_graph()
    
    print(f"  ✓ Grafo construido")
    print(f"    Nodos: {len(graph['nodes'])}")
    print(f"    Aristas: {graph['edges'].shape[1]}")
    print(f"    Features nodos: {graph['node_features'].shape}")
    print(f"    Features aristas: {graph['edge_features'].shape}")
except ImportError as e:
    print(f"  ⚠ PyTorch Geometric no disponible: {e}")

print("\n" + "="*70)
print("TEST COMPLETADO")
print("="*70)
print("\nPara ejecutar el pipeline completo:")
print("  python run_spring_tire_pipeline.py --phase all")
