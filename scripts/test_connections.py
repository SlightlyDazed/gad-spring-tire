#!/usr/bin/env python3
"""
test_connections.py - Test de validación REAL de conexiones v5.2

Valida la malla STL generada, no puntos teóricos.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("TEST: Validación REAL de Conexiones v5.2")
print("="*70)

# Test 1: Genoma y parámetros
print("\n[1] Genoma y Parámetros")
print("-"*50)

from utils.spring_tire_genome import create_baseline_genome, decode_genome

genome = create_baseline_genome('tpu_homogeneous')
params = decode_genome(genome)

print(f"  d_ext: {params['d_ext']:.1f} mm")
print(f"  d_rim: {params['d_rim']:.1f} mm")
print(f"  wall_thickness: {params['wall_thickness']:.1f} mm")
print(f"  n_radios: {params['n_radios']}")

# Calcular espacio para radios
tread_thick = params['wall_thickness'] * 2.0
rim_thick = params['wall_thickness'] * 1.5
r_tread_inner = params['d_ext'] / 2 - tread_thick
r_rim_outer = params['d_rim'] / 2 + rim_thick
gap = r_tread_inner - r_rim_outer

print(f"\n  Espacio para radios: {gap:.1f} mm")
print(f"  {'✓' if gap >= 10 else '✗'} Espacio suficiente: {gap:.1f}mm >= 10mm")

# Test 2: Generar geometría con validación
print("\n[2] Generación y Validación de Malla")
print("-"*50)

try:
    from utils.spring_tire_geometry_v4 import SpringTireGeometry
    from utils.spring_tire_validator import validate_spring_tire_stl, print_validation_report
    
    tire = SpringTireGeometry(params)
    mesh, validation = tire.generate_and_validate()
    
    if mesh:
        print(f"  ✓ Malla generada: {len(mesh.vertices):,} vértices, {len(mesh.faces):,} caras")
        
        # Imprimir reporte de validación
        print_validation_report(validation)
        
        # Guardar STL
        os.makedirs('results/test_v3', exist_ok=True)
        mesh.export('results/test_v3/spring_tire_validated.stl')
        print(f"\n  STL guardado: results/test_v3/spring_tire_validated.stl")
        
    else:
        print("  ✗ Error generando malla")
        
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Caso problemático (espacio insuficiente)
print("\n[3] Test Caso Problemático")
print("-"*50)

bad_params = params.copy()
bad_params['d_ext'] = 300
bad_params['d_rim'] = 280  # Muy grande, casi igual a d_ext

print(f"  d_ext={bad_params['d_ext']}, d_rim={bad_params['d_rim']}")

try:
    bad_tire = SpringTireGeometry(bad_params)
    print("  ✗ No detectó error geométrico")
except ValueError as e:
    print(f"  ✓ Detectó error correctamente: {e}")

# Test 4: Diferentes estrategias de material
print("\n[4] Test Estrategias de Material")
print("-"*50)

strategies = ['tpu_homogeneous', 'flexible_rigid', 'graded']

for strategy in strategies:
    genome = create_baseline_genome(strategy)
    params_strat = decode_genome(genome)
    
    tire = SpringTireGeometry(params_strat)
    mesh, validation = tire.generate_and_validate()
    
    if mesh and validation['valid']:
        print(f"\n  {strategy}:")
        print(f"    ✓ Válida: {validation['spokes']['connects_both']} radios conectados")
        print(f"    Componentes: tread={validation['components']['tread_outer']['n_vertices']}, "
              f"rim={validation['components']['rim_outer']['n_vertices']}")
    else:
        print(f"\n  {strategy}: ✗ Inválida")
        if validation.get('errors'):
            for e in validation['errors']:
                print(f"    Error: {e}")

print("\n" + "="*70)
print("TEST COMPLETADO")
print("="*70)
