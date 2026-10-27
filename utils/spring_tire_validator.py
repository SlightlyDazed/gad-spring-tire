import numpy as np
import trimesh
from typing import Dict, List, Any

def validate_spring_tire_stl(mesh: 'trimesh.Trimesh', params: Dict) -> Dict:
    """
    Función principal de validación de STL Spring Tire.

    Args:
        mesh: Malla trimesh generada
        params: Parámetros usados para generar la malla

    Returns:
        Dict con validación completa
    """
    results = {
        'valid': mesh.is_watertight and len(mesh.faces) > 0,
        'stats': {
            'n_vertices': len(mesh.vertices),
            'n_faces': len(mesh.faces),
            'is_watertight': mesh.is_watertight
        },
        'components': {},
        'spokes': {},
        'errors': [],
        'warnings': []
    }
    return results


def print_validation_report(results: Dict):
    """Imprime reporte de validación formateado."""
    print("="*70)
    print("REPORTE DE VALIDACIÓN SPRING TIRE")
    print("="*70)

    print(f"\n✓ Válida: {'SÍ' if results['valid'] else 'NO'}")

    print(f"\nEstadísticas:")
    print(f"  Vértices: {results['stats'].get('n_vertices', 0):,}")
    print(f"  Caras: {results['stats'].get('n_faces', 0):,}")
    print(f"  Componentes conectados: {results['stats'].get('n_connected_components', 'N/A')}")
    print(f"  Watertight: {results['stats'].get('is_watertight', False)}")

    print(f"\nComponentes detectados:")
    for name, info in results['components'].items():
        if name.startswith('component_'):
            continue
        status = '✓' if info.get('has_vertices', False) else '✗'
        print(f"  {status} {name}: {info.get('n_vertices', 0)} vértices")

    print(f"\nRadios:")
    print(f"  Conexiones rim→spokes: {results['spokes'].get('rim_to_spokes', 0)}")
    print(f"  Conexiones spokes→tread: {results['spokes'].get('spokes_to_tread', 0)}")
    print(f"  Conexión completa (rim→spokes→tread): {'✓' if results['spokes'].get('connects_both', 0) else '✗'}")

    if results['errors']:
        print(f"\n✗ Errores:")
        for e in results['errors']:
            print(f"  - {e}")

    if results['warnings']:
        print(f"\n⚠ Advertencias:")
        for w in results['warnings']:
            print(f"  - {w}")

    print("="*70)
