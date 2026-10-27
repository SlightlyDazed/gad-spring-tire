"""
spring_tire_fea_integrated.py - FEA Integrado para Spring Tire v6.0

Combina FEA simplificado (rápido) + FEM 3D (preciso) según el modo:
- Modo 'fast': FEA simplificado para optimización (~10ms)
- Modo 'accurate': FEM 3D real para validación (~30-120s)
- Modo 'mixed': FEA simplificado con corrección por FEM 3D

Uso recomendado en el pipeline:
1. NSGA-III usa modo 'fast' (miles de evaluaciones)
2. Top-K del frente Pareto se valida con modo 'accurate'
3. Resultados finales exportados con FEM 3D completo
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Optional, Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importar FEA simplificado (siempre disponible)
# ---------------------------------------------------------------------------
from utils.spring_tire_fea import run_spring_tire_fea as fea_fast

# ---------------------------------------------------------------------------
# Importar FEM 3D (opcional, puede no estar disponible)
# ---------------------------------------------------------------------------
try:
    from utils.spring_tire_fem3d import run_fem3d_spoke, FEM3D_AVAILABLE
except ImportError:
    FEM3D_AVAILABLE = False
    log.warning("FEM 3D no disponible - solo modo 'fast' funcionará")


# ---------------------------------------------------------------------------
# Configuración de escenarios de carga (Rovers NASA)
# ---------------------------------------------------------------------------

ROVER_SCENARIOS = {
    'sojourner': {
        'name': 'Mars Pathfinder Sojourner',
        'mass_kg': 11.5,
        'n_wheels': 6,
        'gravity_mars': 3.71,  # m/s² (0.38g)
        'safety_factor': 3.0,
        'description': 'Rover ligero, primera misión exitosa en Marte (1997)'
    },
    'curiosity': {
        'name': 'Mars Science Laboratory Curiosity',
        'mass_kg': 899.0,
        'n_wheels': 6,
        'gravity_mars': 3.71,
        'safety_factor': 2.5,
        'description': 'Rover de tamaño medio, laboratorio móvil (2012)'
    },
    'perseverance': {
        'name': 'Mars 2020 Perseverance',
        'mass_kg': 1025.0,
        'n_wheels': 6,
        'gravity_mars': 3.71,
        'safety_factor': 2.5,
        'description': 'Rover más pesado, misión de retorno de muestras (2021)'
    },
    'custom': {
        'name': 'Configuración personalizada',
        'mass_kg': 100.0,
        'n_wheels': 4,
        'gravity_mars': 3.71,
        'safety_factor': 2.0,
        'description': 'Definido por el usuario'
    }
}


def get_rover_load_scenario(scenario_name: str = 'perseverance') -> Dict:
    """Obtiene el escenario de carga para un rover específico."""
    scenario = ROVER_SCENARIOS.get(scenario_name, ROVER_SCENARIOS['perseverance'])
    
    mass = scenario['mass_kg']
    n_wheels = scenario['n_wheels']
    g_mars = scenario['gravity_mars']
    sf = scenario['safety_factor']
    
    weight_mars_N = mass * g_mars
    load_per_wheel_N = weight_mars_N / n_wheels
    design_load_N = load_per_wheel_N * sf
    
    return {
        'scenario_name': scenario['name'],
        'mass_kg': mass,
        'n_wheels': n_wheels,
        'weight_mars_N': weight_mars_N,
        'load_per_wheel_N': load_per_wheel_N,
        'safety_factor': sf,
        'design_load_N': design_load_N,
        'description': scenario['description']
    }


def run_fea_integrated(params: Dict,
                       mode: Literal['fast', 'accurate', 'mixed'] = 'fast',
                       scenario: str = 'perseverance',
                       use_fem3d_for_validation: bool = False) -> Dict:
    """
    Ejecuta análisis FEA integrado (simplificado + FEM 3D).
    
    Args:
        params: Parámetros de la Spring Tire
        mode: 'fast' (FEA simplificado), 'accurate' (FEM 3D), 'mixed' (ambos)
        scenario: Escenario de rover ('sojourner', 'curiosity', 'perseverance')
        use_fem3d_for_validation: Si True, corre FEM 3D y compara con FEA rápido
    
    Returns:
        Dict con resultados FEA
    """
    load_scenario = get_rover_load_scenario(scenario)
    F_design = load_scenario['design_load_N']
    
    params_with_load = params.copy()
    params_with_load['F_vertical'] = F_design
    
    results = {
        'scenario': load_scenario,
        'mode': mode,
        'params': params_with_load
    }
    
    # MODO FAST: FEA simplificado (~10ms)
    if mode in ['fast', 'mixed']:
        log.debug(f"FEA FAST | {scenario} | F={F_design:.0f}N")
        
        fea_fast_results = fea_fast(params_with_load)
        results['fea_fast'] = fea_fast_results
        
        results['stiffness_kN_m']      = fea_fast_results['stiffness_kN_m']
        results['stiffness_source']    = 'fea_fast'
        results['mass_kg']             = fea_fast_results['mass_kg']
        results['mass_source']         = 'fea_fast'
        results['max_deflection_mm']   = fea_fast_results['max_deflection_mm']
        results['deflection_source']   = 'fea_fast'
        results['safety_factor']       = fea_fast_results['safety_factor']
        results['safety_factor_source'] = 'fea_fast'
    
    # MODO ACCURATE: FEM 3D (~30-120s)
    if mode in ['accurate', 'mixed']:
        if not FEM3D_AVAILABLE:
            log.error("FEM 3D no disponible - instalar gmsh, meshio, skfem")
            if mode == 'accurate':
                return {}
        else:
            log.info(f"FEM 3D | {scenario} | F={F_design:.0f}N | "
                    f"d_ext={params['d_ext']:.0f}mm | "
                    f"wall={params['wall_thickness']:.1f}mm")
            
            import time
            t0 = time.time()
            
            fem3d_results = run_fem3d_spoke(
                params=params_with_load,
                F_vertical=F_design,
                mesh_size=None,
                compute_buckling=False
            )
            
            elapsed = time.time() - t0
            
            if fem3d_results:
                results['fem3d'] = fem3d_results
                results['fem3d_time_seconds'] = elapsed
                
                # BUG#10: en modo 'mixed', los campos principales también se
                # sobreescriben con el FEM 3D (más preciso). Antes se descartaba.
                results['stiffness_kN_m'] = fem3d_results['k_vertical_kNm']
                results['stiffness_source'] = 'fem3d'

                results['max_deflection_mm'] = fem3d_results['max_displacement_mm']
                results['deflection_source'] = 'fem3d'

                results['safety_factor'] = fem3d_results['SF_strength']
                results['safety_factor_source'] = 'fem3d'

                results['sigma_vm_MPa'] = fem3d_results['sigma_vm_MPa']
                results['Kt_max'] = fem3d_results['Kt_max']

                if 'mass_kg' in fem3d_results:
                    results['mass_kg'] = fem3d_results['mass_kg']
                    results['mass_source'] = 'fem3d'
                else:
                    # mass_kg queda del fea_fast (si modo mixed) o ausente (si modo accurate)
                    # Se marca explícitamente para que el consumidor sepa que es estimación 2D
                    results['mass_source'] = 'fea_fast' if mode == 'mixed' else 'unavailable'
                    log.debug(
                        f"mass_kg no disponible en FEM3D — "
                        f"mass_source='{results.get('mass_source', '')}' "
                        f"(mass_kg={results.get('mass_kg', 'N/A')})"
                    )
                
                log.info(f"  FEM 3D completado en {elapsed:.1f}s | "
                        f"k={fem3d_results['k_vertical_kNm']:.1f}kN/m | "
                        f"σ_vm={fem3d_results['sigma_vm_MPa']:.1f}MPa")
            else:
                log.error("FEM 3D falló")
                if mode == 'accurate':
                    return {}
    
    return results


def run_spring_tire_fea_robust(params: Dict, 
                                use_fem3d: bool = False,
                                scenario: str = 'perseverance') -> Dict:
    """Función compatible con la interfaz anterior."""
    mode = 'accurate' if use_fem3d else 'fast'
    results = run_fea_integrated(params, mode=mode, scenario=scenario)
    
    if mode == 'fast':
        fea = results.get('fea_fast', {})
    else:
        fea = results.get('fem3d', {})
        if fea:
            fea['stiffness_kN_m'] = fea.get('k_vertical_kNm', 0)
            fea['max_deflection_mm'] = fea.get('max_displacement_mm', 0)
    
    scenario_info = results.get('scenario', {})
    fea['design_load_N'] = scenario_info.get('design_load_N', 500)
    fea['scenario'] = scenario_info.get('scenario_name', 'unknown')
    
    return fea


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    
    print("="*70)
    print("TEST: FEA Integrado para Spring Tire v6.0")
    print("="*70)
    
    test_params = {
        'd_ext': 400.0,
        'd_rim': 175.0,
        'width': 150.0,
        'n_radios': 17,
        'wall_thickness': 6.0,
        'angle_cross': 55.0,
        'material_sectors': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
    }
    
    # Test modo FAST
    print("\n--- MODO FAST (FEA simplificado) ---")
    result_fast = run_fea_integrated(test_params, mode='fast', scenario='perseverance')
    print(f"Escenario: {result_fast['scenario']['scenario_name']}")
    print(f"Carga de diseño: {result_fast['scenario']['design_load_N']:.0f} N")
    print(f"Rigidez: {result_fast['stiffness_kN_m']:.1f} kN/m")
    print(f"Masa: {result_fast['mass_kg']:.2f} kg")
    
    print("\n" + "="*70)
