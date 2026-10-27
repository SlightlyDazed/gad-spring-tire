"""
spring_tire_genome.py - Genoma para NASA Spring Tire v6.0

Genoma de 18 variables que definen completamente una Spring Tire:
- 6 variables geométricas básicas
- 8 variables de material (sectores)
- 4 variables de patrón pantográfico
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

log = logging.getLogger(__name__)

# Número de variables del genoma
N_GENES = 18  # 6 geométricas + 8 materiales + 4 pantográficas
N_SECTORS = 8  # Número de sectores para distribución de materiales

# Rangos de parámetros (valores normalizados 0-1)
GENE_RANGES = {
    # Variables geométricas (índices 0-5)
    'd_ext_ratio': (0.0, 1.0),          # → d_ext: 300-500 mm
    'd_rim_ratio': (0.0, 1.0),          # → d_rim: 100-250 mm
    'width_ratio': (0.0, 1.0),          # → width: 50-250 mm
    'n_radios_ratio': (0.0, 1.0),       # → n_radios: 6-28
    'wall_thickness_ratio': (0.0, 1.0), # → wall: 2-10 mm
    'angle_cross_ratio': (0.0, 1.0),    # → angle: 30-80°

    # Variables de material (índices 6-13)
    # Cada sector puede tener material 0-5
}

# Rangos físicos
PHYSICAL_RANGES = {
    'd_ext': (300.0, 500.0),           # mm
    'd_rim': (100.0, 250.0),           # mm
    'width': (50.0, 250.0),            # mm
    'n_radios': (6, 28),               # entero
    'wall_thickness': (2.0, 10.0),     # mm (tread y rim)
    'angle_cross': (30.0, 80.0),       # grados (spokes exteriores)
    # Pantográficas (genes 14-17)
    'n_layers': (1, 15),               # capas Z de radios
    'spoke_thickness': (2.0, 12.0),    # mm (grosor independiente de spokes)
    'hub_angle_cross': (0.0, 60.0),    # grados de cruce del hub
    'tread_rim_factor': (1.0, 4.0),    # multiplicador del grosor tread/rim
}

# Catálogo de materiales
MATERIAL_CATALOG = {
    0: {'name': 'TPU_95A', 'E_MPa': 50.0, 'sigma_yield_MPa': 25.0, 'density_g_cm3': 1.15, 'print_temp_C': 230},
    1: {'name': 'TPU_85A', 'E_MPa': 25.0, 'sigma_yield_MPa': 15.0, 'density_g_cm3': 1.12, 'print_temp_C': 220},
    2: {'name': 'TPU_75D', 'E_MPa': 200.0, 'sigma_yield_MPa': 40.0, 'density_g_cm3': 1.20, 'print_temp_C': 240},
    3: {'name': 'Nylon_PA12', 'E_MPa': 1800.0, 'sigma_yield_MPa': 45.0, 'density_g_cm3': 1.01, 'print_temp_C': 260},
    4: {'name': 'ABS', 'E_MPa': 2200.0, 'sigma_yield_MPa': 40.0, 'density_g_cm3': 1.05, 'print_temp_C': 250},
    5: {'name': 'PLA', 'E_MPa': 3500.0, 'sigma_yield_MPa': 55.0, 'density_g_cm3': 1.25, 'print_temp_C': 200},
}


def decode_genome(genome: np.ndarray) -> Dict:
    """
    Decodifica un genoma Spring Tire a parámetros físicos.

    Args:
        genome: Array de 18 elementos [0-1]

    Returns:
        Dict con parámetros físicos
    """
    if len(genome) != N_GENES:
        raise ValueError(f"Genoma debe tener {N_GENES} genes, tiene {len(genome)}")

    # Genes 0-5: Variables geométricas básicas
    d_ext = PHYSICAL_RANGES['d_ext'][0] + genome[0] * (PHYSICAL_RANGES['d_ext'][1] - PHYSICAL_RANGES['d_ext'][0])
    d_rim = PHYSICAL_RANGES['d_rim'][0] + genome[1] * (PHYSICAL_RANGES['d_rim'][1] - PHYSICAL_RANGES['d_rim'][0])
    width = PHYSICAL_RANGES['width'][0] + genome[2] * (PHYSICAL_RANGES['width'][1] - PHYSICAL_RANGES['width'][0])
    # BUG D fix: usar round() antes de int() para genes enteros.
    # int() trunca: genome[3]=0.3 → n_radios=12, encode(12)→0.2727≠0.3 (drift).
    # round() es la inversa correcta de encode_params para variables discretas.
    n_radios = int(round(PHYSICAL_RANGES['n_radios'][0] + genome[3] * (PHYSICAL_RANGES['n_radios'][1] - PHYSICAL_RANGES['n_radios'][0])))
    wall_thickness = PHYSICAL_RANGES['wall_thickness'][0] + genome[4] * (PHYSICAL_RANGES['wall_thickness'][1] - PHYSICAL_RANGES['wall_thickness'][0])
    angle_cross = PHYSICAL_RANGES['angle_cross'][0] + genome[5] * (PHYSICAL_RANGES['angle_cross'][1] - PHYSICAL_RANGES['angle_cross'][0])

    # Genes 6-13: Materiales por sector
    # BUG: 5.999 causaba que mat_id=3 (Nylon) se decodificara como 2 (TPU_75D)
    # Decodificar materiales (genes 6 a 13)
    # BUG#18 fix: int(g*6.0) castiga los bordes bajo mutación gaussiana.
    # Usar np.round(g*5.0) para que cada material 0-5 tenga el mismo ancho.
    material_sectors = np.array(
        [int(np.clip(np.round(g * 5.0), 0, 5)) for g in genome[6:14]],
        dtype=int
    )

    # Genes 14-17: Patrón pantográfico
    # BUG D fix: mismo problema en n_layers (gen 14)
    n_layers = max(1, int(round(PHYSICAL_RANGES['n_layers'][0] + genome[14] * (PHYSICAL_RANGES['n_layers'][1] - PHYSICAL_RANGES['n_layers'][0]))))
    spoke_thickness = PHYSICAL_RANGES['spoke_thickness'][0] + genome[15] * (PHYSICAL_RANGES['spoke_thickness'][1] - PHYSICAL_RANGES['spoke_thickness'][0])
    hub_angle_cross = PHYSICAL_RANGES['hub_angle_cross'][0] + genome[16] * (PHYSICAL_RANGES['hub_angle_cross'][1] - PHYSICAL_RANGES['hub_angle_cross'][0])
    tread_rim_factor = PHYSICAL_RANGES['tread_rim_factor'][0] + genome[17] * (PHYSICAL_RANGES['tread_rim_factor'][1] - PHYSICAL_RANGES['tread_rim_factor'][0])

    # Info de materiales
    unique_materials = np.unique(material_sectors)
    material_names = [MATERIAL_CATALOG[int(m)]['name'] for m in unique_materials]

    return {
        'd_ext': d_ext,
        'd_rim': d_rim,
        'width': width,
        'n_radios': n_radios,
        'wall_thickness': wall_thickness,
        'angle_cross': angle_cross,
        'tread_pattern': 'smooth',
        'n_layers': n_layers,
        'spoke_thickness': spoke_thickness,
        'hub_angle_cross': hub_angle_cross,
        'tread_rim_factor': tread_rim_factor,
        'material_sectors': material_sectors,
        'material_names': material_names,
        'n_unique_materials': len(unique_materials),
    }


def encode_params(params: Dict) -> np.ndarray:
    """
    Codifica parámetros físicos a genoma.

    Args:
        params: Dict con parámetros físicos

    Returns:
        Array de 18 elementos [0-1]
    """
    genome = np.zeros(N_GENES)

    # Codificar variables geométricas (genes 0-5)
    genome[0] = (params['d_ext'] - PHYSICAL_RANGES['d_ext'][0]) / (PHYSICAL_RANGES['d_ext'][1] - PHYSICAL_RANGES['d_ext'][0])
    genome[1] = (params['d_rim'] - PHYSICAL_RANGES['d_rim'][0]) / (PHYSICAL_RANGES['d_rim'][1] - PHYSICAL_RANGES['d_rim'][0])
    genome[2] = (params['width'] - PHYSICAL_RANGES['width'][0]) / (PHYSICAL_RANGES['width'][1] - PHYSICAL_RANGES['width'][0])
    genome[3] = (params['n_radios'] - PHYSICAL_RANGES['n_radios'][0]) / (PHYSICAL_RANGES['n_radios'][1] - PHYSICAL_RANGES['n_radios'][0])
    genome[4] = (params['wall_thickness'] - PHYSICAL_RANGES['wall_thickness'][0]) / (PHYSICAL_RANGES['wall_thickness'][1] - PHYSICAL_RANGES['wall_thickness'][0])
    genome[5] = (params['angle_cross'] - PHYSICAL_RANGES['angle_cross'][0]) / (PHYSICAL_RANGES['angle_cross'][1] - PHYSICAL_RANGES['angle_cross'][0])

    # Codificar materiales (genes 6-13)
    material_sectors = params.get('material_sectors', np.zeros(8, dtype=int))
    for i in range(8):
        # BUG#18 fix: mapeo inverso correspondiente a int(np.round(g*5.0)).
        # El centro del intervalo para el material m es m / 5.0.
        genome[6 + i] = float(material_sectors[i]) / 5.0

    # BUG#7: codificar genes pantográficos 14-17 (antes siempre quedaban en 0)
    # BUG#7: default n_layers=1 (consistente con run_robust_fea default)
    genome[14] = (params.get('n_layers', 1) - PHYSICAL_RANGES['n_layers'][0]) / (PHYSICAL_RANGES['n_layers'][1] - PHYSICAL_RANGES['n_layers'][0])
    # BUG#7: default spoke_thickness=5.75mm consistente con baseline gene[15]=0.375
    genome[15] = (params.get('spoke_thickness', 5.75) - PHYSICAL_RANGES['spoke_thickness'][0]) / (PHYSICAL_RANGES['spoke_thickness'][1] - PHYSICAL_RANGES['spoke_thickness'][0])
    genome[16] = (params.get('hub_angle_cross', 0.0) - PHYSICAL_RANGES['hub_angle_cross'][0]) / (PHYSICAL_RANGES['hub_angle_cross'][1] - PHYSICAL_RANGES['hub_angle_cross'][0])
    genome[17] = (params.get('tread_rim_factor', 2.0) - PHYSICAL_RANGES['tread_rim_factor'][0]) / (PHYSICAL_RANGES['tread_rim_factor'][1] - PHYSICAL_RANGES['tread_rim_factor'][0])

    return np.clip(genome, 0, 1)


def create_baseline_genome(strategy: str = 'tpu_homogeneous') -> np.ndarray:
    """
    Crea un genoma baseline según estrategia.

    Args:
        strategy: 'tpu_homogeneous', 'flexible_rigid', 'graded', 'nylon_core'

    Returns:
        Genoma de 18 elementos
    """
    genome = np.zeros(N_GENES)

    # Parámetros geométricos base (valores medios)
    genome[0] = 0.5  # d_ext = 400 mm
    genome[1] = 0.5  # d_rim = 175 mm
    genome[2] = 0.5  # width = 150 mm
    genome[3] = 0.5  # n_radios = 17
    genome[4] = 0.5  # wall_thickness = 6 mm
    genome[5] = 0.5  # angle_cross = 55°
    # Pantográficas: valores medios por defecto
    genome[14] = 0.5  # n_layers = 8
    genome[15] = 0.375  # spoke_thickness = 2.0 + 0.375*10 = 5.75 mm
    genome[16] = 0.0   # hub_angle_cross = 0° (radial)
    genome[17] = 0.333  # tread_rim_factor = 2.0x

    # Materiales según estrategia
    if strategy == 'tpu_homogeneous':
        # Todo TPU 95A
        genome[6:14] = 0.0

    elif strategy == 'flexible_rigid':
        # Mitad TPU flexible, mitad TPU rígido
        genome[6:10] = 0.0   # Sectores 0-3: TPU 95A
        genome[10:14] = 0.4  # Sectores 4-7: TPU 75D

    elif strategy == 'graded':
        # Transición suave de materiales (0 a 5 y de vuelta)
        # BUG#18 fix: usar el esquema /5.0 en lugar de /6 para los valores de base
        genome[6]  = 0 / 5.0    # mat 0: TPU 95A
        genome[7]  = 1 / 5.0    # mat 1: TPU 85A
        genome[8]  = 2 / 5.0    # mat 2: TPU 75D
        genome[9]  = 3 / 5.0    # mat 3: Nylon PA12
        genome[10] = 4 / 5.0    # mat 4: ABS
        genome[11] = 5 / 5.0    # mat 5: PLA
        genome[12] = 4 / 5.0    # mat 4: ABS
        genome[13] = 3 / 5.0    # mat 3: Nylon PA12

    elif strategy == 'nylon_core':
        # Núcleo de Nylon PA12 (mat_id=3), exterior de TPU 95A (mat_id=0)
        # BUG#6 fix + BUG#18 fix: decode usa round(g*5.0), centro de mat_id=3 es 3/5.0=0.6
        genome[6:10] = 3 / 5.0   # Sectores 0-3: Nylon PA12 (mat_id=3)
        genome[10:14] = 0.0      # Sectores 4-7: TPU 95A (mat_id=0)

    elif strategy == 'abs_rim':
        # Rim de ABS (mat_id=4), radios de TPU 95A (mat_id=0)
        # BUG#6 fix + BUG#18 fix: centro de mat_id=4 es 4/5.0=0.8
        genome[6:8] = 4 / 5.0    # ABS (mat_id=4)
        genome[8:14] = 0.0       # TPU 95A (mat_id=0)

    return np.clip(genome, 0, 1)


def validate_genome(genome: np.ndarray) -> Tuple[bool, str]:
    """
    Valida que un genoma sea válido.

    Returns:
        (es_valido, mensaje_error)
    """
    if len(genome) != N_GENES:
        return False, f"Genoma debe tener {N_GENES} genes"

    if not np.all((genome >= 0) & (genome <= 1)):
        return False, "Todos los genes deben estar en [0, 1]"

    params = decode_genome(genome)

    # Validar consistencia geométrica
    if params['d_rim'] >= params['d_ext'] * 0.9:
        return False, "d_rim debe ser menor que d_ext"

    # BUG#7 fix: verificar gap suficiente para wall_thickness
    # gap = r_ext - r_rim (mm). Debe ser >= 2*wall_thickness + 10 mm mínimo
    r_ext = params['d_ext'] / 2.0
    r_rim = params['d_rim'] / 2.0
    wall_thick = float(params.get('wall_thickness', 3.0))
    gap = r_ext - r_rim
    min_gap = 2 * wall_thick + 10.0  # 10 mm extra para holgura de montaje
    if gap < min_gap:
        return False, f"Gap geométrico ({gap:.1f}mm) < 2*wall_thickness+10mm ({min_gap:.1f}mm)"

    if params['n_radios'] < 6:
        return False, "n_radios debe ser al menos 6"

    return True, "OK"


def mutate_genome(genome: np.ndarray, mutation_rate: float = 0.1,
                  sigma: float = 0.1) -> np.ndarray:
    """Muta un genoma."""
    mutated = genome.copy()

    for i in range(len(genome)):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, sigma)

    return np.clip(mutated, 0, 1)


def crossover_genome(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crossover de dos genomas."""
    n = len(parent1)
    point = np.random.randint(1, n)

    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])

    return child1, child2
