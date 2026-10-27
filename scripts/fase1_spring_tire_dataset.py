#!/usr/bin/env python3
"""
fase1_spring_tire_dataset.py - Fase 1: Dataset NASA Spring Tire v5.0

Genera dataset de entrenamiento con geometría Spring Tire real y análisis FEA.
"""

import os
import sys
import json
import time
import numpy as np
import logging
import multiprocessing
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Importaciones core ----
from utils.spring_tire_genome import (
    N_GENES, decode_genome, encode_params,
    create_baseline_genome, validate_genome
    # generate_latin_hypercube_samples se define localmente (no existe en genome.py)
)
from utils.spring_tire_geometry_v4 import SpringTireGeometry, export_stl
# BUG#1: Soft import — geometry_v5 puede no existir sin bloquear el script
try:
    from utils.spring_tire_geometry_v5 import SpringTireGeometryV5, export_stl_v5
    V5_AVAILABLE = True
except ImportError:
    V5_AVAILABLE = False
    log.warning("spring_tire_geometry_v5 no disponible — usando v4")
from utils.spring_tire_fea import run_spring_tire_fea

# BUG#3: Importar módulos pesados a nivel de módulo (top-level) para que los
# workers de multiprocessing no repitan el import en cada tarea (lento + propenso
# a errores de pickle/fork). Se hace con try/except para mantener el fallback.
try:
    from utils.spring_tire_fea_integrated import get_rover_load_scenario as _get_rover_load
    from utils.spring_tire_fem3d import run_fem3d_spoke as _run_fem3d_spoke
    from utils.spring_tire_fem3d import compute_correction_factors as _compute_correction
    FEM3D_AVAILABLE_GLOBAL = True
except ImportError:
    FEM3D_AVAILABLE_GLOBAL = False
    _get_rover_load = None
    _run_fem3d_spoke = None
    _compute_correction = None

# Cache del escenario de carga (se computa una vez, compartido por todos los workers via fork)
try:
    _LOAD_SCENARIO = _get_rover_load('perseverance') if _get_rover_load else None
    F_DESIGN_GLOBAL = float(_LOAD_SCENARIO['design_load_N']) if _LOAD_SCENARIO else 500.0
except Exception:
    F_DESIGN_GLOBAL = 500.0


def generate_random_genome() -> np.ndarray:
    """Genera un genoma aleatorio válido."""
    while True:
        genome = np.random.random(N_GENES)
        is_valid, _ = validate_genome(genome)
        if is_valid:
            return genome


def generate_latin_hypercube_samples(n_samples: int) -> np.ndarray:
    """Genera muestras usando Latin Hypercube Sampling."""
    samples = np.zeros((n_samples, N_GENES))

    for i in range(N_GENES):
        # Dividir [0,1] en n_samples intervalos
        intervals = np.linspace(0, 1, n_samples + 1)
        # Seleccionar punto aleatorio en cada intervalo
        points = intervals[:-1] + np.random.random(n_samples) / n_samples
        # Mezclar
        np.random.shuffle(points)
        samples[:, i] = points

    return samples


def create_dataset_entry(genome: np.ndarray, idx: int,
                         export_stl_path: str = None,
                         use_v5_geometry: bool = False) -> Dict:
    """
    Crea una entrada del dataset con genoma, geometría y propiedades FEA.
    Guarda ambos análisis (FEM 2D y FEM 3D) para que la GNN aprenda las diferencias.

    Args:
        genome: Genoma del diseño
        idx: Índice del dataset
        export_stl_path: Si se proporciona, exporta el STL a esta ruta
        use_v5_geometry: Si True, usa SpringTireGeometryV5 (sólidos 3D)

    Returns:
        Dict con toda la información del diseño
    """
    # Decodificar genoma
    params = decode_genome(genome)

    # BUG#1: usar V5_AVAILABLE para evitar NameError si v5 no existe
    if use_v5_geometry and V5_AVAILABLE:
        tire = SpringTireGeometryV5(params)
    elif use_v5_geometry:
        log.warning(f"Muestra {idx}: geometry_v5 no disponible, usando v4")
        tire = SpringTireGeometry(params)
    else:
        tire = SpringTireGeometry(params)

    mesh = tire.generate()

    if mesh is None:
        return None

    # Exportar STL si se solicita
    if export_stl_path:
        try:
            mesh.export(export_stl_path)
            log.info(f"  💾 STL exportado: {export_stl_path}")
        except Exception as e:
            log.warning(f"  No se pudo exportar STL: {e}")

    mesh_stats = {
        'n_vertices': len(mesh.vertices),
        'n_faces': len(mesh.faces),
        'volume_cm3': float(mesh.volume * 0.001) if mesh.is_watertight else 0.0,
        'bounds_mm': mesh.bounds.tolist(),
        'is_watertight': mesh.is_watertight
    }

    # ========== Carga de diseño (usa caché global, no imports dentro del worker) ==========
    F_design = F_DESIGN_GLOBAL
    params_with_load = dict(params)
    params_with_load['F_vertical'] = F_design

    # ========== FEA 2D (análisis completo) ==========
    fea_2d = run_spring_tire_fea(params_with_load)  # usa F_design, no 500N

    # ========== FEM 3D (análisis de alto fidelity) ==========
    fea_3d = {}
    if FEM3D_AVAILABLE_GLOBAL and _run_fem3d_spoke is not None:
        try:
            fea_3d = _run_fem3d_spoke(params_with_load, F_vertical=F_design)
            # Calcular factores de corrección 2D→3D si ambos análisis existen
            if fea_3d and fea_2d and _compute_correction is not None:
                fea_2d_for_compare = {
                    'stiffness_kN_m': fea_2d.get('stiffness_kN_m', 1.0),
                    'max_deflection_mm': fea_2d.get('max_deflection_mm', 1.0),
                    'sigma_vm_max_MPa': fea_2d.get('sigma_vm_max_MPa', 1.0),
                }
                correction_factors = _compute_correction(fea_2d_for_compare, fea_3d)
                fea_3d['correction_factors_2d_to_3d'] = correction_factors
        except Exception as e:
            import traceback as _tb
            log.warning(
                f"  Muestra {idx}: FEM 3D falló - {e}\n"
                f"{_tb.format_exc(limit=6)}"
            )
            fea_3d = {}

    # ========== Determinar qué análisis usar para targets ==========
    # Por defecto, FEM 3D tiene prioridad si está disponible
    use_3d = bool(fea_3d) and 'k_vertical_kNm' in fea_3d

    # BUG#6: inicializar primary_fea por defecto por si se añaden nuevos fea_type en el futuro
    primary_fea = {}
    if not use_3d:
        log.warning(f"  Muestra {idx}: Usando FEA 2D (FEM 3D no disponible o incompleto)")
        primary_fea = fea_2d
        fea_type = 'fem2d'
    else:
        primary_fea = fea_3d
        fea_type = 'fem3d'

    # BUG#11 fix: densidad debe reflejar los materiales reales del genoma, no 1.15 fijo.
    # Tabla de densidades (g/cm³) por mat_id (misma que genome.py MATERIALS)
    _DENSITY_BY_MATID = [1.12, 1.15, 1.18, 1.06, 1.04, 1.25]  # TPU95A,TPU85A,TPU75D,Nylon,ABS,PLA
    material_sectors_raw = params.get('material_sectors', [0]*8)
    mat_densities = [_DENSITY_BY_MATID[int(m) % 6] for m in material_sectors_raw]
    density = float(np.mean(mat_densities)) if mat_densities else 1.15
    vol_cm3_mesh = mesh_stats['volume_cm3']
    mass_kg_est = (vol_cm3_mesh * density) / 1000.0 if vol_cm3_mesh > 0 else fea_2d.get('mass_kg', 0.0)

    # ========== Construir targets ==========
    if use_3d:
        k_vertical = fea_3d.get('k_vertical_kNm', 0.0)
        # BUG#9: unified max_load_N — misma fórmula en ambos paths
        sf_3d = fea_3d.get('SF_strength', 1.0)
        targets = {
            'stiffness_kN_m': k_vertical,
            'lateral_stiffness_kN_m': fea_2d.get('lateral_stiffness_kN_m', 0.0),
            'torsional_stiffness_Nm_rad': fea_2d.get('torsional_stiffness_Nm_rad', 0.0),
            'max_deflection_mm': fea_3d.get('max_displacement_mm', 0.0),
            'max_load_N': F_design * sf_3d,  # carga real de diseño × SF aplicado
            'mass_kg': mass_kg_est,
            'plastic_strain_percent': fea_3d.get('plastic_strain_pct', 0.0),
            'structural_efficiency': (k_vertical / mass_kg_est) if mass_kg_est > 0 else 0.0,
            'safety_factor': sf_3d,
            'adhesion_risk': fea_2d.get('adhesion_risk', 0.0),
            'print_complexity': fea_2d.get('print_complexity', 0.0),
            # Memoria de forma desde FEM 3D
            'recovery_ratio': fea_3d.get('recovery_ratio', 0.0),
            'shape_memory_ratio': fea_3d.get('shape_memory_ratio', 0.0),
            'energy_stored_mJ': fea_3d.get('energy_stored_mJ', 0.0),
            # Fatiga desde FEM 3D
            'life_cycles': fea_3d.get('life_cycles', 0.0),
            'life_km': fea_3d.get('life_km', 0.0),
            'goodman_ratio': fea_3d.get('goodman_ratio', 0.0),
            'fatigue_risk': fea_3d.get('fatigue_risk', 'UNKNOWN'),
        }
    else:
        # BUG#9: unified max_load_N — 500N * SF igual que path FEM3D
        sf_2d = fea_2d.get('safety_factor', 1.0)
        targets = {
            'stiffness_kN_m': fea_2d.get('stiffness_kN_m', 0.0),
            'lateral_stiffness_kN_m': fea_2d.get('lateral_stiffness_kN_m', 0.0),
            'torsional_stiffness_Nm_rad': fea_2d.get('torsional_stiffness_Nm_rad', 0.0),
            'max_deflection_mm': fea_2d.get('max_deflection_mm', 0.0),
            'max_load_N': F_design * sf_2d,  # carga real de diseño × SF aplicado
            'mass_kg': fea_2d.get('mass_kg', mass_kg_est),
            'plastic_strain_percent': fea_2d.get('plastic_strain_percent', 0.0),
            'structural_efficiency': fea_2d.get('structural_efficiency', 0.0),
            'safety_factor': sf_2d,
            'adhesion_risk': fea_2d.get('adhesion_risk', 0.0),
            'print_complexity': fea_2d.get('print_complexity', 0.0),
            # Memoria de forma desde FEA 2D (si existe)
            'recovery_ratio': fea_2d.get('recovery_ratio', 0.0),
            'shape_memory_ratio': fea_2d.get('shape_memory_ratio', 0.0),
            'energy_stored_mJ': fea_2d.get('energy_stored_mJ', 0.0),
            # Fatiga desde FEA 2D (si existe)
            'life_cycles': fea_2d.get('life_cycles', 0.0),
            'life_km': fea_2d.get('life_km', 0.0),
            'goodman_ratio': fea_2d.get('goodman_ratio', 0.0),
            'fatigue_risk': fea_2d.get('fatigue_risk', 'UNKNOWN'),
        }

    # ====== Filtro de calidad física ======
    # BUG#8: watertight_ok removido del filtro DURO — trimesh.concatenate casi nunca
    # produce mallas watertight (uniones no-manifold), esto descartaba ~90-100% muestras.
    # Se guarda como campo informativo para filtrado opcional en postproceso.
    mass_ok   = targets.get('mass_kg', 0.0) > 0.05
    sf_ok     = targets.get('safety_factor', 0.0) >= 1.5
    defl_ok   = targets.get('max_deflection_mm', 999.0) < 100.0
    strain_ok = targets.get('plastic_strain_percent', 999.0) < 15.0
    stiff_ok  = 0.5 <= targets.get('stiffness_kN_m', 0.0) <= 5000.0

    # Advertencia (no rechazo) si la malla no es watertight
    if not mesh_stats.get('is_watertight', True):
        log.debug(f"  Muestra {idx}: malla no-watertight (informativo, no filtrada)")

    if not all([mass_ok, sf_ok, defl_ok, strain_ok, stiff_ok]):
        reasons = []
        if not mass_ok:   reasons.append(f"mass={targets.get('mass_kg', 0):.3f}kg")
        if not sf_ok:     reasons.append(f"SF={targets.get('safety_factor', 0):.2f}")
        if not defl_ok:   reasons.append(f"defl={targets.get('max_deflection_mm', 0):.1f}mm")
        if not strain_ok: reasons.append(f"strain={targets.get('plastic_strain_percent', 0):.1f}%")
        if not stiff_ok:  reasons.append(f"k={targets.get('stiffness_kN_m', 0):.1f}kN/m")
        log.debug(f"  Muestra {idx} filtrada: {', '.join(reasons)}")
        return None

    return {
        'id': idx,
        'genome': genome.tolist(),
        'params': params,
        'mesh_stats': mesh_stats,
        'fea_results': {
            'primary': primary_fea,
            'fea_2d': fea_2d,
            'fea_3d': fea_3d,
            'fea_type_used': fea_type,
            'has_3d': use_3d,
        },
        'targets': targets
    }


# --------------------------------------------------------------------------
# Helpers para paralelización (deben estar a nivel de módulo para pickle)
# --------------------------------------------------------------------------

def _worker_fn(args):
    """Worker de nivel módulo (pickle-able) para multiprocessing.Pool."""
    # Fix F: BLAS/OpenBLAS single-thread — previene SIGALRM deadlock en matrices grandes
    import os as _os
    for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        _os.environ[_env] = '1'

    genome_array, idx = args
    try:
        return create_dataset_entry(genome_array, idx)
    except Exception as e:
        import traceback as _tb
        log.warning(f"Worker {idx} falló: {e}\n{_tb.format_exc(limit=4)}")
        return None


def _get_ram_used_gb() -> float:
    """RAM del sistema usada en GB. Retorna 0 si psutil no está disponible."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().used / (1024 ** 3)
    return 0.0


def _auto_n_workers(ram_limit_gb: float) -> int:
    """
    Calcula n_workers seguros según CPUs disponibles y RAM libre.
    Cap máx: 8 workers (para no saturar I/O de disco con mallas MSH).
    Estimado: ~1.5 GB por worker.
    """
    cpu_count = max(1, os.cpu_count() or 4)
    if PSUTIL_AVAILABLE:
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        workers_by_ram = max(1, int(avail_gb / 1.5))
        return min(cpu_count, workers_by_ram, 8)
    return min(cpu_count, 4)  # conservador sin psutil


def _save_dataset_checkpoint(dataset: list, output_dir: str, stats_only: bool = False) -> str:
    """
    Guarda el dataset actual a disco (checkpoint incremental).
    Sobreescribe el archivo anterior — es idempotente y seguro interrumpir.
    """
    if not dataset:
        return None

    all_targets = {k: [] for k in dataset[0]['targets'].keys()}
    for entry in dataset:
        for k, v in entry['targets'].items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                all_targets[k].append(v)

    stats = {}
    for k, values in all_targets.items():
        if len(values) > 0:
            stats[k] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

    dataset_info = {
        'version': '5.0',
        'n_samples': len(dataset),
        'n_genes': N_GENES,
        'gene_names': [
            'd_ext_ratio', 'd_rim_ratio', 'width_ratio', 'n_radios_ratio',
            'wall_thickness_ratio', 'angle_cross_ratio',
            'material_sector_0', 'material_sector_1', 'material_sector_2',
            'material_sector_3', 'material_sector_4', 'material_sector_5',
            'material_sector_6', 'material_sector_7',
            'n_layers_ratio', 'spoke_thickness_ratio',
            'hub_angle_cross_ratio', 'tread_rim_factor_ratio'
        ],
        'target_names': list(dataset[0]['targets'].keys()),
        'stats': stats,
        'dataset': dataset
    }

    def _json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    output_path = os.path.join(output_dir, 'spring_tire_dataset.json')
    # Escribir a temporal primero, luego renombrar (atómico — evita JSON corrupto)
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(dataset_info, f, indent=2, default=_json_safe)
    os.replace(tmp_path, output_path)
    return output_path


def generate_dataset(n_samples: int = 500, output_dir: str = 'results/data',
                     strategies: List[str] = None,
                     export_candidates: bool = True,
                     n_candidates: int = 6,
                     ram_limit_gb: float = 25.0,
                     n_workers: int = 0,
                     checkpoint_every: int = 500) -> str:
    """
    Genera dataset completo de Spring Tire.

    Args:
        n_samples: Número de muestras a generar
        output_dir: Directorio de salida
        strategies: Lista de estrategias de material a incluir
        export_candidates: Si True, exporta STL de ejemplo a candidatos/
        n_candidates: Número de STL de ejemplo a exportar
        checkpoint_every: Guardar checkpoint cada N muestras aceptadas (0 = solo al final)

    Returns:
        Ruta al archivo dataset.json
    """
    log.info("="*70)
    log.info("FASE 1: Generando Dataset NASA Spring Tire v5.0")
    log.info("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Fix G: RESUME MODE — cargar muestras previas si existe checkpoint
    _resume_path = os.path.join(output_dir, 'spring_tire_dataset.json')
    _resumed_dataset = []
    _resumed_ids = set()
    if os.path.exists(_resume_path):
        try:
            with open(_resume_path, 'r') as _f:
                _prev = json.load(_f)
            _resumed_dataset = _prev.get('dataset', [])
            _resumed_ids = {int(e['id']) for e in _resumed_dataset}
            log.info(f"RESUME: cargadas {len(_resumed_dataset)} muestras previas de {_resume_path}")
        except Exception as _e:
            log.warning(f"RESUME: no se pudo cargar checkpoint ({_e}) — desde cero")
            _resumed_dataset = []
            _resumed_ids = set()

    # Crear carpeta para candidatos STL
    if export_candidates:
        os.makedirs('candidatos', exist_ok=True)
        log.info("STL de ejemplo se guardarán en: candidatos/")

    if strategies is None:
        strategies = ['tpu_homogeneous', 'flexible_rigid', 'graded', 'nylon_core']

    dataset = list(_resumed_dataset)  # Fix G: partir de checkpoint

    # 1. Generar muestras baseline por estrategia de material
    log.info(f"\nGenerando {len(strategies)} muestras baseline de material...")
    for strategy in strategies:
        if _resumed_ids:  # Fix G: saltear baselines si ya tenemos datos previos
            log.info(f"  ↷ Saltando baseline '{strategy}' (resume activo)")
            continue
        genome = create_baseline_genome(strategy)
        entry = create_dataset_entry(genome, len(dataset))
        if entry:
            entry['strategy'] = strategy
            entry['pantograph_style'] = 'medium_cross'
            dataset.append(entry)
            log.info(f"  ✓ {strategy}: stiffness={entry['targets']['stiffness_kN_m']:.1f} kN/m")

    # 1b. Generar muestras baseline por estilo pantográfico
    PANTOGRAPH_STYLES = {
        'sparse_steep':   [0.2, 0.9, 0.0, 0.333],   # pocos radios muy inclinados
        'dense_shallow':  [0.85, 0.1, 0.8, 0.333],  # muchos radios suaves
        'spring_lunar':   [0.4, 0.7, 0.4, 0.333],   # tipo muelle lunar
        'hub_crossed':    [0.5, 0.5, 0.5, 0.75],    # hub muy cruzado
        'minimal_spring': [0.1, 0.95, 0.1, 0.2],    # mínimo extremo
        'maxdense':       [0.95, 0.05, 0.9, 0.5],   # máxima densidad
    }
    log.info(f"\nGenerando {len(PANTOGRAPH_STYLES)} muestras baseline pantográficas...")
    for style_name, panto_genes in PANTOGRAPH_STYLES.items():
        genome = create_baseline_genome('tpu_homogeneous')
        # genes 14-17: n_layers, spoke_thickness, hub_angle_cross, tread_rim_factor
        genome[14], genome[15], genome[16], genome[17] = panto_genes
        entry = create_dataset_entry(genome, len(dataset))
        if entry:
            entry['strategy'] = 'tpu_homogeneous'
            entry['pantograph_style'] = style_name
            dataset.append(entry)
            log.info(f"  ✓ {style_name}: stiffness={entry['targets']['stiffness_kN_m']:.1f} kN/m")

    # 2. Generar muestras LHS (Latin Hypercube Sampling)
    # BUG#2: n_lhs debe descontar TODOS los baselines ya generados (no solo len(strategies))
    n_already_in = len(dataset)
    n_lhs = max(0, n_samples - n_already_in)
    log.info(f"Baselines generados: {n_already_in}, LHS restantes: {n_lhs}")
    if n_lhs > 0:
        log.info(f"\nGenerando {n_lhs} muestras LHS...")
        lhs_samples = generate_latin_hypercube_samples(n_lhs)
    else:
        log.info(f"\nNo se generan muestras LHS (n_samples={n_samples} <= estrategias={len(strategies)})")
        lhs_samples = np.array([]).reshape(0, N_GENES)

    if len(lhs_samples) > 0:
        # ── Paralelización con multiprocessing + guard de RAM ──────────────────
        # BUG#2: n_workers=0 significa auto-detect
        eff_workers = n_workers if n_workers > 0 else _auto_n_workers(ram_limit_gb)
        log.info(f"Usando {eff_workers} workers paralelos (límite RAM: {ram_limit_gb:.0f} GB)")
        log.info(f"RAM usada ahora: {_get_ram_used_gb():.1f} GB")

        # Preparar argumentos para los workers
        start_idx = len(dataset)  # índices continuos
        worker_args = [
            (genome.copy(), start_idx + i)
            for i, genome in enumerate(lhs_samples)
        ]

        n_done = 0
        n_ok = 0
        batch_size = max(eff_workers * 4, 64)  # tamaño de lote para controlar memoria
        lhs_accepted = []  # para export_candidates post-pool

        with tqdm(total=len(lhs_samples), desc="LHS samples (paralelo)") as pbar:
            # Procesar en lotes para liberar RAM entre lotes
            for batch_start in range(0, len(worker_args), batch_size):
                batch = worker_args[batch_start:batch_start + batch_size]

                # Guard de RAM (pre-lote): esperar si estamos cerca del límite
                while _get_ram_used_gb() > ram_limit_gb:
                    log.debug(f"RAM usada {_get_ram_used_gb():.1f}GB > límite {ram_limit_gb}GB, esperando...")
                    time.sleep(1.0)

                with multiprocessing.Pool(
                    processes=eff_workers,
                    maxtasksperchild=16
                ) as pool:
                    # Fix H: apply_async+timeout 60s — mata workers colgados por BLAS
                    _async_results = [
                        (pool.apply_async(_worker_fn, (arg,)), arg[1])
                        for arg in batch
                    ]
                    for _ar, _task_idx in _async_results:
                        try:
                            entry = _ar.get(timeout=60)
                        except multiprocessing.TimeoutError:
                            log.warning(f"Worker timeout (60s) idx={_task_idx} — saltando")
                            entry = None
                        except Exception as _ex:
                            log.warning(f"Worker error idx={_task_idx}: {_ex}")
                            entry = None
                        if entry:
                            entry['strategy'] = 'lhs_random'
                            dataset.append(entry)
                            lhs_accepted.append(entry)
                            n_ok += 1
                            # ── Checkpoint incremental ──────────────────────
                            if checkpoint_every > 0 and len(dataset) % checkpoint_every == 0:
                                _save_dataset_checkpoint(dataset, output_dir)
                                log.info(f"  ✓ Checkpoint: {len(dataset)} muestras guardadas")
                        n_done += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'ok': n_ok,
                            'RAM_GB': f"{_get_ram_used_gb():.1f}"
                        })
                        # BUG#5: guard de RAM DENTRO del loop (cada 10 resultados)
                        if n_done % 10 == 0 and _get_ram_used_gb() > ram_limit_gb:
                            log.warning(f"RAM {_get_ram_used_gb():.1f}GB > límite durante lote, pausando 2s...")
                            time.sleep(2.0)

        log.info(f"Muestras LHS: {n_ok}/{len(lhs_samples)} aceptadas")

        # BUG#4: export_candidates en el path paralelo
        # Se exportan los primeros n_candidates de las muestras LHS aceptadas
        if export_candidates and lhs_accepted:
            log.info(f"Exportando {min(n_candidates, len(lhs_accepted))} STL candidatos LHS...")
            for ci, cand_entry in enumerate(lhs_accepted[:n_candidates]):
                try:
                    g = np.array(cand_entry['genome'])
                    p = decode_genome(g)
                    if V5_AVAILABLE:
                        tire = SpringTireGeometryV5(p)
                    else:
                        tire = SpringTireGeometry(p)
                    cmesh = tire.generate()
                    if cmesh is not None:
                        stl_path = f"candidatos/lhs_{ci:03d}_idx{cand_entry['id']}.stl"
                        cmesh.export(stl_path)
                        log.info(f"  Candidato STL: {stl_path}")
                except Exception as ex:
                    log.debug(f"No se pudo exportar candidato LHS {ci}: {ex}")

    # 4. Guardar dataset (checkpoint final)
    log.info("\nCalculando estadísticas...")
    if not dataset:
        log.error("Dataset vacío — ninguna muestra pasó los filtros de calidad")
        return None

    output_path = _save_dataset_checkpoint(dataset, output_dir)

    log.info(f"\nDataset guardado: {output_path}")
    log.info(f"Total de muestras: {len(dataset)}")

    # Recalcular stats para el log final
    all_targets = {k: [] for k in dataset[0]['targets'].keys()}
    for entry in dataset:
        for k, v in entry['targets'].items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                all_targets[k].append(v)
    log.info(f"\nEstadísticas de targets:")
    for k, values in all_targets.items():
        if len(values) > 0:
            log.info(f"  {k}: min={float(np.min(values)):.2f}, max={float(np.max(values)):.2f}, mean={float(np.mean(values)):.2f}")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generar dataset Spring Tire')
    parser.add_argument('--n-samples', type=int, default=500, help='Número de muestras')
    parser.add_argument('--output-dir', type=str, default='results/data', help='Directorio de salida')
    parser.add_argument('--ram-limit-gb', type=float, default=25.0,
                        help='Límite de RAM en GB para el pool paralelo (default: 25.0)')
    parser.add_argument('--n-workers', type=int, default=0,
                        help='Número de workers paralelos (0 = auto-detect)')
    parser.add_argument('--checkpoint-every', type=int, default=500,
                        help='Guardar checkpoint cada N muestras aceptadas (default: 500)')
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        ram_limit_gb=args.ram_limit_gb,
        n_workers=args.n_workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == '__main__':
    main()
