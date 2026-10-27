#!/usr/bin/env python3
"""
export_top20_pareto.py
======================
Lee el JSON Pareto de fase3, rankea las 103 soluciones con un score
multi-objetivo compuesto, selecciona el Top 20 y genera los STL en:

  results/Final_Pareto/
  ├── spring_tire_000/
  │   ├── spring_tire_000.stl
  │   ├── spring_tire_000_parts/
  │   │   └── <Material>.stl
  │   └── spring_tire_000_metadata.json
  ├── ...
  └── ranking_report.json

Criterio de ranking (mayor = mejor):
  score = 0.20 * safety_factor_norm
        + 0.20 * life_cycles_norm
        + 0.15 * stiffness_norm
        + 0.15 * structural_efficiency_norm
        + 0.10 * energy_stored_norm
        + 0.10 * (1 - mass_norm)        ← menos masa = mejor
        + 0.05 * (1 - print_complexity_norm)
        + 0.05 * (1 - adhesion_risk_norm)

Candidatos con plastic_strain_percent > 0 son penalizados (-0.5).
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ruta base del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'scripts'))

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PARETO_JSON = BASE_DIR / 'results' / 'pareto' / 'pareto_spring_tire_20260314_223722.json'
OUTPUT_DIR  = BASE_DIR / 'results' / 'Final_Pareto'
TOP_N       = 20

# Pesos del score compuesto
WEIGHTS = {
    'safety_factor':           0.20,
    'life_cycles':             0.20,
    'stiffness_kN_m':          0.15,
    'structural_efficiency':   0.15,
    'energy_stored_mJ':        0.10,
    'mass_kg':                 0.10,   # se invierte: menos masa = mejor
    'print_complexity':        0.05,   # se invierte
    'adhesion_risk':           0.05,   # se invierte
}

PENALTY_PLASTIC_STRAIN = -0.5  # penalización si plastic_strain_percent > 0


# ---------------------------------------------------------------------------
# Funciones de ranking
# ---------------------------------------------------------------------------

def normalize_column(values: np.ndarray) -> np.ndarray:
    """Normaliza una columna al rango [0, 1]. Si rango == 0, retorna 0.5."""
    mn, mx = values.min(), values.max()
    if mx == mn:
        return np.full_like(values, 0.5)
    return (values - mn) / (mx - mn)


def compute_scores(solutions: list) -> np.ndarray:
    """
    Calcula el score compuesto para cada solución.
    Retorna array de shape (n,) con los scores en [0, 1] + penalizaciones.
    """
    n = len(solutions)

    def get_obj(sol, key, default=0.0):
        val = sol.get('objectives', {}).get(key)
        if val is None:
            val = sol.get('fea_results', {}).get(key, default)
        # life_cycles puede estar capped en 1e12, normalizar internamente
        return float(val)

    # Extraer métricas
    safety      = np.array([get_obj(s, 'safety_factor')          for s in solutions])
    life        = np.array([get_obj(s, 'life_cycles')             for s in solutions])
    stiff       = np.array([get_obj(s, 'stiffness_kN_m')          for s in solutions])
    eff         = np.array([get_obj(s, 'structural_efficiency')    for s in solutions])
    energy      = np.array([get_obj(s, 'energy_stored_mJ')         for s in solutions])
    mass        = np.array([get_obj(s, 'mass_kg',          default=5.0) for s in solutions])
    complexity  = np.array([get_obj(s, 'print_complexity', default=1.0) for s in solutions])
    adhesion    = np.array([get_obj(s, 'adhesion_risk',    default=1.0) for s in solutions])
    plastic     = np.array([
        float(s.get('fea_results', {}).get('plastic_strain_percent', 0.0))
        for s in solutions
    ])

    # Normalizar
    safety_n    = normalize_column(safety)
    life_n      = normalize_column(np.log1p(life))   # escala log (rangos enormes)
    stiff_n     = normalize_column(stiff)
    eff_n       = normalize_column(eff)
    energy_n    = normalize_column(energy)
    mass_n      = 1.0 - normalize_column(mass)        # invertir
    complexity_n = 1.0 - normalize_column(complexity) # invertir
    adhesion_n  = 1.0 - normalize_column(adhesion)    # invertir

    score = (
        WEIGHTS['safety_factor']         * safety_n +
        WEIGHTS['life_cycles']           * life_n +
        WEIGHTS['stiffness_kN_m']        * stiff_n +
        WEIGHTS['structural_efficiency'] * eff_n +
        WEIGHTS['energy_stored_mJ']      * energy_n +
        WEIGHTS['mass_kg']               * mass_n +
        WEIGHTS['print_complexity']      * complexity_n +
        WEIGHTS['adhesion_risk']         * adhesion_n
    )

    # Penalizar diseños con deformación plástica
    penalty = np.where(plastic > 0, PENALTY_PLASTIC_STRAIN, 0.0)
    score += penalty

    return score


def select_top_n(solutions: list, n: int = TOP_N):
    """Retorna los n mejores con sus scores."""
    scores = compute_scores(solutions)
    ranked_indices = np.argsort(scores)[::-1]  # mayor score primero
    top_indices = ranked_indices[:n]
    top = [(solutions[i], float(scores[i])) for i in top_indices]
    return top, scores


# ---------------------------------------------------------------------------
# Exportación STL (reutiliza fase6)
# ---------------------------------------------------------------------------

def export_top20():
    log.info("=" * 70)
    log.info("  EXPORT TOP 20 PARETO → Final_Pareto")
    log.info("=" * 70)

    # Importar dependencias del pipeline
    try:
        from utils.spring_tire_genome import decode_genome
        from utils.spring_tire_geometry_v4 import SpringTireGeometry, export_stl
    except ImportError as e:
        log.error(f"No se pudo importar módulo del pipeline: {e}")
        log.error("Asegúrate de ejecutar desde el directorio raíz del proyecto o con sys.path correcto.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Cargar Pareto
    # ----------------------------------------------------------------
    log.info(f"\nCargando: {PARETO_JSON}")
    with open(PARETO_JSON, 'r') as f:
        pareto_data = json.load(f)

    solutions = pareto_data['soluciones']
    log.info(f"  Total soluciones Pareto: {len(solutions)}")

    # ----------------------------------------------------------------
    # Ranking y selección Top 20
    # ----------------------------------------------------------------
    log.info(f"\nCalculando scores y seleccionando Top {TOP_N}...")
    top20, all_scores = select_top_n(solutions, TOP_N)

    log.info(f"\n{'Rank':>4}  {'ID':>4}  {'Score':>7}  {'Safety':>7}  {'Life(km)':>12}  "
             f"{'Stiff':>8}  {'Mass':>6}  {'Mats'}")
    log.info("-" * 80)

    ranking_rows = []
    for rank, (sol, score) in enumerate(top20, 1):
        sid  = sol['id']
        obj  = sol.get('objectives', {})
        fea  = sol.get('fea_results', {})
        sfty = obj.get('safety_factor', fea.get('safety_factor', '-'))
        life_km = fea.get('life_km', fea.get('fatigue', {}).get('life_km', '-'))
        stiff = obj.get('stiffness_kN_m', fea.get('stiffness_kN_m', '-'))
        mass  = obj.get('mass_kg', fea.get('mass_kg', '-'))
        mats  = sol.get('params', {}).get('material_names', [])

        sfty_s  = f"{sfty:.2f}" if isinstance(sfty, float) else str(sfty)
        life_s  = f"{life_km:.0f}" if isinstance(life_km, float) else str(life_km)
        stiff_s = f"{stiff:.1f}" if isinstance(stiff, float) else str(stiff)
        mass_s  = f"{mass:.3f}" if isinstance(mass, float) else str(mass)
        mats_s  = '+'.join(mats)

        log.info(f"{rank:>4}  #{sid:>3}  {score:>7.4f}  {sfty_s:>7}  {life_s:>12}  "
                 f"{stiff_s:>8}  {mass_s:>6}  {mats_s}")

        ranking_rows.append({
            'rank':       rank,
            'id':         sid,
            'score':      score,
            'safety_factor': sfty,
            'life_km':    life_km,
            'stiffness_kN_m': stiff,
            'mass_kg':    mass,
            'materials':  mats,
        })

    # ----------------------------------------------------------------
    # Importar subconjunto Pareto filtrado para fase6
    # ----------------------------------------------------------------
    from scripts.fase6_spring_tire_export import export_design_with_parts, MATERIAL_NAMES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"\nExportando STL → {OUTPUT_DIR}\n")
    exported  = []
    n_ok      = 0
    n_fail    = 0

    for rank, (sol, score) in enumerate(top20, 1):
        sol_id   = sol['id']
        genome   = np.array(sol['genome'])
        params   = decode_genome(genome)
        objectives = sol.get('objectives', {})
        fem3d    = sol.get('fem3d_validation')

        # Carpeta individual por diseño
        design_dir = str(OUTPUT_DIR / f"spring_tire_{sol_id:03d}")

        log.info(f"[{rank:02d}/{TOP_N}]  #ID={sol_id}  score={score:.4f}")

        result = export_design_with_parts(
            sol_id=sol_id,
            genome=genome,
            params=params,
            output_dir=design_dir,
            objectives=objectives,
            fem3d_data=fem3d,
        )

        if 'error' in result:
            n_fail += 1
            log.error(f"  ✗ #{sol_id}: {result['error']}")
        else:
            n_ok += 1
            result['rank']  = rank
            result['score'] = score
            exported.append(result)
            log.info(f"  ✓ Exportado OK → {design_dir}")

    # ----------------------------------------------------------------
    # Guardar ranking_report.json
    # ----------------------------------------------------------------
    report = {
        'pareto_file':   str(PARETO_JSON),
        'output_dir':    str(OUTPUT_DIR),
        'top_n':         TOP_N,
        'n_exported_ok': n_ok,
        'n_failed':      n_fail,
        'weights':       WEIGHTS,
        'penalty_plastic_strain': PENALTY_PLASTIC_STRAIN,
        'ranking':       ranking_rows,
    }

    report_path = OUTPUT_DIR / 'ranking_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=_json_safe)

    log.info("\n" + "=" * 70)
    log.info(f"  COMPLETADO: {n_ok}/{TOP_N} exportados OK  |  {n_fail} fallidos")
    log.info(f"  Carpeta: {OUTPUT_DIR}")
    log.info(f"  Reporte: {report_path}")
    log.info("=" * 70)

    return report


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    export_top20()
