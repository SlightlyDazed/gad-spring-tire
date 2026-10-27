#!/usr/bin/env python3
"""
run_spring_tire_pipeline.py - Pipeline completo NASA Spring Tire v6.0

Ejecuta todo el flujo:
1. Fase 1: Generar dataset
2. Fase 2: Entrenar GNN (transfer learning)
3. Fase 3: Optimización NSGA-III + Surrogate Refresh FEM 3D
6. Fase 6: Exportar STL con separación por material

Cambios v6.0 respecto v5.0:
  - FIX: pareto_json se extrae del dict que devuelve fase3, no se construye
         con un timestamp independiente (que era diferente al real)
  - FIX: run_phase3() ahora pasa n_obj a run_nsga3_optimization()
  - FIX: se añaden --n-obj, --no-fem3d, --refresh-every, --top-k,
         --final-validate, --mesh-size al argparse
  - FIX: check_dependencies() incluye skfem, gmsh, meshio (FEM 3D)
  - FIX: error claro cuando --phase 6 se llama sin --pareto
  - FIX: si fase3 devuelve None, se aborta con mensaje en vez de continuar
  - FIX: pip install sugerido solo para paquetes pip (gmsh/meshio no son
         solo pip en todos los sistemas)
"""

import os
import sys
import glob
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verificación de dependencias
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    """Verifica dependencias instaladas e informa cuáles faltan."""

    # (nombre_import, nombre_pip, obligatorio)
    DEPS = [
        ('numpy',          'numpy',             True),
        ('trimesh',        'trimesh',            True),
        ('pymoo',          'pymoo',              True),
        ('sklearn',        'scikit-learn',       False),
        ('skfem',          'scikit-fem',         False),   # FEM 3D
        ('gmsh',           'gmsh',               False),   # FEM 3D
        ('meshio',         'meshio',             False),   # FEM 3D
        ('torch',          'torch',              False),   # GNN Fase 2
        ('torch_geometric','torch_geometric',    False),   # GNN Fase 2
    ]

    log.info("=" * 70)
    log.info("  VERIFICACIÓN DE DEPENDENCIAS")
    log.info("=" * 70)

    missing_required = []
    missing_optional = []

    for import_name, pip_name, required in DEPS:
        try:
            __import__(import_name)
            log.info(f"  ✓ {import_name}")
        except ImportError:
            tag = "REQUERIDO" if required else "opcional"
            log.info(f"  ✗ {import_name}  [{tag}]")
            if required:
                missing_required.append(pip_name)
            else:
                missing_optional.append(pip_name)

    if missing_required:
        log.error(f"\n  Dependencias requeridas faltantes:")
        log.error(f"    pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        log.warning(f"\n  Dependencias opcionales faltantes (algunas funciones desactivadas):")
        log.warning(f"    pip install {' '.join(missing_optional)}")
        log.warning(f"    (FEM 3D requiere skfem + gmsh + meshio)")
        log.warning(f"    (GNN Fase 2 requiere torch + torch_geometric)")

    log.info(f"\n  ✓ Dependencias requeridas OK")
    return True


# ---------------------------------------------------------------------------
# Wrappers de cada fase
# ---------------------------------------------------------------------------

def run_phase1(n_samples: int = 500,
               output_dir: str = 'results/data') -> str:
    """
    Fase 1: Generar dataset.
    Returns: ruta al JSON del dataset.
    """
    log.info("\n" + "=" * 70)
    log.info("  FASE 1: Generando Dataset")
    log.info("=" * 70)

    from fase1_spring_tire_dataset import generate_dataset

    dataset_path = generate_dataset(
        n_samples=n_samples,
        output_dir=output_dir,
    )
    return dataset_path


def run_phase2(dataset_path: str,
               pretrained_dir: str = None,
               output_dir: str = 'results/models') -> dict:
    """
    Fase 2: Transfer Learning GNN.
    Returns: dict con info del modelo, o None si falla.
    """
    log.info("\n" + "=" * 70)
    log.info("  FASE 2: Transfer Learning GNN")
    log.info("=" * 70)

    try:
        from fase2_spring_tire_gnn import train_gnn_transfer_learning

        model_info = train_gnn_transfer_learning(
            dataset_path=dataset_path,
            pretrained_models_dir=pretrained_dir,
            output_dir=output_dir,
        )
        return model_info

    except Exception as e:
        log.error(f"  Error en Fase 2: {e}", exc_info=True)
        log.warning("  ⚠ Fase 2 falló pero el pipeline continuará (GNN no es requerido para fase 3)")
        return None


def run_phase3(pop_size:             int   = 100,
               n_gen:                int   = 200,
               n_obj:                int   = 8,
               output_dir:           str   = 'results/pareto',
               seed:                 int   = 42,
               use_fem3d:            bool  = True,
               fem3d_refresh_every:  int   = 15,
               fem3d_top_k:          int   = 5,
               fem3d_final_validate: int   = 20,
               fem3d_mesh_size:      float = None,
               checkpoint_every:     int   = 25) -> tuple:
    """
    Fase 3: NSGA-III + Surrogate Refresh FEM 3D.

    Returns:
        (results_dict, pareto_json_path)  — pareto_json_path es None si falla.
    """
    log.info("\n" + "=" * 70)
    log.info("  FASE 3: Optimización NSGA-III")
    log.info("=" * 70)

    try:
        from fase3_spring_tire_nsga import run_nsga3_optimization

        results = run_nsga3_optimization(
            pop_size=pop_size,
            n_generations=n_gen,
            n_obj=n_obj,
            output_dir=output_dir,
            seed=seed,
            use_fem3d=use_fem3d,
            fem3d_refresh_every=fem3d_refresh_every,
            fem3d_top_k=fem3d_top_k,
            fem3d_final_validate=fem3d_final_validate,
            fem3d_mesh_size=fem3d_mesh_size,
            checkpoint_every=checkpoint_every,
        )

        if not results:
            log.error("  Fase 3 devolvió resultados vacíos")
            return None, None

        # FIX: buscar el archivo más reciente en output_dir en vez de
        #      construir el timestamp aquí (que podría diferir del real)
        pattern = os.path.join(output_dir, 'pareto_spring_tire_*.json')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            log.error(f"  No se encontró ningún archivo pareto en {output_dir}")
            return results, None

        pareto_json = candidates[-1]   # el más reciente
        log.info(f"  Archivo Pareto: {pareto_json}")
        return results, pareto_json

    except Exception as e:
        log.error(f"  Error en Fase 3: {e}", exc_info=True)
        return None, None


def run_phase6(pareto_json: str,
               output_dir:  str = 'results/stl') -> dict:
    """
    Fase 6: Exportación STL con separación por material.
    Returns: dict con reporte de exportación.
    """
    log.info("\n" + "=" * 70)
    log.info("  FASE 6: Exportación STL")
    log.info("=" * 70)

    from fase6_spring_tire_export import export_pareto_solutions

    report = export_pareto_solutions(
        pareto_json=pareto_json,
        output_dir=output_dir,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pipeline NASA Spring Tire v6.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Fase a ejecutar
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', '1', '2', '3', '6'],
                        help='Fase a ejecutar')

    # Fase 1
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Número de muestras para Fase 1')

    # Fase 2
    parser.add_argument('--pretrained-dir', type=str, default=None,
                        help='Directorio con modelos pre-entrenados (Fase 2)')

    # Fase 3 — NSGA-III
    parser.add_argument('--pop-size', type=int, default=100,
                        help='Tamaño de población NSGA-III')
    parser.add_argument('--n-gen', type=int, default=200,
                        help='Número de generaciones NSGA-III')
    parser.add_argument('--n-obj', type=int, default=8,          # FIX: era hardcodeado
                        help='Número de objetivos NSGA-III (2-8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria')

    # Fase 3 — FEM 3D Surrogate Refresh                          # FIX: parámetros nuevos
    parser.add_argument('--no-fem3d', action='store_true',
                        help='Desactivar FEM 3D (modo legacy, más rápido)')
    parser.add_argument('--refresh-every', type=int, default=15,
                        help='Generaciones entre refreshes FEM 3D')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Soluciones Pareto evaluadas por refresh FEM 3D')
    parser.add_argument('--final-validate', type=int, default=20,
                        help='Soluciones validadas con FEM 3D al final')
    parser.add_argument('--mesh-size', type=float, default=None,
                        help='Tamaño de elemento FEM 3D en mm (None=auto)')
    parser.add_argument('--checkpoint-every', type=int, default=25,
                        help='Guardar checkpoint Pareto cada N generaciones (default: 25)')

    # Entradas para fases intermedias
    parser.add_argument('--dataset', type=str, default=None,
                        help='Ruta al dataset JSON (omite Fase 1)')
    parser.add_argument('--pareto', type=str, default=None,
                        help='Ruta al Pareto JSON (omite Fases 1-3)')

    args = parser.parse_args()

    _t_pipeline_start = time.time()

    log.info("=" * 70)
    log.info("  NASA SPRING TIRE PIPELINE v6.0")
    log.info("=" * 70)

    # ---- Verificar dependencias ----
    if not check_dependencies():
        log.error("\n  Instalar dependencias requeridas antes de continuar")
        return 1

    # ---- Crear directorios ----
    for d in ['results/data', 'results/models', 'results/pareto', 'results/stl']:
        os.makedirs(d, exist_ok=True)

    # ----------------------------------------------------------------
    # Fase 1
    # ----------------------------------------------------------------
    if args.phase in ['all', '1']:
        dataset_path = run_phase1(n_samples=args.n_samples)
    else:
        dataset_path = args.dataset or 'results/data/spring_tire_dataset.json'

    # ----------------------------------------------------------------
    # Fase 2
    # ----------------------------------------------------------------
    if args.phase in ['all', '2']:
        run_phase2(
            dataset_path=dataset_path,
            pretrained_dir=args.pretrained_dir,
        )

    # ----------------------------------------------------------------
    # Fase 3
    # ----------------------------------------------------------------
    pareto_json = args.pareto   # puede ser None si no se pasó --pareto

    if args.phase in ['all', '3']:
        _, pareto_json = run_phase3(
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            n_obj=args.n_obj,                          # FIX: se pasa n_obj
            seed=args.seed,
            use_fem3d=not args.no_fem3d,               # FIX: parámetro FEM 3D
            fem3d_refresh_every=args.refresh_every,
            fem3d_top_k=args.top_k,
            fem3d_final_validate=args.final_validate,
            fem3d_mesh_size=args.mesh_size,
            checkpoint_every=args.checkpoint_every,
        )

        if pareto_json is None:
            log.error("\n  Fase 3 falló — abortando pipeline")
            return 1

    # ----------------------------------------------------------------
    # Fase 6
    # ----------------------------------------------------------------
    if args.phase in ['all', '6']:
        # FIX: error claro si no hay pareto disponible
        if not pareto_json:
            log.error("\n  No hay archivo Pareto disponible para Fase 6.")
            log.error("  Opciones:")
            log.error("    - Ejecutar con --phase all  (corre fases 1→3→6)")
            log.error("    - Pasar --pareto <ruta_al_pareto.json>")
            return 1

        if not os.path.isfile(pareto_json):
            log.error(f"\n  Archivo Pareto no encontrado: {pareto_json}")
            return 1

        run_phase6(pareto_json=pareto_json)

    # ----------------------------------------------------------------
    # Resumen final
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("  PIPELINE COMPLETADO")
    log.info("=" * 70)

    if pareto_json:
        log.info(f"  Pareto:    {pareto_json}")
    log.info(f"  STL:       results/stl/")
    log.info(f"  Modelos:   results/models/")
    # FIX: dataset_path puede no estar definido si solo se corrió fase 6
    _ds = locals().get('dataset_path') or args.dataset or '(no generado en esta ejecución)'
    log.info(f"  Dataset:   {_ds}")
    log.info(f"  Tiempo total: {time.time() - _t_pipeline_start:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
