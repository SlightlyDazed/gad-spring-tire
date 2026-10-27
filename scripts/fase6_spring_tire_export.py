#!/usr/bin/env python3
"""
fase6_spring_tire_export.py - Fase 6: Exportación STL Spring Tire v6.0

Para cada solución Pareto genera:

  results/stl/
  ├── spring_tire_000.stl              ← neumático completo (impresión monolítica)
  ├── spring_tire_000_parts/
  │   ├── TPU_95A.stl                  ← todas las piezas de ese material juntas
  │   ├── TPU_85A.stl
  │   └── Nylon_PA12.stl
  ├── spring_tire_000_metadata.json
  ├── spring_tire_001.stl
  ├── spring_tire_001_parts/
  │   └── ...
  └── export_report.json               ← resumen de toda la exportación

Lógica de separación por material:
  - La Spring Tire tiene 8 sectores angulares de 45° cada uno
  - material_sectors[i] define el material del sector i
  - Cada cara del STL se asigna al sector según el ángulo del centroide
  - Las caras de un mismo material se combinan en un único STL de esa parte
  - Resultado: un STL por material, listo para cargar en el slicer

Dependencias: trimesh, numpy
"""

import os
import sys
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.spring_tire_genome import decode_genome, MATERIAL_CATALOG
from utils.spring_tire_geometry_v4 import SpringTireGeometry, export_stl

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    log.warning("trimesh no disponible — validación y separación de partes desactivadas")


# ---------------------------------------------------------------------------
# Catálogo de materiales (para nombres de archivo)
# ---------------------------------------------------------------------------
MATERIAL_NAMES = {
    0: 'TPU_95A',
    1: 'TPU_85A',
    2: 'TPU_75D',
    3: 'Nylon_PA12',
    4: 'ABS',
    5: 'PLA',
}


# ---------------------------------------------------------------------------
# Separación de STL en partes por material
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Separación de STL en partes por componente estructural (RADIAL, no angular)
# ---------------------------------------------------------------------------
# La geometría de la Spring Tire tiene 4 zonas radiales:
#
#    r < r_hub_outer          → HUB (disco central + radios al eje)
#    r_hub_outer ≤ r < r_rim_outer → RIM (aro interior)
#    r_rim_outer ≤ r < r_tread_inner → SPOKES (radios exteriores cruzados)
#    r ≥ r_tread_inner        → TREAD (banda exterior de rodadura)
#
# Cada componente recibe el material que el genoma asignó:
#   sectors[6-7] promedio → HUB
#   sectors[4-5] promedio → RIM
#   sectors[2-3] promedio → SPOKES
#   sectors[0-1] promedio → TREAD
#
# Si todos los sectores usan el mismo material, se devuelve solo un STL.
# ---------------------------------------------------------------------------

COMPONENT_SECTOR_MAP = {
    # component_name: índice ÚNICO del sector del genoma que determina su material
    # Usamos sectores pares (0,2,4,6) para cubrir los 4 cuadrantes del genoma
    'tread':  0,   # banda exterior → sector 0
    'spokes': 2,   # radios exteriores → sector 2
    'rim':    4,   # aro interior → sector 4
    'hub':    6,   # disco central → sector 6
}


def _majority_material(material_sectors: List[int], sector_index) -> int:
    """Devuelve el material del sector dado (índice único, sin votación)."""
    if isinstance(sector_index, list):
        # Compatibilidad: si aún se pasa lista, usar el primero
        sector_index = sector_index[0]
    if sector_index < len(material_sectors):
        return material_sectors[sector_index]
    return material_sectors[0] if material_sectors else 0


def _compute_radial_bounds(params: Dict) -> Dict[str, float]:
    """
    Calcula los radios de frontera entre componentes estructurales.

    Devuelve un dict con las claves:
      r_hub_outer, r_rim_inner, r_rim_outer, r_tread_inner, r_tread_outer
    """
    d_ext          = params.get('d_ext', 300.0)
    d_rim          = params.get('d_rim', 150.0)
    wall_thickness = params.get('wall_thickness', 5.0)
    tread_rim_fac  = params.get('tread_rim_factor', 2.0)

    r_ext         = d_ext / 2.0
    r_rim         = d_rim / 2.0
    tread_thick   = wall_thickness * tread_rim_fac
    rim_thick     = wall_thickness * 1.5

    r_tread_inner = r_ext - tread_thick
    r_rim_outer   = r_rim + rim_thick
    r_rim_inner   = max(r_rim - rim_thick * 0.5, 20.0)
    r_hub_outer   = 30.0   # radio del hub (siempre ~30mm)

    return {
        'r_tread_outer': r_ext,
        'r_tread_inner': r_tread_inner,
        'r_rim_outer':   r_rim_outer,
        'r_rim_inner':   r_rim_inner,
        'r_hub_outer':   r_hub_outer,
    }


def split_mesh_by_material(
        mesh: 'trimesh.Trimesh',
        material_sectors: List[int],
        params: Dict = None,
) -> Dict[int, 'trimesh.Trimesh']:
    """
    Divide el mesh en submeshes por COMPONENTE ESTRUCTURAL (radial).

    Clasifica cada cara según la distancia radial de su centroide al eje Z:
      - r >= r_tread_inner          → TREAD
      - r_rim_outer <= r < r_tread_inner → SPOKES
      - r_rim_inner <= r < r_rim_outer   → RIM
      - r < r_rim_inner             → HUB

    A cada componente se le asigna el material del genoma que le corresponde.

    Args:
        mesh: malla completa de la Spring Tire
        material_sectors: lista de 8 ints (material por sector del genoma)
        params: parámetros geométricos decodificados (para calcular radios)

    Returns:
        dict {material_id: submesh} agrupado por material final
    """
    if params is None:
        params = {}

    bounds = _compute_radial_bounds(params)
    centroids = mesh.triangles_center          # (n_faces, 3)
    r         = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)  # radio en plano XY

    # Asignar cada cara a un componente (0=hub, 1=rim, 2=spokes, 3=tread)
    component_labels = np.zeros(len(r), dtype=int)  # default = hub
    component_labels[r >= bounds['r_rim_inner']]   = 1  # rim
    component_labels[r >= bounds['r_rim_outer']]   = 2  # spokes
    component_labels[r >= bounds['r_tread_inner']] = 3  # tread

    # Mapear componente → material usando el genoma
    component_to_mat = {
        3: _majority_material(material_sectors, COMPONENT_SECTOR_MAP['tread']),   # tread
        2: _majority_material(material_sectors, COMPONENT_SECTOR_MAP['spokes']),  # spokes
        1: _majority_material(material_sectors, COMPONENT_SECTOR_MAP['rim']),     # rim
        0: _majority_material(material_sectors, COMPONENT_SECTOR_MAP['hub']),     # hub
    }

    # Convertir labels de componente a labels de material
    face_material = np.vectorize(component_to_mat.get)(component_labels)

    log.info(f"    Segmentación radial:")
    log.info(f"      TREAD  (r≥{bounds['r_tread_inner']:.1f}mm) → mat {component_to_mat[3]} ({MATERIAL_NAMES.get(component_to_mat[3], '?')})")
    log.info(f"      SPOKES ({bounds['r_rim_outer']:.1f}≤r<{bounds['r_tread_inner']:.1f}mm) → mat {component_to_mat[2]} ({MATERIAL_NAMES.get(component_to_mat[2], '?')})")
    log.info(f"      RIM    ({bounds['r_rim_inner']:.1f}≤r<{bounds['r_rim_outer']:.1f}mm) → mat {component_to_mat[1]} ({MATERIAL_NAMES.get(component_to_mat[1], '?')})")
    log.info(f"      HUB    (r<{bounds['r_rim_inner']:.1f}mm) → mat {component_to_mat[0]} ({MATERIAL_NAMES.get(component_to_mat[0], '?')})")

    unique_mats = np.unique(face_material)
    result = {}

    for mat_id in unique_mats:
        face_mask = face_material == mat_id
        face_idx  = np.where(face_mask)[0]

        if len(face_idx) == 0:
            continue

        submesh = mesh.submesh([face_idx], append=True)

        if submesh is None:
            continue
        if isinstance(submesh, list):
            if len(submesh) == 0:
                continue
            submesh = trimesh.util.concatenate(submesh)

        result[int(mat_id)] = submesh

    return result


def validate_mesh(mesh: 'trimesh.Trimesh') -> Dict:
    """Calcula estadísticas de validación de una malla."""
    return {
        'n_vertices':   int(len(mesh.vertices)),
        'n_faces':      int(len(mesh.faces)),
        'is_watertight': bool(mesh.is_watertight),
        'volume_cm3':   float(mesh.volume * 1e-3) if mesh.is_watertight else 0.0,
        'bounds_mm':    mesh.bounds.tolist(),
        'euler_number': int(mesh.euler_number),
    }


# ---------------------------------------------------------------------------
# Exportar un único diseño completo + partes por material
# ---------------------------------------------------------------------------

def export_design_with_parts(
        sol_id:     int,
        genome:     np.ndarray,
        params:     Dict,
        output_dir: str,
        objectives: Dict = None,
        fem3d_data: Dict = None,
) -> Dict:
    """
    Exporta un diseño Spring Tire con toda su estructura de archivos:

      output_dir/
      ├── spring_tire_NNN.stl              ← completo
      ├── spring_tire_NNN_parts/
      │   ├── TPU_95A.stl
      │   ├── TPU_85A.stl
      │   └── ...
      └── spring_tire_NNN_metadata.json

    Args:
        sol_id:     ID de la solución
        genome:     genoma de 14 genes
        params:     parámetros decodificados
        output_dir: carpeta raíz de salida
        objectives: objetivos del NSGA-III (opcional, para metadatos)
        fem3d_data: validación FEM 3D (opcional, para metadatos)

    Returns:
        Dict con rutas de archivos y estadísticas de validación
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name  = f'spring_tire_{sol_id:03d}'
    stl_full   = os.path.join(output_dir, f'{base_name}.stl')
    parts_dir  = os.path.join(output_dir, f'{base_name}_parts')
    meta_path  = os.path.join(output_dir, f'{base_name}_metadata.json')

    result = {
        'id':           sol_id,
        'stl_full':     stl_full,
        'parts_dir':    parts_dir,
        'parts':        {},       # {material_name: stl_path}
        'validation':   {},
        'objectives':   objectives or {},
    }

    # ----------------------------------------------------------------
    # 1. Exportar STL completo
    # ----------------------------------------------------------------
    success, validation_info = export_stl(params, stl_full)

    if not success:
        log.error(f"  ✗ #{sol_id}: export_stl falló")
        result['error'] = 'export_stl falló'
        return result

    log.info(f"  #{sol_id:03d} STL completo exportado → {stl_full}")

    # ----------------------------------------------------------------
    # 2. Validar malla completa con trimesh
    # ----------------------------------------------------------------
    if not TRIMESH_AVAILABLE:
        result['validation']['full'] = validation_info
        return result

    try:
        mesh_full = trimesh.load_mesh(stl_full)
    except Exception as ex:
        log.warning(f"  #{sol_id}: no se pudo cargar STL con trimesh: {ex}")
        result['validation']['full'] = {'error': str(ex)}
        return result

    full_stats = validate_mesh(mesh_full)
    full_stats['geometry_validation'] = validation_info
    result['validation']['full'] = full_stats

    log.info(f"         Caras={full_stats['n_faces']:,}  "
             f"Watertight={full_stats['is_watertight']}  "
             f"Vol={full_stats['volume_cm3']:.1f}cm³")

    # ----------------------------------------------------------------
    # 3. Separar por material
    # ----------------------------------------------------------------
    material_sectors = list(params.get('material_sectors', [0]*8))
    unique_mat_ids   = list(dict.fromkeys(material_sectors))  # orden de aparición

    # Si solo hay un material, no tiene sentido crear subcarpeta
    if len(unique_mat_ids) == 1:
        mat_name = MATERIAL_NAMES.get(unique_mat_ids[0], f'material_{unique_mat_ids[0]}')
        log.info(f"         Material único: {mat_name} (no se crean subcarpetas)")
        result['parts'][mat_name] = stl_full   # la parte = el STL completo
        result['validation']['parts'] = {mat_name: full_stats}
        _write_metadata(meta_path, sol_id, genome, params, result, objectives, fem3d_data)
        return result

    # Múltiples materiales → crear carpeta de partes
    os.makedirs(parts_dir, exist_ok=True)

    try:
        parts_by_mat = split_mesh_by_material(mesh_full, material_sectors, params=params)
    except Exception as ex:
        log.warning(f"  #{sol_id}: split_mesh_by_material falló: {ex}")
        result['validation']['split_error'] = str(ex)
        _write_metadata(meta_path, sol_id, genome, params, result, objectives, fem3d_data)
        return result

    result['validation']['parts'] = {}

    log.info(f"         Partes por material ({len(parts_by_mat)} materiales):")

    for mat_id, submesh in parts_by_mat.items():
        mat_name  = MATERIAL_NAMES.get(mat_id, f'material_{mat_id}')
        part_path = os.path.join(parts_dir, f'{mat_name}.stl')

        try:
            submesh.export(part_path)
        except Exception as ex:
            log.warning(f"         ✗ {mat_name}: export falló: {ex}")
            continue

        part_stats = validate_mesh(submesh)
        result['parts'][mat_name]                       = part_path
        result['validation']['parts'][mat_name]         = part_stats

        # Sectores que usan este material
        sectors_with_mat = [i for i, m in enumerate(material_sectors) if m == mat_id]
        sector_angles    = [f"{i*45}°-{(i+1)*45}°" for i in sectors_with_mat]

        log.info(f"           {mat_name:15s}: "
                 f"{part_stats['n_faces']:5,} caras  "
                 f"vol={part_stats['volume_cm3']:.2f}cm³  "
                 f"sectores={sector_angles}")

    # ----------------------------------------------------------------
    # 4. Guardar metadatos JSON
    # ----------------------------------------------------------------
    _write_metadata(meta_path, sol_id, genome, params, result, objectives, fem3d_data)

    return result


def _write_metadata(
        meta_path: str,
        sol_id:    int,
        genome:    np.ndarray,
        params:    Dict,
        result:    Dict,
        objectives: Optional[Dict],
        fem3d_data: Optional[Dict],
):
    """Escribe el JSON de metadatos para un diseño."""
    material_sectors = list(params.get('material_sectors', [0]*8))

    # Mapa de materiales por sector (para el slicer)
    sector_map = []
    for i, mat_id in enumerate(material_sectors):
        mat_name = MATERIAL_NAMES.get(mat_id, f'material_{mat_id}')
        sector_map.append({
            'sector':       i,
            'angle_range':  f"{i*45}°–{(i+1)*45}°",
            'material_id':  int(mat_id),
            'material_name': mat_name,
        })

    # Guía de impresión por material
    print_guide = {}
    unique_mats = dict.fromkeys(material_sectors)
    for mat_id in unique_mats:
        mat_name = MATERIAL_NAMES.get(mat_id, f'material_{mat_id}')
        stl_path = result['parts'].get(mat_name, '')
        n_sectors_mat = sum(1 for m in material_sectors if m == mat_id)

        print_guide[mat_name] = {
            'stl_file':        os.path.basename(stl_path) if stl_path else 'N/A',
            'n_sectors':       n_sectors_mat,
            'pct_of_tire':     round(n_sectors_mat / len(material_sectors) * 100, 1),
            'print_temp_C':    _get_print_temp(mat_id),
            'bed_temp_C':      _get_bed_temp(mat_id),
            'print_order':     _get_print_order(mat_id),
            'notes':           _get_print_notes(mat_id),
        }

    metadata = {
        'version':          '6.0',
        'id':               sol_id,
        'genome':           genome.tolist() if hasattr(genome, 'tolist') else list(genome),
        'params': {
            'd_ext_mm':        params.get('d_ext'),
            'd_rim_mm':        params.get('d_rim'),
            'width_mm':        params.get('width'),
            'n_radios':        params.get('n_radios'),
            'wall_thickness_mm': params.get('wall_thickness'),
            'angle_cross_deg': params.get('angle_cross'),
        },
        'sector_map':       sector_map,
        'print_guide':      print_guide,
        'objectives':       objectives or {},
        'fem3d_validation': fem3d_data,
        'files': {
            'full_stl':  os.path.basename(result['stl_full']),
            'parts_dir': os.path.basename(result['parts_dir']) if result['parts'] else None,
            'parts':     {k: os.path.basename(v) for k, v in result['parts'].items()},
        },
        'validation':       result.get('validation', {}),
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_safe)

    result['metadata_path'] = meta_path


def _get_print_temp(mat_id: int) -> int:
    temps = {0: 230, 1: 220, 2: 240, 3: 260, 4: 250, 5: 200}
    return temps.get(mat_id, 220)

def _get_bed_temp(mat_id: int) -> int:
    return 60 if mat_id in (0, 1, 2) else 80

def _get_print_order(mat_id: int) -> int:
    """Orden recomendado de impresión (menor número = primero)."""
    # Materiales más rígidos primero (mejor adhesión de capa)
    order = {3: 1, 4: 2, 5: 3, 2: 4, 0: 5, 1: 6}
    return order.get(mat_id, 5)

def _get_print_notes(mat_id: int) -> str:
    notes = {
        0: 'TPU flexible: imprimir lento (20-30mm/s), ventilador al 50%',
        1: 'TPU muy flexible: imprimir muy lento (15-20mm/s), sin retracción',
        2: 'TPU semirígido: imprimir a 30-40mm/s, ventilador al 100%',
        3: 'Nylon PA12: cama cerrada recomendada, secar filamento 8h a 70°C',
        4: 'ABS: cama cerrada obligatoria, evitar corrientes de aire',
        5: 'PLA: fácil de imprimir, enfriar bien entre capas',
    }
    return notes.get(mat_id, '')

def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Exportar todas las soluciones Pareto
# ---------------------------------------------------------------------------

def export_pareto_solutions(
        pareto_json: str,
        output_dir:  str = 'results/stl',
) -> Dict:
    """
    Exporta todas las soluciones Pareto a STL con separación por material.

    Para cada solución genera:
      - spring_tire_NNN.stl              ← neumático completo
      - spring_tire_NNN_parts/
          ├── MATERIAL_A.stl
          └── MATERIAL_B.stl
      - spring_tire_NNN_metadata.json    ← guía de impresión

    Args:
        pareto_json: ruta al JSON de soluciones Pareto (de fase3)
        output_dir:  carpeta raíz de salida

    Returns:
        Dict con reporte completo de exportación
    """
    log.info("=" * 70)
    log.info("  FASE 6: Exportación STL Spring Tire v6.0")
    log.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    with open(pareto_json, 'r') as f:
        pareto_data = json.load(f)

    solutions = pareto_data['soluciones']
    log.info(f"Soluciones Pareto: {len(solutions)}")
    log.info(f"Directorio de salida: {output_dir}\n")

    exported   = []
    n_ok       = 0
    n_fail     = 0

    pbar = tqdm(solutions, desc="Exportando STL", unit="diseño")
    for sol in pbar:
        sol_id    = sol['id']
        genome    = np.array(sol['genome'])
        params    = decode_genome(genome)
        objectives = sol.get('objectives', {})
        fem3d     = sol.get('fem3d_validation')   # presente si se usó fase3 v6.0

        pbar.set_postfix({'id': sol_id, 'ok': n_ok, 'fail': n_fail})

        result = export_design_with_parts(
            sol_id=sol_id,
            genome=genome,
            params=params,
            output_dir=output_dir,
            objectives=objectives,
            fem3d_data=fem3d,
        )

        if 'error' in result:
            n_fail += 1
            log.error(f"  ✗ #{sol_id}: {result['error']}")
        else:
            n_ok += 1
            exported.append(result)

    # ----------------------------------------------------------------
    # Resumen de materiales usados en todo el Pareto
    # ----------------------------------------------------------------
    all_mats_used = set()
    for sol in solutions:
        sectors = sol.get('params', {}).get('material_sectors', [0]*8)
        all_mats_used.update(int(m) for m in sectors)

    materials_summary = {
        MATERIAL_NAMES.get(m, f'material_{m}'): {
            'print_temp_C': _get_print_temp(m),
            'bed_temp_C':   _get_bed_temp(m),
            'print_order':  _get_print_order(m),
            'notes':        _get_print_notes(m),
        }
        for m in sorted(all_mats_used)
    }

    # ----------------------------------------------------------------
    # Generar configuración para slicer multi-material
    # ----------------------------------------------------------------
    slicer_config = _generate_slicer_config(solutions, all_mats_used)
    slicer_path   = os.path.join(output_dir, 'slicer_config.json')
    with open(slicer_path, 'w') as f:
        json.dump(slicer_config, f, indent=2)

    # ----------------------------------------------------------------
    # Reporte final
    # ----------------------------------------------------------------
    report = {
        'version':            '6.0',
        'pareto_file':        pareto_json,
        'output_dir':         output_dir,
        'n_total':            len(solutions),
        'n_exported_ok':      n_ok,
        'n_failed':           n_fail,
        'materials_used':     materials_summary,
        'slicer_config_path': slicer_path,
        'solutions':          exported,
    }

    report_path = os.path.join(output_dir, 'export_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=_json_safe)

    # ----------------------------------------------------------------
    # Log resumen
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("  RESUMEN DE EXPORTACIÓN")
    log.info("=" * 70)
    log.info(f"  Exportados OK:   {n_ok}/{len(solutions)}")
    log.info(f"  Fallidos:        {n_fail}")
    log.info(f"\n  Materiales en el Pareto:")
    for mat_name, info in materials_summary.items():
        _note = info['notes']
        _note_short = (_note[:50] + '...') if len(_note) > 50 else _note
        log.info(f"    {mat_name:15s}: {info['print_temp_C']}°C  —  {_note_short}")
    log.info(f"\n  Estructura de salida:")
    log.info(f"    {output_dir}/")
    log.info(f"    ├── spring_tire_NNN.stl           (neumático completo)")
    log.info(f"    ├── spring_tire_NNN_parts/")
    for mat_name in materials_summary:
        log.info(f"    │   ├── {mat_name}.stl")
    log.info(f"    ├── spring_tire_NNN_metadata.json  (guía de impresión)")
    log.info(f"    ├── slicer_config.json             (config multi-material)")
    log.info(f"    └── export_report.json             (reporte completo)")
    log.info("=" * 70)

    return report


# ---------------------------------------------------------------------------
# Configuración para slicer multi-material
# ---------------------------------------------------------------------------

def _generate_slicer_config(solutions: List[Dict], mat_ids: set) -> Dict:
    """
    Genera configuración para slicers como Bambu Studio o PrusaSlicer
    con soporte para multi-material (AMS / MMU).
    """
    materials_config = {}

    for mat_id in sorted(mat_ids):
        mat_name = MATERIAL_NAMES.get(mat_id, f'material_{mat_id}')
        materials_config[mat_name] = {
            'extruder':       _get_print_order(mat_id),   # número de extrusor sugerido
            'print_temp_C':   _get_print_temp(mat_id),
            'bed_temp_C':     _get_bed_temp(mat_id),
            'first_layer_temp_C': _get_print_temp(mat_id) + 5,
            'retraction_mm':  0.5 if mat_id in (0, 1) else 1.5,  # TPU sin retracción
            'retraction_speed_mm_s': 20 if mat_id in (0, 1) else 45,
            'print_speed_mm_s': 20 if mat_id in (0, 1) else 50,
            'fan_speed_pct':  50 if mat_id in (0, 1) else 100,
            'notes':          _get_print_notes(mat_id),
        }

    # Volumen de purga recomendado entre materiales (en mm³)
    # Mayor diferencia de temperatura → más purga
    purge_volumes = {}
    mat_list = sorted(mat_ids)
    for i, a in enumerate(mat_list):
        for b in mat_list[i+1:]:
            key = f"{MATERIAL_NAMES.get(a,'?')} → {MATERIAL_NAMES.get(b,'?')}"
            temp_diff = abs(_get_print_temp(a) - _get_print_temp(b))
            purge_volumes[key] = int(50 + temp_diff * 2)   # mm³

    return {
        'version':          '6.0',
        'slicer_target':    'Bambu Studio / PrusaSlicer (MMU/AMS)',
        'n_materials':      len(mat_ids),
        'materials':        materials_config,
        'purge_volumes_mm3': purge_volumes,
        'general_notes': [
            'Imprimir materiales en orden ascendente de print_order',
            'Para TPU (85A/95A): desactivar retracción o usar retracción mínima',
            'Secar todos los filamentos de nylon/PA12 antes de imprimir',
            'Usar cama flexible o lámina PEI para mejor adhesión de TPU',
        ],
    }


# ---------------------------------------------------------------------------
# Funciones de compatibilidad con fase6 v5.0
# ---------------------------------------------------------------------------

def export_single_design(genome: np.ndarray,
                          output_path: str,
                          metadata: Dict = None) -> bool:
    """
    Exporta un único diseño a STL.
    Alias de compatibilidad con la versión anterior.
    """
    params  = decode_genome(genome)
    success, _ = export_stl(params, output_path)

    if success and metadata:
        meta_path = output_path.replace('.stl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({'params': params, 'metadata': metadata}, f, indent=2)

    return success


def generate_slicer_config(pareto_json: str, output_path: str = None) -> Dict:
    """Genera configuración de slicer. Alias de compatibilidad."""
    with open(pareto_json, 'r') as f:
        pareto_data = json.load(f)

    all_mats = set()
    for sol in pareto_data['soluciones']:
        sectors = sol.get('params', {}).get('material_sectors', [0]*8)
        all_mats.update(int(m) for m in sectors)

    config = _generate_slicer_config(pareto_data['soluciones'], all_mats)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        log.info(f"Configuración slicer: {output_path}")

    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Exportar STL Spring Tire con separación por material'
    )
    parser.add_argument('--pareto',     type=str, required=True,
                        help='Archivo JSON con soluciones Pareto (de fase3)')
    parser.add_argument('--output-dir', type=str, default='results/stl',
                        help='Directorio de salida (default: results/stl)')
    args = parser.parse_args()

    report = export_pareto_solutions(
        pareto_json=args.pareto,
        output_dir=args.output_dir,
    )

    print(f"\nExportados: {report['n_exported_ok']}/{report['n_total']}")
    print(f"Reporte: {args.output_dir}/export_report.json")


if __name__ == '__main__':
    main()
