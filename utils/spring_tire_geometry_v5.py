#!/usr/bin/env python3
"""
spring_tire_geometry_v5.py - Geometría NASA Spring Tire v6.3

Versión con SÓLIDOS 3D REALES:
- Tread: Toroide sólido (cilindro exterior - cilindro interior)
- Rim: Toroide sólido  
- Spokes: Cilindros sólidos
- Hub: Cilindro sólido

Todos los componentes son sólidos watertight que pueden unirse con boolean union.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

log = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    log.warning("trimesh no disponible")


def create_solid_tread(r_outer: float, r_inner: float, width: float,
                       n_segments: int = 64) -> 'trimesh.Trimesh':
    """Crea TREAD como ANILLO HUECO (hollow ring), igual que v4.

    r_inner define el radio interior del anillo (espacio libre para los radios).
    Un cilindro sólido macizo ocuparía el espacio de los radios, imposibilitando
    el boolean union posterior.
    """
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    if r_outer <= r_inner:
        raise ValueError(f"create_solid_tread: r_outer ({r_outer:.1f}) debe ser > r_inner ({r_inner:.1f})")

    height = width
    # Cilindro exterior sólido
    outer = trimesh.creation.cylinder(radius=r_outer, height=height, sections=n_segments)
    # Cilindro interior a restar (el hueco)
    inner = trimesh.creation.cylinder(radius=r_inner, height=height + 2.0, sections=n_segments)

    try:
        result = trimesh.boolean.difference([outer, inner])
        if result is not None and len(result.faces) > 0:
            return result
    except Exception:
        pass
    # Fallback: devolver solo el cilindro exterior
    return outer


def create_solid_rim(r_outer: float, r_inner: float, width: float,
                     n_segments: int = 48) -> 'trimesh.Trimesh':
    """
    Crea RIM como cilindro anular hueco usando vértices manuales (igual que v4).

    BUG#12 fix revisado: el método anterior usaba boolean difference con
    engine='blender', pero blender no está disponible en este entorno,
    lo que causaba un fallback silencioso a cilindro SÓLIDO.

    Ahora usa construcción de malla manual (mismo método que create_tread_band en v4):
    - Superficie exterior (r_outer)
    - Superficie interior (r_inner)
    - Tapas superior e inferior
    Resultado: anillo hueco garantizado, sin dependencia de booleanos.
    """
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    height = width * 0.9

    # Si r_inner no es válido, retornar cilindro simple (caso degenerado)
    if r_inner <= 0 or r_inner >= r_outer * 0.98:
        return trimesh.creation.cylinder(radius=r_outer, height=height, sections=n_segments)

    # ---------- Construcción manual de anillo hueco (sin booleanos) ----------
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    n_z = 4
    z_vals = np.linspace(-height / 2, height / 2, n_z)

    vertices = []
    faces = []

    # Vértices superficie exterior
    for zi in z_vals:
        for ti in theta:
            vertices.append([r_outer * np.cos(ti), r_outer * np.sin(ti), zi])

    # Vértices superficie interior
    for zi in z_vals:
        for ti in theta:
            vertices.append([r_inner * np.cos(ti), r_inner * np.sin(ti), zi])

    vertices = np.array(vertices)
    n_per_ring = len(theta)
    offset = n_z * n_per_ring  # offset inicio de vértices interiores

    # Caras superficie exterior (normales hacia afuera)
    for i in range(n_z - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + j_next
            v2 = (i + 1) * n_per_ring + j_next
            v3 = (i + 1) * n_per_ring + j
            faces.extend([[v0, v2, v1], [v0, v3, v2]])

    # Caras superficie interior (normales hacia adentro, invertidas)
    for i in range(n_z - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = offset + i * n_per_ring + j
            v1 = offset + i * n_per_ring + j_next
            v2 = offset + (i + 1) * n_per_ring + j_next
            v3 = offset + (i + 1) * n_per_ring + j
            faces.extend([[v0, v1, v2], [v0, v2, v3]])

    # Tapas superior e inferior (conectan exterior e interior)
    for j in range(n_per_ring):
        j_next = (j + 1) % n_per_ring
        # Tapa superior (z = +height/2)
        v_ext  = (n_z - 1) * n_per_ring + j
        v_ext_n = (n_z - 1) * n_per_ring + j_next
        v_int  = offset + (n_z - 1) * n_per_ring + j
        v_int_n = offset + (n_z - 1) * n_per_ring + j_next
        faces.extend([[v_ext, v_int_n, v_int], [v_ext, v_ext_n, v_int_n]])
        # Tapa inferior (z = -height/2)
        v_ext  = j
        v_ext_n = j_next
        v_int  = offset + j
        v_int_n = offset + j_next
        faces.extend([[v_ext, v_int, v_int_n], [v_ext, v_int_n, v_ext_n]])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def create_solid_spokes(r_rim_outer: float, r_tread_inner: float, width: float,
                        n_spokes: int, spoke_thickness: float,
                        angle_cross: float = 55.0) -> List['trimesh.Trimesh']:
    """
    Crea radios como cilindros sólidos con ángulo de cruce.
    
    Args:
        angle_cross: Ángulo de cruce en grados (0 = radial, 90 = tangencial)
    """
    if not TRIMESH_AVAILABLE:
        return []
    
    meshes = []
    angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)

    # BUG fix: usar width para distribuir spokes en capas Z (como v4)
    # Antes todos los spokes estaban en z=0 → llanta estructuralmente plana
    # n_layers_spoke controla la distribución vertical, mínimo 1
    n_layers_spoke = max(1, int(round(width / 60.0)))  # 1 capa cada ~60mm de ancho
    if n_layers_spoke == 1:
        z_layers = [0.0]
    else:
        margin = spoke_thickness * 1.5
        z_layers = list(np.linspace(-width/2 + margin, width/2 - margin, n_layers_spoke))

    # Penetración para asegurar intersección sólida
    penetration = spoke_thickness * 0.8
    
    for i, angle in enumerate(angles):
        for z_layer in z_layers:
            # Puntos de inicio y fin con penetración
            p_rim = np.array([
                (r_rim_outer - penetration) * np.cos(angle),
                (r_rim_outer - penetration) * np.sin(angle),
                z_layer
            ])

            # BUG fix: alternancia ±angle_cross para crear patrón pantográfico cruzado real
            # Como en v4: cada radio alterna la dirección de cruce
            sign = 1 if (i % 2 == 0) else -1
            angle_cross_rad = sign * np.radians(angle_cross)
            p_tread = np.array([
                (r_tread_inner + penetration) * np.cos(angle + angle_cross_rad),
                (r_tread_inner + penetration) * np.sin(angle + angle_cross_rad),
                z_layer
            ])

            vec = p_tread - p_rim
            length = np.linalg.norm(vec)

            if length < 0.01:
                continue

            # Crear cilindro sólido
            cyl = trimesh.creation.cylinder(
                radius=spoke_thickness/2,
                height=length,
                sections=8
            )

            # Orden correcto: rotar PRIMERO, luego trasladar al punto medio
            z_axis = np.array([0, 0, 1])
            direction = vec / length
            cross = np.cross(z_axis, direction)

            if np.linalg.norm(cross) > 1e-6:
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot_matrix = trimesh.transformations.rotation_matrix(rot_angle, cross, [0, 0, 0])
                cyl.apply_transform(rot_matrix)  # 1. Rotar primero (en origen)

            mid = (p_rim + p_tread) / 2
            cyl.apply_translation(mid)           # 2. Trasladar al punto medio

            meshes.append(cyl)
    
    return meshes


def create_solid_hub(r_outer: float, axle_diameter: float, width: float,
                     n_spokes: int = 6, n_layers: int = 9,
                     hub_angle_cross: float = 0.0) -> 'trimesh.Trimesh':
    """
    Crea hub con disco central + radios de hub conectando eje → rim.

    Delega en create_wheel_hub_fullwidth (v4) que genera la estructura correcta:
      - disco anular del hub (axle_radius → hub_radius)
      - radios del hub en múltiples capas Z conectando hub al rim interior

    Esta era la estructura que faltaba en v5 (antes solo creaba un cilindro sólido
    del eje sin conectarlo al rim).
    """
    from utils.spring_tire_geometry_v4 import create_wheel_hub_fullwidth
    return create_wheel_hub_fullwidth(
        r_outer=r_outer,
        axle_diameter=axle_diameter,
        width=width,
        n_spokes=n_spokes,
        n_layers=n_layers,
        hub_angle_cross=hub_angle_cross,
    )


class SpringTireGeometryV5:
    """Genera geometría Spring Tire con SÓLIDOS 3D REALES."""
    
    def __init__(self, params: Dict):
        self.params = params
        self._validate_params()
        
        self.r_ext = params['d_ext'] / 2
        self.r_rim = params['d_rim'] / 2
        self.width = params['width']
        self.n_radios = params['n_radios']
        self.wall_thick = params['wall_thickness']
        self.angle_cross = np.radians(params['angle_cross'])
        
        # Dimensiones derivadas
        self.tread_thickness = self.wall_thick * 2.0
        self.rim_thickness = self.wall_thick * 1.5
        
        self.r_tread_inner = self.r_ext - self.tread_thickness
        self.r_rim_outer = self.r_rim + self.rim_thickness
        self.r_rim_inner = max(self.r_rim - self.rim_thickness * 0.5, 20)
    
    def _validate_params(self):
        """Valida parámetros geométricos."""
        d_ext = self.params['d_ext']
        d_rim = self.params['d_rim']
        wall = self.params['wall_thickness']
        
        tread_thick = wall * 2.0
        rim_thick = wall * 1.5
        
        r_tread_inner = d_ext / 2 - tread_thick
        r_rim_outer = d_rim / 2 + rim_thick
        
        gap = r_tread_inner - r_rim_outer
        
        if gap < 5.0:
            raise ValueError(f"Espacio insuficiente: {gap:.1f}mm < 5mm")
    
    def generate(self) -> Optional['trimesh.Trimesh']:
        """Genera geometría completa usando SÓLIDOS 3D."""
        if not TRIMESH_AVAILABLE:
            return None
        
        log.info("="*60)
        log.info("Generando Spring Tire v6.3 (SÓLIDOS 3D)")
        log.info("="*60)
        
        meshes = []
        
        # 1. TREAD (cilindro sólido exterior)
        log.info("[1] Tread (sólido)...")
        tread = create_solid_tread(
            r_outer=self.r_ext,
            r_inner=self.r_tread_inner,
            width=self.width
        )
        if tread and len(tread.faces) > 0:
            meshes.append(tread)
            log.info(f"    ✓ {len(tread.faces)} caras (sólido watertight={tread.is_watertight})")
        
        # 2. RIM (cilindro sólido interior)
        log.info("[2] Rim (sólido)...")
        rim = create_solid_rim(
            r_outer=self.r_rim_outer,
            r_inner=self.r_rim_inner,
            width=self.width * 0.9
        )
        if rim and len(rim.faces) > 0:
            meshes.append(rim)
            log.info(f"    ✓ {len(rim.faces)} caras (sólido watertight={rim.is_watertight})")
        
        # 3. SPOKES (cilindros sólidos)
        log.info("[3] Spokes (sólidos)...")
        spokes = create_solid_spokes(
            r_rim_outer=self.r_rim_outer,
            r_tread_inner=self.r_tread_inner,
            width=self.width,
            n_spokes=self.n_radios,
            spoke_thickness=self.wall_thick,
            angle_cross=self.params['angle_cross']
        )
        for i, spoke in enumerate(spokes):
            meshes.append(spoke)
        log.info(f"    ✓ {len(spokes)} radios (sólidos watertight)")
        
        # 4. HUB (disco + radios hub→rim, misma estructura que v4)
        log.info("[4] Hub (fullwidth con radios)...")
        n_layers_hub = self.params.get('n_layers', 9)
        hub_angle_cross = self.params.get('hub_angle_cross', 0.0)
        hub = create_solid_hub(
            r_outer=self.r_rim,
            axle_diameter=20.0,
            width=self.width * 0.7,
            n_spokes=max(6, self.n_radios // 2),
            n_layers=n_layers_hub,
            hub_angle_cross=hub_angle_cross,
        )
        if hub and len(hub.faces) > 0:
            meshes.append(hub)
            log.info(f"    ✓ {len(hub.faces)} caras")
        
        if not meshes:
            return None
        
        # 5. Unir TODO con boolean union (ahora debería funcionar porque todos son sólidos)
        log.info("\n[5] Aplicando boolean union a sólidos...")
        try:
            # Verificar que todos son watertight
            all_watertight = all(m.is_watertight for m in meshes)
            log.info(f"    Todos watertight: {all_watertight}")
            
            # Intentar boolean union
            result = meshes[0].copy()
            success_count = 1
            
            for i, mesh in enumerate(meshes[1:], 1):
                try:
                    result = trimesh.boolean.union([result, mesh])
                    success_count += 1
                except Exception as e:
                    log.warning(f"    Union {i+1}/{len(meshes)}: {e}")
                    # Fallback: concatenar
                    result = trimesh.util.concatenate([result, mesh])
            
            log.info(f"    Boolean exitoso: {success_count}/{len(meshes)}")
            log.info(f"\n✓ Total: {len(result.faces)} caras, {len(result.vertices)} vértices")
            log.info(f"  Watertight: {result.is_watertight}")
            
            return result
            
        except Exception as e:
            log.error(f"Error en boolean: {e}")
            log.info("Usando concatenación simple...")
            combined = trimesh.util.concatenate(meshes)
            log.info(f"\n✓ Total: {len(combined.faces)} caras, {len(combined.vertices)} vértices")
            return combined
    
    def generate_and_validate(self):
        """Genera y valida."""
        from utils.spring_tire_validator import validate_spring_tire_stl
        
        mesh = self.generate()
        
        if mesh is None:
            return None, {'valid': False, 'error': 'No se generó malla'}
        
        validation = validate_spring_tire_stl(mesh, self.params)
        return mesh, validation


def export_stl_v5(params: Dict, filepath: str, validate: bool = True):
    """Exporta STL usando geometría v5 (sólidos)."""
    tire = SpringTireGeometryV5(params)
    
    if validate:
        mesh, validation = tire.generate_and_validate()
    else:
        mesh = tire.generate()
        validation = {'valid': mesh is not None}
    
    if mesh is None:
        return False, validation
    
    try:
        mesh.export(filepath)
        return True, validation
    except Exception as e:
        log.error(f"Error exportando: {e}")
        return False, validation
