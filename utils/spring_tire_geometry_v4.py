"""
spring_tire_geometry_v4.py - Geometría NASA Spring Tire v5.4 (FULL-WIDTH)

Basado en el diseño original que genera test_spring_tire_FULL_WIDTH.stl:
- Tread y Rim: Anillos huecos (hollow bands) con vértices calculados manualmente
- Spokes: 9 capas de radios cruzados alternando ángulo
- Hub: 9 capas de radios del eje
- Fusión: trimesh.util.concatenate() (sin boolean union)
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


def create_tread_band(r_outer: float, r_inner: float, width: float,
                      n_segments: int = 64) -> 'trimesh.Trimesh':
    """
    Crea la banda de rodadura como anillo hueco (hollow band).
    
    Calcula vértices manualmente para formar un cilindro hueco.
    """
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    # BUG#15: validar r_outer > r_inner para evitar malla invertida o degenereada
    if r_outer <= r_inner:
        raise ValueError(f"create_tread_band: r_outer ({r_outer:.1f}mm) debe ser > r_inner ({r_inner:.1f}mm)")

    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    n_z = 5
    z = np.linspace(-width/2, width/2, n_z)

    vertices = []
    faces = []

    # Vértices superficie exterior
    for zi in z:
        for ti in theta:
            vertices.append([r_outer * np.cos(ti), r_outer * np.sin(ti), zi])

    # Vértices superficie interior
    for zi in z:
        for ti in theta:
            vertices.append([r_inner * np.cos(ti), r_inner * np.sin(ti), zi])

    vertices = np.array(vertices)
    n_per_ring = len(theta)

    # Faces superficie exterior
    for i in range(n_z - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + j_next
            v2 = (i + 1) * n_per_ring + j_next
            v3 = (i + 1) * n_per_ring + j
            faces.extend([[v0, v2, v1], [v0, v3, v2]])

    # Faces superficie interior
    offset = n_z * n_per_ring
    for i in range(n_z - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            v0 = offset + i * n_per_ring + j
            v1 = offset + i * n_per_ring + j_next
            v2 = offset + (i + 1) * n_per_ring + j_next
            v3 = offset + (i + 1) * n_per_ring + j
            faces.extend([[v0, v1, v2], [v0, v2, v3]])

    # Tapas (top y bottom)
    for j in range(n_per_ring):
        j_next = (j + 1) % n_per_ring
        # Tapa superior
        v_ext = (n_z - 1) * n_per_ring + j
        v_ext_n = (n_z - 1) * n_per_ring + j_next
        v_int = offset + (n_z - 1) * n_per_ring + j
        v_int_n = offset + (n_z - 1) * n_per_ring + j_next
        faces.extend([[v_ext, v_int_n, v_int], [v_ext, v_ext_n, v_int_n]])
        
        # Tapa inferior
        v_ext = j
        v_ext_n = j_next
        v_int = offset + j
        v_int_n = offset + j_next
        faces.extend([[v_ext, v_int, v_int_n], [v_ext, v_int_n, v_ext_n]])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def create_rim_band(r_outer: float, r_inner: float, width: float,
                    n_segments: int = 48) -> 'trimesh.Trimesh':
    """Crea el rim como anillo hueco."""
    return create_tread_band(r_outer, r_inner, width, n_segments)


def create_spokes_fullwidth(r_rim_outer: float, r_tread_inner: float, width: float,
                             n_spokes: int, spoke_thickness: float, 
                             angle_cross: float, n_layers: int = 9) -> 'trimesh.Trimesh':
    """
    Crea radios en múltiples capas Z (FULL-WIDTH).
    
    Args:
        n_layers: Número de capas Z (default 9 para 150mm width)
        angle_cross: Ángulo de cruce en grados
    """
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    meshes = []
    
    # Calcular posiciones Z para cada capa
    z_positions = np.linspace(-width/2 + spoke_thickness/2, 
                               width/2 - spoke_thickness/2, 
                               n_layers)
    
    log.info(f"Creando {n_spokes} radios x {n_layers} capas = {n_spokes * n_layers} radios totales")
    log.info(f"(penetración={spoke_thickness*0.8:.1f}mm, angle_cross={angle_cross:.1f}°)")
    
    # Penetración para asegurar intersección
    penetration = spoke_thickness * 0.8
    
    for layer_idx, z_layer in enumerate(z_positions):
        # Alternar ángulo de cruce en cada capa
        current_cross = angle_cross if layer_idx % 2 == 0 else -angle_cross
        angle_rad = np.radians(current_cross)
        
        angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
        
        for i, angle in enumerate(angles):
            # Puntos con penetración calculados para formar un ángulo de cruce
            p_rim = np.array([
                (r_rim_outer - penetration) * np.cos(angle),
                (r_rim_outer - penetration) * np.sin(angle),
                z_layer
            ])
            
            p_tread = np.array([
                (r_tread_inner + penetration) * np.cos(angle + angle_rad),
                (r_tread_inner + penetration) * np.sin(angle + angle_rad),
                z_layer
            ])
            
            vec = p_tread - p_rim
            length = np.linalg.norm(vec)
            
            if length < 0.01:
                continue
            
            # Crear cilindro y transformar: ROTAR PRIMERO, luego transladar
            cyl = trimesh.creation.cylinder(radius=spoke_thickness/2, height=length, sections=8)
            
            z_axis = np.array([0, 0, 1])
            direction = vec / length
            cross_v = np.cross(z_axis, direction)
            
            if np.linalg.norm(cross_v) > 1e-6:
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot_matrix = trimesh.transformations.rotation_matrix(rot_angle, cross_v)
                cyl.apply_transform(rot_matrix)
            
            # Transladar DESPUÉS de rotar
            mid = (p_rim + p_tread) / 2
            cyl.apply_translation(mid)
            
            meshes.append(cyl)
    
    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()


def create_wheel_hub_fullwidth(r_outer: float, axle_diameter: float, width: float,
                                n_spokes: int = 6, n_layers: int = 9,
                                hub_angle_cross: float = 0.0) -> 'trimesh.Trimesh':
    """
    Crea hub central con radios en múltiples capas Z.
    
    Args:
        hub_angle_cross: Ángulo de cruce de los radios del hub en grados (0=radial, 30=cruzado)
    """
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    meshes = []
    r_axle = axle_diameter / 2
    # BUG#12: usar r_axle para determinar r_inner del hub, no r_outer*0.3
    # r_outer*0.3 escala con el rim, no con el eje (siempre 20mm)
    r_inner = max(r_axle * 1.2, 10.0)  # disco del hub empieza sobre el eje real
    
    # Disco central (hub body)
    theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
    n_z = 3
    z = np.linspace(-width/2 * 0.7, width/2 * 0.7, n_z)
    
    vertices = []
    faces = []
    
    for zi in z:
        for ti in theta:
            vertices.append([r_inner * np.cos(ti), r_inner * np.sin(ti), zi])
    
    for zi in z:
        for ti in theta:
            vertices.append([r_axle * np.cos(ti), r_axle * np.sin(ti), zi])
    
    vertices = np.array(vertices)
    n_per_ring = len(theta)
    
    for i in range(n_z - 1):
        for j in range(n_per_ring):
            j_next = (j + 1) % n_per_ring
            
            v0 = i * n_per_ring + j
            v1 = i * n_per_ring + j_next
            v2 = (i + 1) * n_per_ring + j_next
            v3 = (i + 1) * n_per_ring + j
            faces.extend([[v0, v1, v2], [v0, v2, v3]])
            
            offset = n_z * n_per_ring
            v0i = offset + i * n_per_ring + j
            v1i = offset + i * n_per_ring + j_next
            v2i = offset + (i + 1) * n_per_ring + j_next
            v3i = offset + (i + 1) * n_per_ring + j
            faces.extend([[v0i, v2i, v1i], [v0i, v3i, v2i]])
    
    # Tapas
    for j in range(n_per_ring):
        j_next = (j + 1) % n_per_ring
        
        v_ext = (n_z - 1) * n_per_ring + j
        v_ext_n = (n_z - 1) * n_per_ring + j_next
        v_int = n_z * n_per_ring + (n_z - 1) * n_per_ring + j
        v_int_n = n_z * n_per_ring + (n_z - 1) * n_per_ring + j_next
        faces.extend([[v_ext, v_int, v_ext_n], [v_ext_n, v_int, v_int_n]])
        
        v_ext = j
        v_ext_n = j_next
        v_int = n_z * n_per_ring + j
        v_int_n = n_z * n_per_ring + j_next
        faces.extend([[v_ext, v_ext_n, v_int], [v_ext_n, v_int_n, v_int]])
    
    hub_mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    meshes.append(hub_mesh)
    
    # Radios del hub al rim en múltiples capas
    z_positions = np.linspace(-width/2 * 0.6, width/2 * 0.6, n_layers)
    
    hub_angle_rad = np.radians(hub_angle_cross)
    for layer_idx, z_layer in enumerate(z_positions):
        layer_cross = hub_angle_rad if layer_idx % 2 == 0 else -hub_angle_rad
        for i in range(n_spokes):
            # BUG#15 fix: Renombrar variable local del loop a spoke_angle para evitar 
            # shadowing con la variable 'theta' del hub body en el scope superior.
            spoke_angle = 2 * np.pi * i / n_spokes
            
            p_inner = np.array([r_inner * np.cos(spoke_angle), r_inner * np.sin(spoke_angle), z_layer])
            p_outer = np.array([r_outer * np.cos(spoke_angle + layer_cross), r_outer * np.sin(spoke_angle + layer_cross), z_layer])
            
            vec = p_outer - p_inner
            length = np.linalg.norm(vec)
            
            if length < 0.01:
                continue
            
            cyl = trimesh.creation.cylinder(radius=r_axle * 0.4, height=length, sections=8)
            
            # ROTAR PRIMERO, luego transladar
            z_axis = np.array([0, 0, 1])
            direction = vec / length
            cross_v = np.cross(z_axis, direction)
            
            if np.linalg.norm(cross_v) > 1e-6:
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot_matrix = trimesh.transformations.rotation_matrix(rot_angle, cross_v)
                cyl.apply_transform(rot_matrix)
            
            mid = (p_inner + p_outer) / 2
            cyl.apply_translation(mid)
            
            meshes.append(cyl)
    
    return trimesh.util.concatenate(meshes)


class SpringTireGeometry:
    """
    Genera geometría Spring Tire FULL-WIDTH (implementación canónica v4).

    NOTA DE AMBIGÜEDAD (BUG#13 fix): spring_tire_geometry.py (sin sufijo _v4) define
    otra clase con este mismo nombre que usa boolean union en lugar de
    concatenate. Esta clase (_v4) es la CANÓNICA usada por el pipeline:
      - fase1_spring_tire_dataset.py
      - spring_tire_genome.py (vía export_stl)

    Si importas SpringTireGeometry desde spring_tire_geometry.py obtendrás
    la versión con boolean union (más lenta, menos estable).
    Importa siempre desde spring_tire_geometry_v4 para el pipeline de producción.
    """
    
    def __init__(self, params: Dict):
        self.params = params
        self._validate_params()
        
        self.r_ext = params['d_ext'] / 2
        self.r_rim = params['d_rim'] / 2
        self.width = params['width']
        self.n_radios = params['n_radios']
        self.wall_thick = params['wall_thickness']
        self.angle_cross = params['angle_cross']
        
        # Parámetros pantográficos del genoma extendido
        self.spoke_thick = params.get('spoke_thickness', self.wall_thick)  # grosor independiente de spokes
        self.n_layers = params.get('n_layers', 9)                          # capas Z
        self.hub_angle_cross = params.get('hub_angle_cross', 0.0)          # ángulo cruce hub
        tread_rim_factor = params.get('tread_rim_factor', 2.0)             # multiplicador grosor tread/rim
        
        # Espacio disponible entre tread y rim
        available_gap = (self.r_ext - self.r_rim) / 2  # mitad del gap total disponible
        
        # Calcular espesores deseados y limitarlos para garantizar gap >= 10mm
        tread_thick_desired = self.wall_thick * tread_rim_factor
        rim_thick_desired   = self.wall_thick * tread_rim_factor * 0.75
        total_desired = tread_thick_desired + rim_thick_desired
        min_gap = 10.0
        max_total = (self.r_ext - self.r_rim) - min_gap
        
        if total_desired > max_total and max_total > 0:
            scale = max_total / total_desired
            tread_thick_desired *= scale
            rim_thick_desired   *= scale
        
        self.tread_thickness = max(tread_thick_desired, self.wall_thick)
        self.rim_thickness   = max(rim_thick_desired,   self.wall_thick * 0.5)
        
        # Superficies de conexión
        self.r_tread_inner = self.r_ext - self.tread_thickness
        self.r_rim_outer = self.r_rim + self.rim_thickness
        self.r_rim_inner = max(self.r_rim - self.rim_thickness * 0.5, 20)
    
    def _validate_params(self):
        """Valida parámetros geométricos básicos."""
        d_ext = self.params['d_ext']
        d_rim = self.params['d_rim']
        
        if d_rim >= d_ext * 0.9:
            raise ValueError(f"d_rim ({d_rim}mm) debe ser menor que d_ext ({d_ext}mm)")
    
    def generate(self, n_layers: int = None, hub_angle_cross: float = None) -> Optional['trimesh.Trimesh']:
        """
        Genera geometría completa FULL-WIDTH.
        
        Args:
            n_layers: Capas Z (si None, usa params['n_layers'] del genoma)
            hub_angle_cross: Ángulo hub (si None, usa params['hub_angle_cross'] del genoma)
        """
        # Los valores del genoma son por defecto
        if n_layers is None:
            n_layers = self.n_layers
        if hub_angle_cross is None:
            hub_angle_cross = self.hub_angle_cross
        if not TRIMESH_AVAILABLE:
            return None

        log.info("="*60)
        log.info(f"Generando Spring Tire v5.4 (FIXED, {n_layers} capas)")
        log.info("="*60)

        meshes = []

        # 1. TREAD
        log.info("[1] Tread...")
        tread = create_tread_band(
            r_outer=self.r_ext,
            r_inner=self.r_tread_inner,
            width=self.width
        )
        if tread and len(tread.faces) > 0:
            meshes.append(tread)
            log.info(f"    ✓ {len(tread.faces)} caras")

        # 2. RIM
        log.info("[2] Rim...")
        rim = create_rim_band(
            r_outer=self.r_rim_outer,
            r_inner=self.r_rim_inner,
            width=self.width * 0.9
        )
        if rim and len(rim.faces) > 0:
            meshes.append(rim)
            log.info(f"    ✓ {len(rim.faces)} caras")

        # 3. SPOKES (múltiples capas)
        log.info("[3] Spokes...")
        spokes = create_spokes_fullwidth(
            r_rim_outer=self.r_rim_outer,
            r_tread_inner=self.r_tread_inner,
            width=self.width,
            n_spokes=self.n_radios,
            spoke_thickness=self.spoke_thick,   # grosor independiente del genoma
            angle_cross=self.angle_cross,
            n_layers=n_layers
        )
        meshes.append(spokes)
        log.info(f"    ✓ {len(spokes.faces)} caras")

        # 4. HUB
        log.info("[4] Hub...")
        hub = create_wheel_hub_fullwidth(
            r_outer=self.r_rim,
            axle_diameter=20.0,
            width=self.width * 0.7,
            n_spokes=max(6, self.n_radios // 2),
            n_layers=n_layers,
            hub_angle_cross=hub_angle_cross
        )
        if hub and len(hub.faces) > 0:
            meshes.append(hub)
            log.info(f"    ✓ {len(hub.faces)} caras")

        if not meshes:
            return None

        # 5. Fusionar vértices cercanos (mejora calidad)
        log.info("\n[5] Fusionando vértices cercanos...")
        combined = trimesh.util.concatenate(meshes)
        
        n_vertices_before = len(combined.vertices)
        combined.merge_vertices()
        n_merged = n_vertices_before - len(combined.vertices)
        
        if n_merged > 0:
            log.info(f"    Vértices fusionados: {n_merged}")

        log.info(f"\n✓ Total: {len(combined.faces)} caras, {len(combined.vertices)} vértices")
        
        return combined

    def generate_and_validate(self) -> Tuple[Optional['trimesh.Trimesh'], Dict]:
        """Genera y valida."""
        from utils.spring_tire_validator import validate_spring_tire_stl
        
        mesh = self.generate()
        
        if mesh is None:
            return None, {'valid': False, 'error': 'No se generó malla'}
        
        validation = validate_spring_tire_stl(mesh, self.params)
        return mesh, validation


def export_stl(params: Dict, filepath: str, validate: bool = True, 
               n_layers: int = 9) -> Tuple[bool, Dict]:
    """Exporta STL con validación."""
    tire = SpringTireGeometry(params)

    if validate:
        mesh = tire.generate(n_layers=n_layers)
        from utils.spring_tire_validator import validate_spring_tire_stl
        validation = validate_spring_tire_stl(mesh, params) if mesh else {'valid': False}
    else:
        mesh = tire.generate(n_layers=n_layers)
        validation = {'valid': mesh is not None}

    if mesh is None:
        return False, validation

    try:
        mesh.export(filepath)
        return True, validation
    except Exception as e:
        log.error(f"Error exportando: {e}")
        return False, validation
