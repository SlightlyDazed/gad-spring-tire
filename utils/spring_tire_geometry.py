"""
spring_tire_geometry.py - Geometría NASA Spring Tire v6.2

Usa operaciones booleanas para unir componentes y crear geometría conectada.
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
    """Crea la banda de rodadura como cilindro hueco."""
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    n_z = 5
    z = np.linspace(-width/2, width/2, n_z)

    vertices = []
    faces = []

    for zi in z:
        for ti in theta:
            vertices.append([r_outer * np.cos(ti), r_outer * np.sin(ti), zi])

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

    # Tapas
    for j in range(n_per_ring):
        j_next = (j + 1) % n_per_ring
        v_ext = (n_z - 1) * n_per_ring + j
        v_ext_n = (n_z - 1) * n_per_ring + j_next
        v_int = offset + (n_z - 1) * n_per_ring + j
        v_int_n = offset + (n_z - 1) * n_per_ring + j_next
        faces.extend([[v_ext, v_int_n, v_int], [v_ext, v_ext_n, v_int_n]])

        v_ext = j
        v_ext_n = j_next
        v_int = offset + j
        v_int_n = offset + j_next
        faces.extend([[v_ext, v_int, v_int_n], [v_ext, v_int_n, v_ext_n]])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def create_rim_band(r_outer: float, r_inner: float, width: float,
                    n_segments: int = 48) -> 'trimesh.Trimesh':
    """Crea el rim como cilindro hueco."""
    return create_tread_band(r_outer, r_inner, width, n_segments)


def create_spoke_cylinders(r_rim_outer: float, r_tread_inner: float, width: float,
                           n_spokes: int, spoke_thickness: float, angle_cross: float) -> List['trimesh.Trimesh']:
    """
    Crea radios como cilindros que penetran en rim y tread.
    """
    if not TRIMESH_AVAILABLE:
        return []

    meshes = []
    
    n_layers = max(1, int(width / (spoke_thickness * 3)))
    if n_layers % 2 == 0: n_layers += 1
    z_layers = np.linspace(-width/2 + spoke_thickness*1.5, width/2 - spoke_thickness*1.5, n_layers) if n_layers > 1 else [0.0]
    
    penetration = spoke_thickness * 1.5
    
    log.info(f"Creando {n_spokes} radios x {n_layers} capas = {n_spokes * n_layers} radios totales (penetración={penetration:.1f}mm, angle_cross={np.degrees(angle_cross):.1f}°)")
    
    layer_idx = 0
    for zi in z_layers:
        layer_idx += 1
        current_cross = angle_cross if layer_idx % 2 == 0 else -angle_cross
        
        for i in range(n_spokes):
            angle = 2 * np.pi * i / n_spokes
            
            p_rim = np.array([
                (r_rim_outer - penetration) * np.cos(angle),
                (r_rim_outer - penetration) * np.sin(angle),
                zi
            ])
            p_tread = np.array([
                (r_tread_inner + penetration) * np.cos(angle + current_cross),
                (r_tread_inner + penetration) * np.sin(angle + current_cross),
                zi
            ])
            
            vec = p_tread - p_rim
            length = np.linalg.norm(vec)
            
            if length < 0.01:
                continue
            
            cyl = trimesh.creation.cylinder(radius=spoke_thickness/2, height=length, sections=8)
            
            # Z axis rotation
            z_axis = np.array([0, 0, 1])
            direction = vec / length
            cross = np.cross(z_axis, direction)
            
            if np.linalg.norm(cross) > 1e-6:
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot_matrix = trimesh.transformations.rotation_matrix(rot_angle, cross, [0, 0, 0])
                cyl.apply_transform(rot_matrix)
                
            # Translation
            mid = (p_rim + p_tread) / 2
            cyl.apply_translation(mid)
            
            meshes.append(cyl)

    return meshes


def create_hub_center(r_outer: float, axle_diameter: float, width: float,
                      n_spokes: int = 6) -> 'trimesh.Trimesh':
    """Crea el hub central con radios al rim."""
    if not TRIMESH_AVAILABLE:
        return trimesh.Trimesh()

    meshes = []
    r_axle = axle_diameter / 2
    r_inner = r_outer * 0.3

    # Disco central
    theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
    n_z = 3
    z = np.linspace(-width/2, width/2, n_z)

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

    # Radios del hub al rim
    # Radios del hub al rim
    spoke_radius = r_axle * 0.4
    # Distribuir en el ancho
    n_layers = max(3, int(width / (spoke_radius * 4)))
    if n_layers % 2 == 0:
        n_layers += 1
    z_layers = np.linspace(-width/2 + spoke_radius*2, width/2 - spoke_radius*2, n_layers)
    
    for zi in z_layers:
        for i in range(n_spokes):
            # Alternar el ángulo ligeramente entre capas para mayor rigidez (patrón cruzado)
            theta = 2 * np.pi * i / n_spokes
            
            p_inner = np.array([r_inner * np.cos(theta), r_inner * np.sin(theta), zi])
            p_outer = np.array([r_outer * np.cos(theta), r_outer * np.sin(theta), zi])
            vec = p_outer - p_inner
            length = np.linalg.norm(vec)

            if length < 0.01:
                continue

            cyl = trimesh.creation.cylinder(radius=spoke_radius, height=length, sections=8)
            
            # Orientar el cilindro A LO LARGO del eje Z local y luego rotarlo a la dirección deseada
            # 1. Rotación del cilindro para que apunte en la dirección `vec`
            z_axis = np.array([0, 0, 1])
            direction = vec / length
            cross = np.cross(z_axis, direction)
            
            if np.linalg.norm(cross) > 1e-6:
                rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                rot_matrix = trimesh.transformations.rotation_matrix(rot_angle, cross, [0, 0, 0])
                cyl.apply_transform(rot_matrix)
            
            # 2. Traslación al punto medio
            mid = (p_inner + p_outer) / 2
            cyl.apply_translation(mid)

            meshes.append(cyl)

    return trimesh.util.concatenate(meshes)


class SpringTireGeometry:
    """
    Geometría Spring Tire con boolean union (versión experimental).

    ADVERTENCIA (BUG#13 fix): Esta clase comparte nombre con SpringTireGeometry en
    spring_tire_geometry_v4.py. Para el pipeline de producción usa _v4.
    Esta versión usa trimesh.boolean.union (requiere blender/manifold backend)
    y es más lenta e inestable. Usar solo para prototipado o comparación.
    """

    def __init__(self, params: Dict):
        self.params = params
        self._validate_params()

        self.r_ext = params['d_ext'] / 2
        self.r_rim = params['d_rim'] / 2
        self.width = params['width']
        self.n_radios = params['n_radios']
        self.wall_thick = params['wall_thickness']
        self.angle_cross = np.radians(params['angle_cross'])

        # Espesores
        self.tread_thickness = self.wall_thick * 2.0
        self.rim_thickness = self.wall_thick * 1.5

        # Superficies de conexión
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

        if gap < 10.0:
            raise ValueError(f"Espacio insuficiente: {gap:.1f}mm < 10mm")

    def generate(self) -> Optional['trimesh.Trimesh']:
        """Genera geometría completa usando boolean union."""
        if not TRIMESH_AVAILABLE:
            return None

        log.debug("="*60)
        log.debug("Generando Spring Tire v6.2 - Boolean Union")
        log.debug("="*60)

        meshes = []

        # 1. TREAD
        log.debug("[1] Tread...")
        tread = create_tread_band(
            r_outer=self.r_ext,
            r_inner=self.r_tread_inner,
            width=self.width
        )
        if tread and len(tread.faces) > 0:
            meshes.append(tread)
            log.debug(f"    ✓ {len(tread.faces)} caras")

        # 2. RIM
        log.debug("[2] Rim...")
        rim = create_rim_band(
            r_outer=self.r_rim_outer,
            r_inner=self.r_rim_inner,
            width=self.width * 0.9
        )
        if rim and len(rim.faces) > 0:
            meshes.append(rim)
            log.debug(f"    ✓ {len(rim.faces)} caras")

        # 3. SPOKES (cilindros penetrantes)
        log.debug("[3] Spokes...")
        spokes = create_spoke_cylinders(
            r_rim_outer=self.r_rim_outer,
            r_tread_inner=self.r_tread_inner,
            width=self.width,
            n_spokes=self.n_radios,
            spoke_thickness=self.wall_thick,
            angle_cross=self.angle_cross
        )
        meshes.extend(spokes)
        log.debug(f"    ✓ {len(spokes)} radios")

        # 4. HUB
        log.debug("[4] Hub...")
        hub = create_hub_center(
            r_outer=self.r_rim,
            axle_diameter=20.0,
            width=self.width * 0.7,
            n_spokes=max(6, self.n_radios // 2)
        )
        if hub and len(hub.faces) > 0:
            meshes.append(hub)
            log.debug(f"    ✓ {len(hub.faces)} caras")

        if not meshes:
            return None

        # 5. Combinar todo con boolean union
        log.debug("\n[5] Aplicando boolean union...")
        try:
            result = meshes[0].copy()

            for i, mesh in enumerate(meshes[1:], 1):
                try:
                    result = trimesh.boolean.union([result, mesh])
                    log.debug(f"    Union {i+1}/{len(meshes)}: OK")
                except Exception as e:
                    log.debug(f"    Union {i+1}/{len(meshes)}: {e} → concat")
                    result = trimesh.util.concatenate([result, mesh])

            log.debug(f"\n✓ Total: {len(result.faces)} caras, {len(result.vertices)} vértices")
            return result

        except Exception as e:
            log.debug(f"Error en boolean union: {e} → concatenación simple")
            combined = trimesh.util.concatenate(meshes)
            return combined

    def generate_and_validate(self) -> Tuple[Optional['trimesh.Trimesh'], Dict]:
        """Genera y valida."""
        from utils.spring_tire_validator import validate_spring_tire_stl

        mesh = self.generate()

        if mesh is None:
            return None, {'valid': False, 'error': 'No se generó malla'}

        validation = validate_spring_tire_stl(mesh, self.params)
        return mesh, validation


def export_stl(params: Dict, filepath: str, validate: bool = True) -> Tuple[bool, Dict]:
    """Exporta STL con validación."""
    tire = SpringTireGeometry(params)

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
