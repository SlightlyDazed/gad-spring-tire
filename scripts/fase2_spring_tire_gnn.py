#!/usr/bin/env python3
"""
fase2_spring_tire_gnn.py - Fase 2: Transfer Learning GNN v5.0

Entrena modelos GNN usando transfer learning desde los modelos existentes.
Los modelos previos fueron entrenados en dataset de pantógrafo genérico (fase 2).
Ahora hacemos fine-tuning con datos Spring Tire.
"""

import os
import sys
import json
import numpy as np
import logging
from typing import Dict, List, Tuple
import pickle

# =============================================================================
# ROCm / amdgpu GPU ESTABILIDAD
# MUST set BEFORE importing torch — onces torch/hip init, estas vars no tienen efecto.
# Causa del reboot: amdgpu GL_UNKNOWN_CONTEXT_RESET_KHR → X11 crash → kernel panic
# cuando la GPU se queda sin VRAM o hay un memory race en HSA.
# =============================================================================
os.environ.setdefault('HSA_ENABLE_SDMA', '0')          # desactiva SDMA DMA que causa hangs
os.environ.setdefault('GPU_MAX_HEAP_SIZE', '90')        # límite de heap al 90% del VRAM
os.environ.setdefault('GPU_MAX_ALLOC_PERCENT', '90')    # limita allocaciones al 90% del VRAM
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'max_split_size_mb:512')  # evita fragmentación extrema
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')       # usa solo GPU[0] (RX 9070), no la integrada
os.environ.setdefault('ROC_ENABLE_PRE_VEGA', '0')       # prevenimos compatibilidad fallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)
log.info("ROCm safety env vars aplicadas (antes de importar torch)")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spring_tire_genome import N_GENES, decode_genome, MATERIAL_CATALOG

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch no disponible")

try:
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    log.warning("PyTorch Geometric no disponible")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SpringTireGraphBuilder:
    """Construye grafo de Spring Tire para GNN."""

    def __init__(self, params: Dict):
        self.params = params
        self.r_ext = params['d_ext'] / 2
        self.r_rim = params['d_rim'] / 2
        self.width = params['width']
        self.n_radios = params['n_radios']
        self.wall_thick = params['wall_thickness']
        self.angle_cross = np.radians(params['angle_cross'])

    def build_graph(self, params: Dict) -> Dict:
        """Construye representación en grafo de la Spring Tire incluyendo info 3D y materiales."""
        nodes = []
        edges = []
        edge_features = []

        # Información 3D: número de capas (codificado como fíisico a los nodos)
        n_layers = params.get('n_layers', 1)
        spoke_thick = params.get('spoke_thickness', self.wall_thick)
        hub_angle_rad = np.radians(params.get('hub_angle_cross', 0.0))
        width = params.get('width', 100.0)

        # Material por sector: leer E y densidad del catálogo
        from utils.spring_tire_genome import MATERIAL_CATALOG
        material_sectors = params.get('material_sectors', [0]*8)

        def sector_material(theta):
            """Obtiene E_MPa y density según ángulo theta."""
            sector_idx = int((theta % (2*np.pi)) / (2*np.pi) * 8)
            mat_id = int(material_sectors[min(sector_idx, 7)])
            mat = MATERIAL_CATALOG.get(mat_id, MATERIAL_CATALOG[0])
            return mat['E_MPa'], mat['density_g_cm3'], mat['sigma_yield_MPa']

        z_layer = 0.0

        # Nodos del rim
        for i in range(self.n_radios):
            theta = 2 * np.pi * i / self.n_radios
            x = self.r_rim * np.cos(theta)
            y = self.r_rim * np.sin(theta)
            nodes.append([x, y, z_layer, n_layers, width])

        # Nodos del tread
        for i in range(self.n_radios):
            theta = 2 * np.pi * i / self.n_radios
            x = self.r_ext * np.cos(theta)
            y = self.r_ext * np.sin(theta)
            nodes.append([x, y, z_layer, n_layers, width])

        nodes = np.array(nodes)
        n_rim = self.n_radios
        # BUG A fix: NO agregar hub aquí — hub_idx se calcularía en 2*n_radios
        # que coincide exactamente con el PRIMER cross node (vstack lo sobreescribe).
        # Se agrega hub DESPUÉS de todos los cross nodes → hub_idx correcto.

        # Aristas primarios rim → tread
        for i in range(self.n_radios):
            rim_idx = i
            tread_idx = n_rim + i

            p1 = nodes[rim_idx, :3]
            p2 = nodes[tread_idx, :3]
            length = np.linalg.norm(p2 - p1)
            theta = 2 * np.pi * i / self.n_radios
            E, rho, sigma_y = sector_material(theta)

            edges.append([rim_idx, tread_idx])
            edges.append([tread_idx, rim_idx])
            edge_features.append([length, 0.0, spoke_thick, 1.0, n_layers, E/1000, rho, sigma_y])
            edge_features.append([length, 0.0, spoke_thick, 1.0, n_layers, E/1000, rho, sigma_y])

        # Aristas cruzados (patrón pantográfico)
        # FIX Bug #2: pre-construir todos los cross nodes en lista y hacer un
        # único np.vstack al final, en vez de reasignar `nodes` en cada iteración
        # (cada vstack copiaba todo el array → O(n²) en RAM por sample).
        r_mid = (self.r_rim + self.r_ext) / 2
        cross_nodes_list = []
        cross_edge_data = []  # guardar datos de aristas para procesarlos después

        for i in range(self.n_radios):
            i_next = (i + 1) % self.n_radios
            theta_i = 2 * np.pi * i / self.n_radios
            theta_cross = theta_i + self.angle_cross / 2

            x_cross = r_mid * np.cos(theta_cross)
            y_cross = r_mid * np.sin(theta_cross)
            cross_nodes_list.append([x_cross, y_cross, z_layer, n_layers, width])

            E, rho, sigma_y = sector_material(theta_i)
            cross_edge_data.append((i, i_next, E, rho, sigma_y))

        # Un solo vstack para todos los cross nodes
        cross_nodes_arr = np.array(cross_nodes_list)  # [n_radios, 5]
        base_cross_idx = len(nodes)
        nodes = np.vstack([nodes, cross_nodes_arr])

        # Ahora construir aristas usando los índices ya consolidados
        for k, (i, i_next, E, rho, sigma_y) in enumerate(cross_edge_data):
            cross_node_idx = base_cross_idx + k

            p1 = nodes[i, :3]; p2 = nodes[cross_node_idx, :3]
            length = np.linalg.norm(p2 - p1)
            edges.append([i, cross_node_idx]); edges.append([cross_node_idx, i])
            edge_features.append([length, self.angle_cross, spoke_thick*0.8, 2.0, n_layers, E/1000, rho, sigma_y])
            edge_features.append([length, self.angle_cross, spoke_thick*0.8, 2.0, n_layers, E/1000, rho, sigma_y])

            p1 = nodes[cross_node_idx, :3]; p2 = nodes[i_next, :3]
            length = np.linalg.norm(p2 - p1)
            edges.append([cross_node_idx, i_next]); edges.append([i_next, cross_node_idx])
            edge_features.append([length, self.angle_cross, spoke_thick*0.8, 2.0, n_layers, E/1000, rho, sigma_y])
            edge_features.append([length, self.angle_cross, spoke_thick*0.8, 2.0, n_layers, E/1000, rho, sigma_y])

        # BUG A fix: agregar hub DESPUÉS de todos los cross nodes
        hub_idx = len(nodes)
        nodes = np.vstack([nodes, [[0, 0, z_layer, n_layers, width]]])

        # Hub → rim (con hub_angle_cross)
        for i in range(self.n_radios):
            p1 = nodes[hub_idx, :3]; p2 = nodes[i, :3]
            length = np.linalg.norm(p2 - p1)
            theta = 2 * np.pi * i / self.n_radios
            E, rho, sigma_y = sector_material(theta)
            edges.append([hub_idx, i]); edges.append([i, hub_idx])
            edge_features.append([length, hub_angle_rad, self.wall_thick*1.5, 0.0, n_layers, E/1000, rho, sigma_y])
            edge_features.append([length, hub_angle_rad, self.wall_thick*1.5, 0.0, n_layers, E/1000, rho, sigma_y])

        # Node features: [r, theta, z, node_type, n_layers, width]
        node_features = []
        for i, node in enumerate(nodes):
            r = np.linalg.norm(node[:2])
            theta = np.arctan2(node[1], node[0])
            if i == hub_idx:      node_type = 0.0
            elif i < n_rim:       node_type = 1.0
            elif i < 2 * n_rim:   node_type = 2.0
            else:                  node_type = 3.0
            node_features.append([r, theta, z_layer, node_type, n_layers, width/1000.0])

        return {
            'nodes': np.array(nodes),
            'edges': np.array(edges).T,
            'node_features': np.array(node_features),   # [N, 6]
            'edge_features': np.array(edge_features)    # [E, 8]
        }


class SpringTireDataset:
    """Dataset de Spring Tire para entrenamiento."""

    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)

        self.samples = self.data['dataset']

        # BUG E fix: verificar primeros 10 samples, no solo sample[0]
        # Si sample[0] fue con FEA 2D y carece de keys de FEM 3D, se excluirían
        # targets válidos que aparecen en el resto del dataset.
        self.target_names = []
        check_n = min(10, len(self.samples))
        for name in self.data['target_names']:
            # Un target es numérico si al menos 8 de 10 samples lo tienen como número
            numeric_count = sum(
                isinstance(s['targets'].get(name), (int, float, np.integer, np.floating))
                for s in self.samples[:check_n]
            )
            if numeric_count >= max(1, check_n * 8 // 10):
                self.target_names.append(name)

        # Scalers para targets
        self.target_scalers = {}
        for target_name in self.target_names:
            scaler = StandardScaler()
            values = np.array([s['targets'][target_name] for s in self.samples]).reshape(-1, 1)
            scaler.fit(values)
            self.target_scalers[target_name] = scaler

        log.info(f"Dataset cargado: {len(self.samples)} muestras")
        log.info(f"Targets numéricos a predecir ({len(self.target_names)}): {self.target_names}")

        # Keys FEM con rangos enormes que necesitan log1p antes de escalar
        self.LOG1P_KEYS = {'life_cycles', 'energy_stored_mJ', 'max_load_N', 'k_spoke_Nmm',
                           'stiffness_kN_m', 'lateral_stiffness_kN_m', 'torsional_stiffness_Nm_rad'}

        self.FEM2D_KEYS = [
            'stiffness_kN_m', 'lateral_stiffness_kN_m', 'torsional_stiffness_Nm_rad',
            'max_deflection_mm', 'max_load_N', 'sigma_vm_max_MPa', 'safety_factor',
            'plastic_strain_percent', 'recovery_ratio', 'shape_memory_ratio',
            'energy_stored_mJ', 'life_cycles', 'goodman_ratio'
        ]
        self.FEM3D_KEYS = [
            'max_displacement_mm', 'k_spoke_Nmm', 'k_vertical_kNm',
            'sigma_vm_MPa', 'sigma_normal_MPa', 'tau_max_MPa',
            'plastic_strain_pct', 'SF_strength', 'buckling_load_factor'
        ]
        self.ALL_FEA_KEYS = self.FEM2D_KEYS + self.FEM3D_KEYS

        # Construir fea_vecs de todo el dataset para calcular scalers
        raw_feas = []
        for s in self.samples:
            raw_feas.append(self._extract_fea_raw(s))
        raw_feas = np.array(raw_feas)  # [N, 22]

        self.fea_scaler = StandardScaler()
        self.fea_scaler.fit(raw_feas)
        log.info(f"fea_vec scaler ajustado: {raw_feas.shape[1]} features")

        # Cambio 1: liberar RAM pesada que ya no servirá
        del raw_feas
        self.samples = list(self.samples)  # desconectar de self.data
        del self.data
        import gc; gc.collect()

    def _safe_float(self, d, k):
        """Extrae float seguro, retorna 0 para None/NaN/Inf."""
        v = d.get(k, 0.0)
        try:
            v = float(v)
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return np.log1p(abs(v)) if k in self.LOG1P_KEYS else v
        except:
            return 0.0

    def _extract_fea_raw(self, sample):
        """Extrae fea_vec crudo (pre-escaler) para un sample.
        BUG#11: Respeta fea_type_used para construir fea_vec desde el mismo
        análisis que se usó para calcular targets en Fase 1.
        """
        fea = sample.get('fea_results', {})
        fea_type = fea.get('fea_type_used', 'fem2d')
        fea2d = fea.get('fea_2d', {})
        fea3d = fea.get('fea_3d', {})
        # Usar el primary correcto según fea_type_used
        primary = fea3d if (fea_type == 'fem3d' and fea3d) else fea2d
        return np.array(
            [self._safe_float(primary, k) for k in self.FEM2D_KEYS] +
            [self._safe_float(fea3d, k) for k in self.FEM3D_KEYS],
            dtype=np.float32
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        params = sample['params']

        # Construir grafo con info 3D y materiales
        builder = SpringTireGraphBuilder(params)
        graph = builder.build_graph(params)

        # Targets normalizados
        targets = []
        for target_name in self.target_names:
            value = sample['targets'][target_name]
            normalized = self.target_scalers[target_name].transform([[value]])[0][0]
            targets.append(normalized)

        # --- Extraer y normalizar features FEM 2D+3D ---
        raw_fea = self._extract_fea_raw(sample)
        fea_vec = self.fea_scaler.transform(raw_fea.reshape(1, -1))[0].astype(np.float32)
        # Clampear outliers extremos
        fea_vec = np.clip(fea_vec, -5.0, 5.0)

        # Cambio 2: devolver Data(torch) directo para ahorrar pico temporal de RAM,
        # en lugar de dict NumPy
        return Data(
            x=torch.tensor(graph['node_features'], dtype=torch.float),
            edge_index=torch.tensor(graph['edges'], dtype=torch.long),
            edge_attr=torch.tensor(graph['edge_features'], dtype=torch.float),
            y=torch.tensor(np.array(targets, dtype=np.float32), dtype=torch.float).unsqueeze(0),
            genome=torch.tensor(np.array(sample['genome'], dtype=np.float32), dtype=torch.float),
            fea_vec=torch.tensor(fea_vec, dtype=torch.float)
        )

    def get_target_scalers(self):
        return self.target_scalers

if PYG_AVAILABLE:
    class SpringTireGNN(torch.nn.Module):
        """GNN para predecir propiedades de Spring Tire.

        Inputs:
        - Grafo: nodos [r, theta, z, node_type, n_layers, width], aristas [len, angle, thick, type, n_layers, E, rho, sigma_y]
        - Genoma: vector de 18 genes [0-1]
        - FEA vector: 22 features clave de FEM 2D (13) + 3D (9)
        """

        def __init__(self, node_dim=6, edge_dim=8, genome_dim=18, fea_dim=22,
                     hidden_dim=256, out_dim=17):
            super().__init__()

            # Encoder de nodos y aristas
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # Encoder del genoma (18 genes normalizados [0-1])
            self.genome_encoder = nn.Sequential(
                nn.Linear(genome_dim, 64), nn.ReLU(),
                nn.Linear(64, 64)
            )

            # Encoder de features FEM 2D+3D
            self.fea_encoder = nn.Sequential(
                nn.Linear(fea_dim, 64), nn.ReLU(),
                nn.Linear(64, 64)
            )

            # Capas GNN (GATConv)
            self.conv1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, edge_dim=hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=2, edge_dim=hidden_dim)
            self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=hidden_dim)

            # Dim total tras concatenar: hidden + genome(64) + fea(64)
            fused_dim = hidden_dim + 64 + 64

            # MLPs de salida: una cabeza por target
            self.target_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim), nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(out_dim)
            ])

        def forward(self, x, edge_index, edge_attr, batch, genome, fea_vec):
            x = self.node_encoder(x.float())
            edge_attr = self.edge_encoder(edge_attr.float())

            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))

            # Pooling global → 1 vector por grafo
            z_graph = global_mean_pool(x, batch)           # [B, hidden]
            B = z_graph.shape[0]

            # PyG aplana tensores auxiliares al hacer batch → reshape necesario
            z_genome = self.genome_encoder(genome.float().view(B, -1)) # [B, 64]
            z_fea    = self.fea_encoder(fea_vec.float().view(B, -1))   # [B, 64]

            # Fusionar las 3 fuentes de información
            z = torch.cat([z_graph, z_genome, z_fea], dim=1)  # [B, hidden+128]

            outs = [head(z) for head in self.target_heads]
            return torch.cat(outs, dim=1)
def train_gnn_transfer_learning(dataset_path: str,
                                 pretrained_models_dir: str = None,
                                 output_dir: str = 'results/models',
                                 n_models: int = 5,
                                 epochs: int = 100,
                                 lr: float = 0.001,
                                 batch_size: int = 2048,
                                 hidden_dim: int = 256) -> Dict:
    """
    Entrena GNN con transfer learning desde modelos pre-entrenados.

    Args:
        dataset_path: Ruta al dataset Spring Tire
        pretrained_models_dir: Directorio con modelos pre-entrenados (opcional)
        output_dir: Directorio de salida
        n_models: Número de modelos en ensemble
        epochs: Épocas de entrenamiento
        lr: Learning rate

    Returns:
        Dict con información de los modelos entrenados
    """
    """
    Entrena GNN con transfer learning desde modelos pre-entrenados.

    Args:
        dataset_path: Ruta al dataset Spring Tire
        pretrained_models_dir: Directorio con modelos pre-entrenados (opcional)
        output_dir: Directorio de salida
        n_models: Número de modelos en ensemble
        epochs: Épocas de entrenamiento
        lr: Learning rate

    Returns:
        Dict con información de los modelos entrenados
    """
    if not TORCH_AVAILABLE or not PYG_AVAILABLE or not SKLEARN_AVAILABLE:
        log.error("Dependencias faltantes: torch, torch_geometric, sklearn")
        return {}

    log.info("="*70)
    log.info("FASE 2: Transfer Learning GNN v5.0")
    log.info("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Cargar dataset
    dataset = SpringTireDataset(dataset_path)
    target_scalers = dataset.get_target_scalers()

    # Guardar scalers
    scalers_path = os.path.join(output_dir, 'target_scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(target_scalers, f)
    log.info(f"Scalers guardados: {scalers_path}")

    # Guardar fea_scaler (necesario para Fase 3 NSGA)
    fea_scaler_path = os.path.join(output_dir, 'fea_scaler.pkl')
    with open(fea_scaler_path, 'wb') as f:
        pickle.dump(dataset.fea_scaler, f)
    log.info(f"fea_scaler guardado: {fea_scaler_path}")

    # Cambio 3 + 4: Evitar loop que pre-construye y acumula grafos.
    # Usar _IdxSubset para que DataLoader llame a __getitem__ (lazy construction).
    class _IdxSubset:
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.ds[self.indices[i]]

    _indices = list(range(len(dataset)))
    _train_idx, _val_idx = train_test_split(_indices, test_size=0.15, random_state=42)

    train_subset = _IdxSubset(dataset, _train_idx)
    val_subset   = _IdxSubset(dataset, _val_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Usando dispositivo: {device}")
    # AMP: FP16 en GPU → mitad de VRAM por tensor, ~2x throughput
    use_amp = device.type == 'cuda'
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    log.info(f"AMP FP16: {'ACTIVADO' if use_amp else 'desactivado'}")

    # FIX Bug #1: pin_memory y num_workers solo son útiles con GPU.
    # En CPU: pin_memory duplica cada batch en RAM; persistent_workers=True
    # mantiene 8 procesos vivos entre epochs acumulando memoria continuamente.
    _use_pin    = device.type == 'cuda'
    _n_workers  = 4 if device.type == 'cuda' else 0
    _persistent = _n_workers > 0

    train_loader = GeoDataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=_n_workers, pin_memory=_use_pin,
                                 persistent_workers=_persistent)
    val_loader   = GeoDataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                                 num_workers=_n_workers, pin_memory=_use_pin,
                                 persistent_workers=_persistent)

    model_paths = []
    # Entrenar ensemble
    out_dim = len(dataset.target_names)
    for m in range(n_models):
        log.info(f"--- Entrenando Modelo {m+1}/{n_models} ---")

        model = SpringTireGNN(node_dim=6, edge_dim=8, genome_dim=18, fea_dim=22,
                              hidden_dim=hidden_dim, out_dim=out_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        model_path = os.path.join(output_dir, f'gnn_model_v5_{m}.pt')
        patience_counter = 0
        patience_limit = 25

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                batch.genome, batch.fea_vec)
                    loss = criterion(out, batch.y)
                scaler_amp.scale(loss).backward()
                # CRITICAL FIX: gradient clipping antes del step.
                # Sin clip, gradientes → inf/NaN → GPU calcula basura → VRAM corruption → amdgpu reset.
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                train_loss += loss.item() * batch.num_graphs
                del out, loss, batch  # FIX Bug #3b: liberar tensores en training loop
            train_loss /= len(train_subset)

            # Validation
            # FIX Bug #3: del explícito de out y batch para que PyTorch libere
            # la memoria de los tensores sin esperar al siguiente ciclo del GC.
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                batch.genome, batch.fea_vec)
                    val_loss += criterion(out, batch.y).item() * batch.num_graphs
                    del out, batch
            val_loss /= len(val_subset)

            scheduler.step(val_loss)

            # FIX: forzar GC al final de cada epoch para liberar fragmentación
            import gc; gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
                if epoch % 10 == 0:
                    log.info(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} (Mejor!)")
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    log.info(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

            if patience_counter >= patience_limit:
                log.info(f"Early stopping en epoch {epoch} (paciencia={patience_limit})")
                break

        model_paths.append(model_path)
        log.info(f"Modelo {m+1} finalizado. Mejor Val MSE: {best_val_loss:.4f}")

    # BUG B fix: incluir todas las dimensiones necesarias para reconstruir el modelo
    # en Fase 3 sin hardcodear out_dim, hidden_dim, ni fea_dim.
    model_info = {
        'version': '5.0',
        'n_models': n_models,
        'target_names': dataset.target_names,
        'scalers_path': scalers_path,
        'model_paths': model_paths,
        # Dimensiones del modelo (necesarias para load en Fase 3)
        'out_dim': out_dim,
        'node_dim': 6,
        'edge_dim': 8,
        'genome_dim': 18,
        'fea_dim': 22,
        'hidden_dim': hidden_dim,
        'fea_scaler_path': fea_scaler_path,
    }

    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    log.info(f"\nEntrenamiento de {n_models} modelos completado exitosamente.")
    log.info(f"Info guardada en: {info_path}")

    return model_info


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Entrenar GNN Spring Tire')
    parser.add_argument('--dataset', type=str, default='results/data/spring_tire_dataset.json',
                       help='Ruta al dataset')
    parser.add_argument('--pretrained-dir', type=str, default=None,
                       help='Directorio con modelos pre-entrenados')
    parser.add_argument('--output-dir', type=str, default='results/models',
                       help='Directorio de salida')
    parser.add_argument('--n-models', type=int, default=5,
                       help='Número de modelos en ensemble')
    parser.add_argument('--epochs', type=int, default=300, help='Épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=512,
                         help='Batch size seguro (RX9070). Nunca subir sobre 2048.')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Dim GNN. 256=seguro, 512=max RX9070')
    args = parser.parse_args()

    train_gnn_transfer_learning(
        dataset_path=args.dataset,
        pretrained_models_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        n_models=args.n_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim
    )


if __name__ == '__main__':
    main()
