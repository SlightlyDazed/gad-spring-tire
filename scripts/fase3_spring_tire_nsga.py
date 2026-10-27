#!/usr/bin/env python3
"""
fase3_spring_tire_nsga.py - Fase 3: NSGA-III + FEM 3D Surrogate Refresh v6.1

Estrategia A: Surrogate Refresh
  - El NSGA-III corre con el FEA 2D rápido (~0.01s por evaluación)
  - Cada `fem3d_refresh_every` generaciones:
      1. Toma el top-K del frente de Pareto actual
      2. Los evalúa con FEM 3D de alta fidelidad (~5s por evaluación)
      3. Calcula factores de corrección (k_correction, sigma_correction, etc.)
      4. Aplica esos factores al FEA 2D para el resto de la optimización
  - Al final, el frente de Pareto completo se re-evalúa con FEM 3D

Esto implementa "surrogate-assisted optimization":
  FEA 2D (surrogate rápido) ← calibrado por → FEM 3D (oracle lento)

El ciclo completo:
  NSGA-III gen 1..N  (FEA 2D)
       ↓ cada fem3d_refresh_every gens
  FEM 3D top-K  →  factores de corrección  →  FEA 2D recalibrado
       ↓ fin
  FEM 3D frente Pareto completo  →  resultados finales validados

Cambios v6.1 (bugfixes):
  FIX A: KeyError 'life_km' en SurrogateState.apply() — añadido guard 'in corrected'
  FIX B: k_correction explosivo — añadido clamp físico (K_CORRECTION_MAX = 5.0) en
         SurrogateState.update() y en apply(); se loguea un WARNING si el factor
         calculado excede el límite para que sea visible en los logs
  FIX C: Importación redundante de json (import json as _json) eliminada en dos sitios;
         se usa el json del top-level
  FIX D: model_confidence = 0 no descarta el refresh pero sí reduce el peso del EMA
         (alpha se escala por confidence) para que factores de baja confianza no
         sobrescriban el histórico agresivamente
"""

import os
import sys
import json
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
from utils.spring_tire_genome import (
    N_GENES, decode_genome, create_baseline_genome, validate_genome,
    mutate_genome, crossover_genome
)
from utils.spring_tire_fea import run_spring_tire_fea

# FEM 3D (filtro de alta fidelidad)
try:
    from utils.spring_tire_fem3d import (
        run_fem3d_spoke,
        compute_correction_factors,
        FEM3D_AVAILABLE
    )
except ImportError:
    FEM3D_AVAILABLE = False
    log.warning("utils.spring_tire_fem3d no encontrado → FEM 3D desactivado")

    def run_fem3d_spoke(*args, **kwargs):
        return {}

    def compute_correction_factors(*args, **kwargs):
        return {}

# pymoo
try:
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    log.warning("pymoo no disponible")

# GNN surrogate (Fase 2)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pickle
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch/PyG no disponible — GNN surrogate desactivado")


# ---------------------------------------------------------------------------
# GNN ensemble para predicción rápida en NSGA
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class _SpringTireGNN(torch.nn.Module):
        """Arquitectura GNN identica a la de fase 2 (debe coincidir)."""
        def __init__(self, node_dim=6, edge_dim=8, genome_dim=18, fea_dim=22,
                     hidden_dim=256, out_dim=17):
            super().__init__()
            self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.genome_encoder = nn.Sequential(nn.Linear(genome_dim, 64), nn.ReLU(), nn.Linear(64, 64))
            self.fea_encoder = nn.Sequential(nn.Linear(fea_dim, 64), nn.ReLU(), nn.Linear(64, 64))
            self.conv1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, edge_dim=hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=2, edge_dim=hidden_dim)
            self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=hidden_dim)
            fused = hidden_dim + 128
            self.target_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(fused, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))
                for _ in range(out_dim)
            ])
        def forward(self, x, edge_index, edge_attr, batch, genome, fea_vec):
            x = self.node_encoder(x.float()); edge_attr = self.edge_encoder(edge_attr.float())
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            z_g = global_mean_pool(x, batch)
            B = z_g.shape[0]
            z = torch.cat([z_g, self.genome_encoder(genome.float().view(B, -1)),
                           self.fea_encoder(fea_vec.float().view(B, -1))], dim=1)
            return torch.cat([h(z) for h in self.target_heads], dim=1)


class GNNEnsemblePredictor:
    """
    Carga los 5 modelos GNN entrenados y hace predicción en ensemble.
    Usado como surrogate rápido en el NSGA-III en lugar de solo FEA 2D.
    """

    def __init__(self, models_dir: str = 'results/models', n_models: int = 5,
                 hidden_dim: int = 512):  # FIX Bug #1: debe coincidir con --hidden-dim de fase2 (512)
        self.available = False
        if not TORCH_AVAILABLE:
            log.warning("GNN surrogate no disponible (PyTorch no instalado)")
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scalers_path = os.path.join(models_dir, 'target_scalers.pkl')
        if not os.path.exists(scalers_path):
            log.warning(f"GNN: scalers no encontrados en {scalers_path}")
            return

        with open(scalers_path, 'rb') as f:
            self.target_scalers = pickle.load(f)
        self.target_names = list(self.target_scalers.keys())

        # Log1p keys (deben coincidir con fase 2)
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

        # Cargar fea scaler — re-calculado on-the-fly desde el dataset si existe
        fea_scaler_path = os.path.join(models_dir, 'fea_scaler.pkl')
        if os.path.exists(fea_scaler_path):
            with open(fea_scaler_path, 'rb') as f:
                self.fea_scaler = pickle.load(f)
        else:
            self.fea_scaler = None
            log.warning("GNN: fea_scaler no encontrado — fea_vec se usará sin normalizar")

        # Cargar modelos
        # FIX Bug #2: leer hidden_dim y out_dim del model_info.json que fase2 guardó,
        # en vez de depender del argumento pasado en tiempo de construcción.
        model_info_path = os.path.join(models_dir, 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path) as _f:
                _info = json.load(_f)
            hidden_dim = _info.get('hidden_dim', hidden_dim)
            _out_dim   = _info.get('out_dim',    len(self.target_names))
            n_models   = _info.get('n_models',   n_models)
            log.info(f"GNN: leyendo arquitectura de model_info.json — hidden={hidden_dim}, out={_out_dim}, n_models={n_models}")
        else:
            _out_dim = len(self.target_names)
            log.warning(f"GNN: model_info.json no encontrado — usando hidden_dim={hidden_dim} (puede crashear si no coincide con fase2)")

        self.models = []
        for m in range(n_models):
            path = os.path.join(models_dir, f'gnn_model_v5_{m}.pt')
            if not os.path.exists(path):
                continue
            model = _SpringTireGNN(hidden_dim=hidden_dim, out_dim=_out_dim)
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        if self.models:
            log.info(f"GNN surrogate: {len(self.models)} modelos cargados ({self.device})")
            self.available = True
        else:
            log.warning("GNN: ningún modelo encontrado en results/models/")

    def _build_graph(self, params: Dict):
        """Construye el grafo para un sample (reutiliza la lógica de fase 2)."""
        from scripts.fase2_spring_tire_gnn import SpringTireGraphBuilder
        builder = SpringTireGraphBuilder(params)
        return builder.build_graph(params)

    def _safe_fea_val(self, d, k):
        v = d.get(k, 0.0)
        try:
            v = float(v)
            if np.isnan(v) or np.isinf(v): return 0.0
            return np.log1p(abs(v)) if k in self.LOG1P_KEYS else v
        except: return 0.0

    def predict(self, genome: np.ndarray, params: Dict, fea2d: Dict) -> Dict:
        """
        Predice los 17 targets para un genoma usando el ensemble GNN.
        fea2d: resultado del FEA 2D para construir fea_vec.
        """
        if not self.available:
            return fea2d  # fallback: devolver FEA 2D crudo

        try:
            from torch_geometric.data import Data
            import torch

            g = self._build_graph(params)

            # fea_vec: extraer y normalizar
            fea_raw = np.array(
                [self._safe_fea_val(fea2d, k) for k in self.FEM2D_KEYS] +
                [self._safe_fea_val({}, k) for k in self.FEM3D_KEYS],  # 3D=0 para muestras nuevas
                dtype=np.float32
            )
            if self.fea_scaler is not None:
                fea_raw = self.fea_scaler.transform(fea_raw.reshape(1, -1))[0].astype(np.float32)
            fea_raw = np.clip(fea_raw, -5.0, 5.0)

            data = Data(
                x=torch.tensor(g['node_features'], dtype=torch.float),
                edge_index=torch.tensor(g['edges'], dtype=torch.long),
                edge_attr=torch.tensor(g['edge_features'], dtype=torch.float),
                genome=torch.tensor(genome, dtype=torch.float),
                fea_vec=torch.tensor(fea_raw, dtype=torch.float)
            ).to(self.device)

            preds_list = []
            with torch.no_grad():
                for model in self.models:
                    batch_idx = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
                    out = model(data.x, data.edge_index, data.edge_attr, batch_idx,
                                data.genome.unsqueeze(0), data.fea_vec.unsqueeze(0))
                    preds_list.append(out.cpu().numpy()[0])

            # Ensemble: promedio de las predicciones (en espacio normalizado)
            avg_pred = np.mean(preds_list, axis=0)

            # Desnormalizar usando los scalers guardados
            result = {}
            for i, name in enumerate(self.target_names):
                scaler = self.target_scalers[name]
                val = float(scaler.inverse_transform([[avg_pred[i]]])[0][0])
                # Fallback a FEA2D si la predicción GNN es físicamente inválida
                fea2d_val = fea2d.get(name)
                if fea2d_val is not None and fea2d_val > 0:
                    # Si GNN predice valor negativo o 100x más grande que FEA2D, usar FEA2D
                    if val <= 0 or val > fea2d_val * 100:
                        val = float(fea2d_val)
                result[name] = val

            return result

        except Exception as e:
            log.warning(f"GNN inference falló: {e} — usando FEA 2D")
            return fea2d


# Instancia global del predictor GNN
_gnn_predictor: Optional[GNNEnsemblePredictor] = None


# ---------------------------------------------------------------------------
# Estado del surrogate (factores de corrección acumulados)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FIX B: límites físicos para los factores de corrección del surrogate.
# Si FEM 3D devuelve valores muy distintos al FEA 2D (p.ej. malla demasiado
# gruesa, material mal calibrado) el EMA sin clamp amplifica el error cada
# refresh hasta valores absurdos (se observó k_correction → 96× en v6.0).
# Estos caps son conservadores pero físicamente razonables para una llanta
# de polímero en condiciones marcianas; ajustar si el FEM 3D mejora.
# ---------------------------------------------------------------------------
K_CORRECTION_MAX         = 5.0   # rigidez: FEM 3D raramente difiere >5× de FEA 2D
DEFLECTION_CORRECTION_MAX= 5.0   # deflexión
SIGMA_CORRECTION_MAX     = 5.0   # esfuerzo Von Mises
KT_3D_MAX                = 4.0   # concentración de esfuerzo (Kt teórico máx ~3–4)


@dataclass
class SurrogateState:
    """
    Mantiene los factores de corrección entre FEA 2D y FEM 3D.

    Se actualiza cada vez que se corre el refresh de FEM 3D.
    El FEA 2D multiplica sus resultados por estos factores antes
    de reportarlos al NSGA-III.
    """
    # Factores de corrección por propiedad (1.0 = sin corrección)
    k_correction:           float = 1.0   # rigidez vertical
    deflection_correction:  float = 1.0   # deflexión máxima
    sigma_correction:       float = 1.0   # esfuerzo Von Mises
    Kt_3d:                  float = 1.0   # factor de concentración de esfuerzo

    # Historial de refreshes
    refresh_history:        List[Dict] = field(default_factory=list)
    n_refreshes:            int = 0
    last_refresh_gen:       int = 0
    model_confidence:       float = 0.5   # confianza en el surrogate (0-1)

    def update(self, new_factors: Dict, generation: int):
        """
        Actualiza los factores con suavizado exponencial (EMA).

        FIX B: el alpha del EMA se escala por la confianza del FEM 3D.
          - confidence alta (≈1) → alpha normal 0.6 → acepta datos nuevos
          - confidence baja (≈0) → alpha reducido → no sobreescribe el histórico

        FIX B: todos los factores se clampean a sus máximos físicos ANTES
        de entrar al EMA. Si el valor crudo excede el cap se emite WARNING.
        """
        if not new_factors:
            return

        # Escalar alpha por confidence (mín 0.15 para que el EMA siempre avance)
        confidence = float(new_factors.get('model_confidence', 0.5))
        alpha = max(0.15, 0.6 * confidence) if confidence > 0 else 0.15

        def _clamped(key, current, cap):
            raw = new_factors.get(key, current)
            if raw > cap:
                log.warning(
                    f"  [FIX B] {key} crudo={raw:.3f} excede cap={cap} → clampeado. "
                    f"Posible malla FEM 3D demasiado gruesa o material mal calibrado."
                )
                raw = cap
            return alpha * raw + (1 - alpha) * current

        self.k_correction          = _clamped('k_correction',          self.k_correction,          K_CORRECTION_MAX)
        self.deflection_correction = _clamped('deflection_correction',  self.deflection_correction,  DEFLECTION_CORRECTION_MAX)
        self.sigma_correction      = _clamped('sigma_correction',       self.sigma_correction,       SIGMA_CORRECTION_MAX)
        self.Kt_3d                 = _clamped('Kt_3d',                  self.Kt_3d,                  KT_3D_MAX)
        self.model_confidence      = confidence

        self.n_refreshes += 1
        self.last_refresh_gen = generation
        self.refresh_history.append({
            'gen':        generation,
            'factors':    new_factors,
            'alpha_used': alpha,
            'confidence': self.model_confidence,
        })

        log.info(f"  Surrogate actualizado (gen {generation}, alpha={alpha:.2f}):")
        log.info(f"    k_corr={self.k_correction:.3f}  "
                 f"δ_corr={self.deflection_correction:.3f}  "
                 f"σ_corr={self.sigma_correction:.3f}  "
                 f"Kt={self.Kt_3d:.3f}  "
                 f"confianza={self.model_confidence:.2f}")

    def apply(self, fea2d_result: Dict) -> Dict:
        """
        Aplica los factores de corrección al resultado del FEA 2D.
        Devuelve una copia corregida.
        """
        if self.n_refreshes == 0:
            return fea2d_result  # sin corrección todavía

        corrected = fea2d_result.copy()

        # FIX: usar 'in' antes de *= para evitar KeyError cuando el dict
        # viene del GNN surrogate y no tiene todas las keys del FEA 2D.
        if 'stiffness_kN_m' in corrected:
            corrected['stiffness_kN_m']         *= self.k_correction
        if 'lateral_stiffness_kN_m' in corrected:
            corrected['lateral_stiffness_kN_m'] *= self.k_correction
        if 'max_deflection_mm' in corrected:
            corrected['max_deflection_mm']       *= self.deflection_correction
        if 'sigma_vm_max_MPa' in corrected:
            corrected['sigma_vm_max_MPa']        *= self.sigma_correction

        # Recalcular factor de seguridad con el sigma corregido
        sigma_y = corrected.get('sigma_yield_MPa', 25.0)
        sigma_vm_corr = corrected.get('sigma_vm_max_MPa', 1.0)
        if sigma_vm_corr > 0:
            corrected['safety_factor'] = sigma_y / sigma_vm_corr

        # Incorporar Kt en la vida a fatiga (Kt reduce la vida)
        if self.Kt_3d > 1.0 and corrected.get('life_cycles'):
            corrected['life_cycles'] /= self.Kt_3d ** 2
            # FIX A: 'life_km' no siempre existe (p.ej. cuando el dict viene del
            # GNN surrogate). Antes esto lanzaba KeyError silenciado por el caller.
            if 'life_km' in corrected:
                corrected['life_km'] /= self.Kt_3d ** 2

        corrected['surrogate_corrected'] = True
        corrected['model_confidence']    = self.model_confidence

        return corrected


# Instancia global compartida entre el problema y el callback
_surrogate_state = SurrogateState()


# ---------------------------------------------------------------------------
# Problema de optimización con surrogate corregible
# ---------------------------------------------------------------------------
class SpringTireOptimizationProblem(Problem):
    """
    Problema NSGA-III con FEA 2D como surrogate.
    Los resultados se corrigen con los factores del SurrogateState global.
    """

    def __init__(self, n_obj: int = 10):
        super().__init__(
            n_var=N_GENES,
            n_obj=n_obj,
            n_constr=4,
            xl=np.zeros(N_GENES),
            xu=np.ones(N_GENES)
        )
        self.n_obj = n_obj

        # Catálogo completo de objetivos Spring Tire.
        # GRUPO 1 - rigidez estructural (maximizar)
        #   stiffness_kN_m            → carga vertical principal (Marte)
        #   lateral_stiffness_kN_m    → estabilidad lateral en terreno irregular
        #   torsional_stiffness_Nm_rad→ transmisión de torque del motor
        # GRUPO 2 - resistencia y fatiga (maximizar)
        #   safety_factor             → margen sobre fluencia
        #   life_cycles               → ciclos hasta falla; TPU gana 10-100x sobre PLA
        # GRUPO 3 - memoria de forma / elasticidad (maximizar)
        #   recovery_ratio            → fracción recuperada tras deformación; TPU/Nylon ~0.9, PLA ~0.05
        #   energy_stored_mJ          → energía almacenada en impacto; amortiguación marciana
        # GRUPO 4 - masa y deflexión (minimizar)
        #   mass_kg                   → peso crítico en misiones espaciales
        #   max_deflection_mm         → controlabilidad bajo carga
        # GRUPO 5 - fabricación (minimizar)
        #   print_complexity          → penaliza mezclas difíciles de imprimir
        #
        # Con --n-obj 6 toma los 6 primeros (rigidez+resistencia+masa).
        self.ALL_OBJECTIVES = [
            'stiffness_kN_m',
            'lateral_stiffness_kN_m',
            'torsional_stiffness_Nm_rad',
            'safety_factor',
            'life_cycles',
            'recovery_ratio',
            'energy_stored_mJ',
            'mass_kg',
            'max_deflection_mm',
            'print_complexity',
        ]
        self.objectives = self.ALL_OBJECTIVES[:n_obj]

        # 1=maximizar, -1=minimizar (pymoo minimiza → se pasa -value*direction)
        self.directions = {
            'stiffness_kN_m':             1,
            'lateral_stiffness_kN_m':     1,
            'torsional_stiffness_Nm_rad': 1,
            'safety_factor':              1,
            'life_cycles':                1,
            'recovery_ratio':             1,
            'energy_stored_mJ':           1,
            'mass_kg':                   -1,
            'max_deflection_mm':         -1,
            'print_complexity':          -1,
        }

        self.n_evaluations = 0

    def _evaluate(self, x, out, *args, **kwargs):
        n = len(x)
        f = np.zeros((n, self.n_obj))
        g = np.zeros((n, self.n_constr))

        for i, genome in enumerate(x):
            is_valid, _ = validate_genome(genome)
            if not is_valid:
                f[i, :] = 1e6
                g[i, 0] = 1.0
                g[i, 1] = 1.0
                continue

            params = decode_genome(genome)

            # Restricciones geométricas
            g[i, 0] = max(0, params['d_rim'] - params['d_ext'] * 0.9)
            g[i, 1] = max(0, 6 - params['n_radios'])
            # Restricción masa máxima: penaliza llantas > 5kg (artefactos como #11 de 38kg)
            # g >= 0 significa violación en pymoo; aquí: mass - 5.0 > 0 → infeasible
            # Se evalúa después del FEA para tener mass_kg real
            # (se rellena abajo tras correr FEA)
            g[i, 2] = 0.0  # placeholder masa
            g[i, 3] = 0.0  # placeholder SF

            # FEA 2D rápido (base + fuente de fea_vec para el GNN)
            fea2d = run_spring_tire_fea(params)

            # Aplicar surrogate GNN si está disponible (más preciso que FEA 2D puro)
            if _gnn_predictor is not None and _gnn_predictor.available:
                fea = _gnn_predictor.predict(genome, params, fea2d)
            else:
                fea = fea2d

            # Aplicar corrección del surrogate FEM 3D (si hay refreshes activos)
            fea = _surrogate_state.apply(fea)

            # Restricciones post-FEA
            mass = fea.get('mass_kg', 0.0)
            sf   = fea.get('safety_factor', 0.0)
            g[i, 2] = max(0.0, mass - 5.0)    # masa máxima 5 kg
            g[i, 3] = max(0.0, 1.5 - sf)      # SF mínimo 1.5

            self.n_evaluations += 1

            for j, obj_name in enumerate(self.objectives):
                value = fea.get(obj_name, 0.0)
                # life_cycles puede ser O(10^6) — normalizar con log1p para que
                # el espacio de objetivos esté en escala similar a los demás
                if obj_name == 'life_cycles':
                    value = float(np.log1p(max(0.0, value)))
                # energy_stored_mJ también puede tener rango muy amplio
                if obj_name == 'energy_stored_mJ':
                    value = float(np.log1p(max(0.0, value)))
                direction = self.directions.get(obj_name, 1)
                f[i, j] = -value * direction

        out["F"] = f
        out["G"] = g


# ---------------------------------------------------------------------------
# Callback de Surrogate Refresh
# ---------------------------------------------------------------------------
class SurrogateRefreshCallback(Callback):
    """
    Se ejecuta al final de cada generación.

    Lógica:
      - Cada `refresh_every` generaciones:
          1. Selecciona top_k soluciones del frente de Pareto actual
          2. Las evalúa con FEM 3D
          3. Calcula factores de corrección promedio
          4. Actualiza el SurrogateState global
    """

    def __init__(self,
                 refresh_every: int = 15,
                 top_k: int = 5,
                 fem3d_mesh_size: float = None,
                 output_dir: str = 'results/pareto',
                 checkpoint_every: int = 25,
                 n_generations: int = 200):

        super().__init__()
        self.refresh_every    = refresh_every
        self.top_k            = top_k
        self.fem3d_mesh_size  = fem3d_mesh_size
        self.output_dir       = output_dir
        self.checkpoint_every = checkpoint_every
        self.n_generations    = n_generations

        # Estadísticas para el reporte final
        self.refresh_log: List[Dict] = []
        self.gen_times:   List[float] = []
        # FIX: inicializar en None; se setea en la primera llamada a notify
        # para no inflar el tiempo de la gen 1 con el tiempo de setUp
        self._t_gen_start = None

    # ------------------------------------------------------------------
    def notify(self, algorithm):
        """Llamado al final de cada generación."""
        gen = algorithm.n_gen
        now = time.time()

        # FIX: primer arrange no infla el tiempo
        if self._t_gen_start is None:
            self._t_gen_start = now
            elapsed = 0.0
        else:
            elapsed = now - self._t_gen_start
        self.gen_times.append(elapsed)
        self._t_gen_start = now

        # ETA estimado basado en tiempo promedio de gen
        if len(self.gen_times) > 1:
            avg_gen_t    = float(np.mean(self.gen_times[-10:]))  # últimas 10 gens
            gens_left    = self.n_generations - gen
            eta_s        = avg_gen_t * gens_left
            eta_str      = (f"{int(eta_s//60)}m{int(eta_s%60)}s"
                            if eta_s >= 60 else f"{eta_s:.0f}s")
        else:
            eta_str = "?"

        # Log de progreso cada 5 generaciones
        if gen % 5 == 0:
            pop = algorithm.pop
            F   = pop.get("F")
            best = -F.min(axis=0)
            n_pareto = len(algorithm.opt) if algorithm.opt is not None else '?'
            obj_list = getattr(algorithm.problem, 'objectives', [])
            def _bv(name):
                if name in obj_list:
                    idx = obj_list.index(name)
                    return best[idx] if idx < len(best) else None
                return None
            parts = [f"Gen {gen:4d}/{self.n_generations}"]
            k  = _bv('stiffness_kN_m');    parts.append(f"k={k:.0f}kN/m") if k is not None else None
            sf = _bv('safety_factor');     parts.append(f"SF={sf:.2f}") if sf is not None else None
            lc = _bv('life_cycles');       parts.append(f"life=e^{lc:.1f}") if lc is not None else None
            rc = _bv('recovery_ratio');    parts.append(f"rec={rc:.2f}") if rc is not None else None
            m  = _bv('mass_kg');           parts.append(f"m={-m:.2f}kg") if m is not None else None
            parts.append(f"Pareto={n_pareto} | t={elapsed:.2f}s | ETA={eta_str}")
            log.info("  " + " | ".join(parts))

        # ---- Checkpoint periódico del Pareto ----
        if gen > 0 and gen % self.checkpoint_every == 0:
            self._save_checkpoint(algorithm, gen)

        # ---- Surrogate Refresh ----
        if (FEM3D_AVAILABLE
                and gen > 0
                and gen % self.refresh_every == 0):

            self._run_fem3d_refresh(algorithm, gen)

    # ------------------------------------------------------------------
    def _run_fem3d_refresh(self, algorithm, gen: int):
        """
        Núcleo del refresh:
          1. Obtener frente de Pareto actual
          2. Seleccionar top_k candidatos (mayor diversidad + mejor objetivo 0)
          3. Correr FEM 3D en cada uno
          4. Promediar factores de corrección
          5. Actualizar surrogate global
        """
        log.info(f"\n{'='*55}")
        log.info(f"  SURROGATE REFRESH — Generación {gen}")
        log.info(f"{'='*55}")

        pop = algorithm.pop
        F   = pop.get("F")
        X   = pop.get("X")

        # --- Obtener frente de Pareto actual ---
        nds    = NonDominatedSorting()
        fronts = nds.do(F, n_stop_if_ranked=self.top_k * 3)
        pareto_idx = fronts[0]

        if len(pareto_idx) == 0:
            log.warning("  Frente de Pareto vacío, saltando refresh")
            return

        # --- Seleccionar top_k con máxima diversidad ---
        # Estrategia: ordenar por objetivo principal (stiffness, negado → col 0)
        # y tomar muestras espaciadas del frente
        pareto_F = F[pareto_idx]
        pareto_X = X[pareto_idx]

        # Ordenar el frente por objetivo 0 (stiffness descendente)
        sort_order = np.argsort(pareto_F[:, 0])   # menor F[0] = mayor stiffness
        n_select = min(self.top_k, len(sort_order))

        # Espaciado uniforme a lo largo del frente ordenado
        if n_select >= len(sort_order):
            selected = sort_order
        else:
            indices = np.round(np.linspace(0, len(sort_order)-1, n_select)).astype(int)
            selected = sort_order[indices]

        log.info(f"  Evaluando {len(selected)} soluciones con FEM 3D...")

        # --- Correr FEM 3D en cada candidato ---
        all_factors = []
        t0_refresh  = time.time()

        for rank, idx in enumerate(selected):
            genome = pareto_X[idx]
            is_valid, _ = validate_genome(genome)
            if not is_valid:
                continue

            params   = decode_genome(genome)
            fea2d    = run_spring_tire_fea(params)

            t0_fem = time.time()
            fem3d  = run_fem3d_spoke(
                params,
                F_vertical=500.0,
                mesh_size=self.fem3d_mesh_size,
            )
            t_fem = time.time() - t0_fem

            if not fem3d:
                log.warning(f"    [{rank+1}/{len(selected)}] FEM 3D falló, saltando")
                continue

            factors = compute_correction_factors(fea2d, fem3d)

            log.info(
                f"    [{rank+1}/{len(selected)}] "
                f"k_corr={factors.get('k_correction', 1):.3f}  "
                f"σ_corr={factors.get('sigma_correction', 1):.3f}  "
                f"Kt={factors.get('Kt_3d', 1):.2f}  "
                f"conf={factors.get('model_confidence', 0):.2f}  "
                f"t={t_fem:.1f}s"
            )

            if factors:
                all_factors.append(factors)

        # --- Promediar factores y actualizar surrogate ---
        if all_factors:
            avg_factors = {
                'k_correction':          float(np.mean([f.get('k_correction', 1)          for f in all_factors])),
                'deflection_correction': float(np.mean([f.get('deflection_correction', 1) for f in all_factors])),
                'sigma_correction':      float(np.mean([f.get('sigma_correction', 1)      for f in all_factors])),
                'Kt_3d':                 float(np.mean([f.get('Kt_3d', 1)                 for f in all_factors])),
                'model_confidence':      float(np.mean([f.get('model_confidence', 0.5)    for f in all_factors])),
            }

            _surrogate_state.update(avg_factors, gen)

            self.refresh_log.append({
                'gen':           gen,
                'n_evaluated':   len(all_factors),
                'avg_factors':   avg_factors,
                't_total_s':     time.time() - t0_refresh,
            })

        else:
            log.warning("  No se obtuvieron factores válidos del FEM 3D")

        log.info(f"  Refresh completado en {time.time()-t0_refresh:.1f}s\n")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, algorithm, gen: int):
        """Guarda checkpoint del frente de Pareto actual en disco."""
        try:
            opt = algorithm.opt
            if opt is None or len(opt) == 0:
                return
            X = opt.get("X")
            F = opt.get("F")
            ckpt = {
                'gen':           gen,
                'n_gen_total':   self.n_generations,
                'n_pareto':      len(X),
                'is_checkpoint': True,
                'k_correction':  _surrogate_state.k_correction,
                'sigma_correction': _surrogate_state.sigma_correction,
                'pareto_X':      X.tolist(),
                'pareto_F':      F.tolist(),
            }
            ckpt_path = os.path.join(self.output_dir, 'pareto_checkpoint.json')
            with open(ckpt_path, 'w') as fh:
                json.dump(ckpt, fh)
            log.info(f"  💾 Checkpoint gen {gen}/{self.n_generations}: "
                     f"{len(X)} soluciones Pareto → {ckpt_path}")
        except Exception as ex:
            log.warning(f"  ⚠ Checkpoint fallido en gen {gen}: {ex}")

    # ------------------------------------------------------------------
    def get_summary(self) -> Dict:
        """Resumen del proceso de refresh para incluir en los resultados."""
        return {
            'n_refreshes':       _surrogate_state.n_refreshes,
            'final_k_corr':      _surrogate_state.k_correction,
            'final_delta_corr':  _surrogate_state.deflection_correction,
            'final_sigma_corr':  _surrogate_state.sigma_correction,
            'final_Kt':          _surrogate_state.Kt_3d,
            'final_confidence':  _surrogate_state.model_confidence,
            'refresh_log':       self.refresh_log,
            'avg_gen_time_s':    float(np.mean(self.gen_times)) if self.gen_times else 0.0,
        }


# ---------------------------------------------------------------------------
# Validación final con FEM 3D (post-optimización)
# ---------------------------------------------------------------------------
def validate_pareto_with_fem3d(pareto_solutions: List[Dict],
                                max_validate: int = 20,
                                mesh_size: float = None) -> List[Dict]:
    """
    Valida las soluciones del frente de Pareto con FEM 3D al final.

    Estrategia: evalúa las `max_validate` mejores soluciones
    (las que tienen mayor stiffness o menor masa).

    Args:
        pareto_solutions: lista de dicts con 'genome', 'params', 'objectives'
        max_validate: cuántas soluciones validar con FEM 3D
        mesh_size: tamaño de malla (None = automático)

    Returns:
        Lista con campo 'fem3d_validation' añadido
    """
    if not FEM3D_AVAILABLE:
        log.warning("FEM 3D no disponible, saltando validación final")
        return pareto_solutions

    log.info(f"\n{'='*55}")
    log.info(f"  VALIDACIÓN FINAL FEM 3D ({min(max_validate, len(pareto_solutions))} soluciones)")
    log.info(f"{'='*55}")

    # Ordenar por stiffness descendente para validar las mejores primero
    sorted_sols = sorted(
        pareto_solutions,
        key=lambda s: s['objectives'].get('stiffness_kN_m', 0),
        reverse=True
    )

    n_validate = min(max_validate, len(sorted_sols))
    validate_ids = {s['id'] for s in sorted_sols[:n_validate]}

    t0_total = time.time()
    n_ok = 0

    for sol in pareto_solutions:
        if sol['id'] not in validate_ids:
            sol['fem3d_validation'] = None
            continue

        params = sol['params']
        t0 = time.time()

        fem3d_result = run_fem3d_spoke(
            params,
            F_vertical=500.0,
            mesh_size=mesh_size,
        )

        t_fem = time.time() - t0

        if fem3d_result:
            # Enriquecer con comparación 2D vs 3D
            fea2d = sol.get('fea_results', {})
            factors = compute_correction_factors(fea2d, fem3d_result) if fea2d else {}

            sol['fem3d_validation'] = {
                # Resultados clave del FEM 3D
                'k_vertical_kNm':    fem3d_result.get('k_vertical_kNm'),
                'max_disp_mm':       fem3d_result.get('max_displacement_mm'),
                'sigma_vm_MPa':      fem3d_result.get('sigma_vm_MPa'),
                'SF_strength':       fem3d_result.get('SF_strength'),
                'plastic_strain_pct': fem3d_result.get('plastic_strain_pct'),
                'Kt_rim':            fem3d_result.get('Kt_rim'),
                'Kt_tread':          fem3d_result.get('Kt_tread'),
                'n_nodes':           fem3d_result.get('n_nodes'),
                'n_elements':        fem3d_result.get('n_elements'),
                # Factores de corrección 2D→3D
                'correction_factors': factors,
                # Meta
                'fem3d_time_s':      t_fem,
                'fea_type':          fem3d_result.get('fea_type'),
            }
            n_ok += 1
            log.info(
                f"  #{sol['id']:3d} ✓ | "
                f"k={fem3d_result.get('k_vertical_kNm', 0):.1f}kN/m | "
                f"σ={fem3d_result.get('sigma_vm_MPa', 0):.2f}MPa | "
                f"SF={fem3d_result.get('SF_strength', 0):.2f} | "
                f"t={t_fem:.1f}s"
            )
        else:
            sol['fem3d_validation'] = {'error': 'FEM 3D falló'}
            log.warning(f"  #{sol['id']:3d} ✗ FEM 3D falló")

    log.info(f"\n  Validados: {n_ok}/{n_validate} soluciones en "
             f"{time.time()-t0_total:.1f}s total")

    return pareto_solutions


# ---------------------------------------------------------------------------
# Función principal de optimización
# ---------------------------------------------------------------------------
def run_nsga3_optimization(
        pop_size:            int   = 100,
        n_generations:       int   = 200,
        n_obj:               int   = 10,
        output_dir:          str   = 'results/pareto',
        seed:                int   = 42,
        # --- Parámetros del Surrogate Refresh ---
        fem3d_refresh_every: int   = 15,
        fem3d_top_k:         int   = 5,
        fem3d_final_validate: int  = 20,
        fem3d_mesh_size:     float = None,
        use_fem3d:           bool  = True,
        checkpoint_every:    int   = 25,
        # FIX Bug #3: exponer models_dir para que el CLI lo pueda pasar
        models_dir:          str   = None,
) -> Dict:
    """
    Ejecuta NSGA-III con Surrogate Refresh de FEM 3D.

    Parámetros de rendimiento (guía):
      - fem3d_refresh_every=15, top_k=5:
          ~25 evaluaciones FEM 3D intermedias (200gen / 15 * 5 ≈ 66)
          ~330s extra de FEM 3D para 200 generaciones
      - fem3d_final_validate=20:
          ~100s adicionales al final
      - Total overhead FEM 3D: ~430s sobre el NSGA-III

    Desactivar FEM 3D (use_fem3d=False):
      Comportamiento idéntico al fase3 original.
    """
    if not PYMOO_AVAILABLE:
        log.error("pymoo no disponible")
        return {}

    # Resetear estado global del surrogate
    global _surrogate_state, _gnn_predictor
    _surrogate_state = SurrogateState()

    # FIX Bug #3: usar models_dir pasado por argumento; fallback al path relativo al script
    _models_dir = models_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results/models'
    )
    # Cargar GNN ensemble si está disponible
    _gnn_predictor = GNNEnsemblePredictor(models_dir=_models_dir)
    if _gnn_predictor.available:
        log.info("  Surrogate GNN: ACTIVO (predicción ensemble de 5 modelos)")
    else:
        log.info("  Surrogate GNN: INACTIVO (usando FEA 2D puro)")

    fem3d_active = use_fem3d and FEM3D_AVAILABLE

    log.info("=" * 65)
    log.info("  FASE 3: NSGA-III + Surrogate Refresh v6.1")
    log.info("=" * 65)
    log.info(f"  Población:          {pop_size}")
    log.info(f"  Generaciones:       {n_generations}")
    log.info(f"  Objetivos:          {n_obj}")
    log.info(f"  FEM 3D activo:      {fem3d_active}")
    if fem3d_active:
        log.info(f"  Refresh cada:       {fem3d_refresh_every} generaciones")
        log.info(f"  Top-K por refresh:  {fem3d_top_k} soluciones")
        log.info(f"  Validación final:   {fem3d_final_validate} soluciones")
    log.info("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Problema ----
    problem = SpringTireOptimizationProblem(n_obj=n_obj)

    # ---- Direcciones de referencia ----
    # n_partitions auto para que ref_dirs ≤ pop_size
    # Das-Dennis: C(n_obj+p-1, p) ≤ pop_size  → buscar p máximo que cumpla
    import math
    def n_ref_dirs(n_obj, p):
        return math.comb(n_obj + p - 1, p)
    p = 1
    while n_ref_dirs(n_obj, p + 1) <= pop_size:
        p += 1
    log.info(f"  ref_dirs: n_partitions={p} → {n_ref_dirs(n_obj, p)} direcciones (pop={pop_size})")
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)

    # ---- Algoritmo ----
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        seed=seed,
    )

    # ---- Callback de refresh (activo siempre para checkpoint + logs) ----
    callback = SurrogateRefreshCallback(
        refresh_every=fem3d_refresh_every if fem3d_active else 999999,
        top_k=fem3d_top_k,
        fem3d_mesh_size=fem3d_mesh_size,
        output_dir=output_dir,
        checkpoint_every=checkpoint_every,
        n_generations=n_generations,
    )

    # ---- Optimizar ----
    log.info("\nIniciando optimización...\n")
    t0 = time.time()

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=seed,
        callback=callback,
        verbose=False,
    )

    t_nsga = time.time() - t0
    log.info(f"\nNSGA-III completado en {t_nsga:.1f}s")
    log.info(f"Soluciones en frente de Pareto: {len(res.X)}")
    log.info(f"Evaluaciones FEA 2D totales:    {problem.n_evaluations}")

    # ---- Procesar frente de Pareto ----
    pareto_solutions = []
    for i, (genome, objectives) in enumerate(zip(res.X, res.F)):
        params     = decode_genome(genome)
        fea_result = run_spring_tire_fea(params)
        fea_corr   = _surrogate_state.apply(fea_result)

        pareto_solutions.append({
            'id':         i,
            'genome':     genome.tolist(),
            'params':     params,
            'fea_results': fea_corr,
            'objectives': {
                # Grupo 1: rigidez
                'stiffness_kN_m':             fea_corr.get('stiffness_kN_m', 0),
                'lateral_stiffness_kN_m':     fea_corr.get('lateral_stiffness_kN_m', 0),
                'torsional_stiffness_Nm_rad': fea_corr.get('torsional_stiffness_Nm_rad', 0),
                # Grupo 2: resistencia y fatiga
                'safety_factor':              fea_corr.get('safety_factor', 0),
                'life_cycles':                fea_corr.get('life_cycles', 0),
                # Grupo 3: memoria de forma
                'recovery_ratio':             fea_corr.get('recovery_ratio', 0),
                'energy_stored_mJ':           fea_corr.get('energy_stored_mJ', 0),
                # Grupo 4: masa y deflexión
                'mass_kg':                    fea_corr.get('mass_kg', 0),
                'max_deflection_mm':          fea_corr.get('max_deflection_mm', 0),
                # Grupo 5: fabricación
                'print_complexity':           fea_corr.get('print_complexity', 0),
                # Extras informativos (no son objetivos pero útiles para análisis)
                'adhesion_risk':              fea_corr.get('adhesion_risk', 0),
                'goodman_ratio':              fea_corr.get('goodman_ratio', 0),
                'structural_efficiency':      fea_corr.get('structural_efficiency', 0),
                'plastic_strain_percent':     fea_corr.get('plastic_strain_percent', 0),
                'shape_memory_ratio':         fea_corr.get('shape_memory_ratio', 0),
            },
            'fem3d_validation': None,  # se rellena abajo
        })

    # ---- Validación final con FEM 3D ----
    if fem3d_active and fem3d_final_validate > 0:
        pareto_solutions = validate_pareto_with_fem3d(
            pareto_solutions,
            max_validate=fem3d_final_validate,
            mesh_size=fem3d_mesh_size,
        )

    # ---- Guardar resultados ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    surrogate_summary = callback.get_summary() if callback else {}

    results = {
        'version':       '6.1',
        'timestamp':     timestamp,
        'n_solutions':   len(pareto_solutions),
        'pop_size':      pop_size,
        'n_generations': n_generations,
        'n_obj':         n_obj,
        'n_fea2d_evals': problem.n_evaluations,
        't_nsga_s':      t_nsga,
        'fem3d_active':  fem3d_active,
        'surrogate':     surrogate_summary,
        'genes': [
            'd_ext_ratio', 'd_rim_ratio', 'width_ratio', 'n_radios_ratio',
            'wall_thickness_ratio', 'angle_cross_ratio',
            'material_sector_0', 'material_sector_1', 'material_sector_2',
            'material_sector_3', 'material_sector_4', 'material_sector_5',
            'material_sector_6', 'material_sector_7',
            'n_layers_ratio', 'spoke_thickness_ratio', 'hub_angle_cross_ratio', 'tread_rim_factor_ratio',
        ],
        'soluciones': pareto_solutions,
    }

    output_path = os.path.join(output_dir, f'pareto_spring_tire_{timestamp}.json')
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return super().default(obj)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    # ---- Resumen ----
    log.info("\n" + "=" * 65)
    log.info("  RESUMEN FINAL")
    log.info("=" * 65)
    log.info(f"  Soluciones Pareto:       {len(pareto_solutions)}")
    log.info(f"  Evaluaciones FEA 2D:     {problem.n_evaluations}")
    if fem3d_active:
        log.info(f"  Refreshes FEM 3D:        {_surrogate_state.n_refreshes}")
        log.info(f"  Confianza surrogate:     {_surrogate_state.model_confidence:.2f}")
        log.info(f"  Factor corrección k:     {_surrogate_state.k_correction:.3f}")
        log.info(f"  Factor corrección σ:     {_surrogate_state.sigma_correction:.3f}")

    log.info(f"\n  Estadísticas del frente de Pareto:")
    for obj_name in ['stiffness_kN_m', 'safety_factor', 'life_cycles',
                     'recovery_ratio', 'mass_kg', 'max_deflection_mm']:
        values = [s['objectives'].get(obj_name) for s in pareto_solutions]
        values = [v for v in values if v is not None and np.isfinite(float(v))]
        if not values:
            log.info(f"    {obj_name}: sin datos válidos")
            continue
        log.info(f"    {obj_name}: "
                 f"min={min(values):.3f}, max={max(values):.3f}, "
                 f"mean={np.mean(values):.3f}")

    log.info(f"\n  Archivo: {output_path}")
    log.info("=" * 65)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='NSGA-III Spring Tire con Surrogate Refresh FEM 3D'
    )
    parser.add_argument('--pop-size',      type=int,   default=100)
    parser.add_argument('--n-gen',         type=int,   default=200)
    parser.add_argument('--n-obj',         type=int,   default=10,
                        help='Número de objetivos (6=rigidez+resist+masa, 8=+memoria_forma, 10=todos)')
    parser.add_argument('--output-dir',    type=str,   default='results/pareto')
    parser.add_argument('--models-dir',    type=str,   default=None,
                        help='Directorio con modelos GNN de fase2 (default: results/models junto al script)')
    parser.add_argument('--seed',          type=int,   default=42)

    # FEM 3D
    parser.add_argument('--no-fem3d',      action='store_true',
                        help='Desactivar FEM 3D (modo legacy)')
    parser.add_argument('--refresh-every', type=int,   default=15,
                        help='Generaciones entre refreshes FEM 3D')
    parser.add_argument('--top-k',         type=int,   default=5,
                        help='Soluciones Pareto a evaluar por refresh')
    parser.add_argument('--final-validate',type=int,   default=20,
                        help='Soluciones a validar con FEM 3D al final')
    parser.add_argument('--mesh-size',     type=float, default=None,
                        help='Tamaño de elemento FEM 3D en mm (None=auto)')
    parser.add_argument('--checkpoint-every', type=int, default=25,
                        help='Guardar checkpoint Pareto cada N generaciones (default: 25)')

    args = parser.parse_args()

    run_nsga3_optimization(
        pop_size=args.pop_size,
        n_generations=args.n_gen,
        n_obj=args.n_obj,
        output_dir=args.output_dir,
        seed=args.seed,
        fem3d_refresh_every=args.refresh_every,
        fem3d_top_k=args.top_k,
        fem3d_final_validate=args.final_validate,
        fem3d_mesh_size=args.mesh_size,
        use_fem3d=not args.no_fem3d,
        checkpoint_every=args.checkpoint_every,
        models_dir=args.models_dir,  # FIX Bug #3
    )


if __name__ == '__main__':
    main()
