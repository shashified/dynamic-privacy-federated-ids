"""
DynamicPrivacyIDS: Federated Learning IDS with Dynamic Differential Privacy
=============================================================================
Based on: https://github.com/nsol-nmsu/FML-Network
Modification: Added ThreatAwarePrivacyScheduler that adjusts DP noise budget
              dynamically based on real-time attack detection rate.

Run: python dynamic_fl_ids.py
"""

import os, warnings, logging, json, csv
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import ray
ray.init(local_mode=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEED         = 42
NUM_CLIENTS  = 3
NUM_ROUNDS   = 40          # attack injected at round 20
ATTACK_ROUND = 20
LOCAL_EPOCHS = 2
BATCH_SIZE   = 256
LR           = 1e-3
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# 1. THREAT-AWARE PRIVACY SCHEDULER  ← KEY CONTRIBUTION
# ─────────────────────────────────────────────────────────────
class ThreatAwarePrivacyScheduler:
    """
    Dynamically adjusts the differential-privacy noise multiplier (ε proxy)
    based on the observed local attack rate.

    Logic:
      • High attack rate  → loosen privacy (ε → ε_max) for better detection
      • Low  attack rate  → tighten privacy (ε → ε_min) for stronger protection

    In practice we simulate ε ↔ noise_multiplier inversely:
        noise_multiplier = BASE_NOISE / eps
    so higher ε ⟹ less noise ⟹ better utility under attack.
    """

    def __init__(self, eps_min: float = 0.5, eps_max: float = 2.0, alpha: float = 0.2):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.alpha   = alpha
        self.eps     = 1.0                     # start at midpoint
        self.attack_history = deque(maxlen=10)

    def update(self, local_attack_rate: float) -> float:
        """Update ε given current batch's attack rate. Returns new ε."""
        self.attack_history.append(local_attack_rate)

        recent = list(self.attack_history)[-5:]
        recent_attack_rate = float(np.mean(recent)) if recent else 0.0

        if recent_attack_rate > 0.1:          # attack threshold
            self.eps = min(self.eps_max, self.eps + self.alpha)
        else:
            self.eps = max(self.eps_min, self.eps - self.alpha)

        return self.eps

    @property
    def noise_multiplier(self) -> float:
        """Convert ε to Gaussian noise multiplier (inversely proportional)."""
        BASE_NOISE = 1.0
        return BASE_NOISE / self.eps


# ─────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATASET  (CICIDS2017-like)
# ─────────────────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 30_000, attack_fraction: float = 0.3,
                             seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic network traffic dataset resembling CICIDS2017.
    Features: flow duration, packet lengths, IAT stats, flag counts, etc.
    Labels: 0 = BENIGN, 1 = ATTACK
    """
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * attack_fraction)
    n_benign = n_samples - n_attack

    def make_flow(n, is_attack):
        base = 1.5 if is_attack else 1.0
        return {
            "flow_duration":        rng.exponential(base * 1e6, n),
            "tot_fwd_pkts":         rng.poisson(base * 20, n).astype(float),
            "tot_bwd_pkts":         rng.poisson(base * 15, n).astype(float),
            "totlen_fwd_pkts":      rng.exponential(base * 5000, n),
            "totlen_bwd_pkts":      rng.exponential(base * 4000, n),
            "fwd_pkt_len_mean":     rng.normal(base * 200, 50, n),
            "bwd_pkt_len_mean":     rng.normal(base * 180, 45, n),
            "flow_byts_s":          rng.exponential(base * 1e5, n),
            "flow_pkts_s":          rng.exponential(base * 500, n),
            "flow_iat_mean":        rng.exponential(base * 1e4, n),
            "flow_iat_std":         rng.exponential(base * 5000, n),
            "fwd_iat_mean":         rng.exponential(base * 8000, n),
            "bwd_iat_mean":         rng.exponential(base * 8000, n),
            "fwd_header_len":       rng.normal(base * 40, 5, n),
            "bwd_header_len":       rng.normal(base * 38, 5, n),
            "fin_flag_cnt":         rng.binomial(1, 0.4, n).astype(float),
            "syn_flag_cnt":         rng.binomial(1, 0.5 if is_attack else 0.2, n).astype(float),
            "rst_flag_cnt":         rng.binomial(1, 0.2 if is_attack else 0.05, n).astype(float),
            "psh_flag_cnt":         rng.binomial(1, 0.6, n).astype(float),
            "ack_flag_cnt":         rng.binomial(1, 0.8, n).astype(float),
            "init_win_byts_fwd":    rng.integers(0, 65535, n).astype(float),
            "init_win_byts_bwd":    rng.integers(0, 65535, n).astype(float),
            "label":                np.ones(n, dtype=int) if is_attack else np.zeros(n, dtype=int),
        }

    df = pd.concat([
        pd.DataFrame(make_flow(n_benign, False)),
        pd.DataFrame(make_flow(n_attack, True)),
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def load_or_generate_data():
    """Try to load FLNET2023 CSV; fall back to synthetic data."""
    candidates = [
        "data/FLNET2023.csv", "FLNET2023.csv",
        "data/cicids2017.csv", "cicids2017.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            log.info(f"Loading real dataset from {path}")
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            # Standardise label column
            if "label" not in df.columns:
                for col in df.columns:
                    if "label" in col or "class" in col:
                        df.rename(columns={col: "label"}, inplace=True)
                        break
            if df["label"].dtype == object:
                df["label"] = (df["label"].str.upper() != "BENIGN").astype(int)
            return df

    log.info("No real dataset found – generating synthetic CICIDS2017-like data")
    return generate_synthetic_data()


def prepare_tensors(df: pd.DataFrame, attack_injected: bool = False):
    """Return (X_tensor, y_tensor, attack_rate)."""
    drop_cols = [c for c in df.columns if c == "label" or df[c].dtype == object]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Inject extra attacks after round 20 to simulate attack surge
    if attack_injected:
        n_inject = int(len(y) * 0.4)
        idx = np.random.choice(np.where(y == 0)[0], size=n_inject, replace=False)
        y[idx] = 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    attack_rate = float(y.mean())

    return torch.tensor(X), torch.tensor(y), attack_rate


# ─────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────
class IDSNet(nn.Module):
    """Lightweight MLP IDS classifier."""

    def __init__(self, input_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 4. FLOWER CLIENT
# ─────────────────────────────────────────────────────────────
class DynamicPrivacyClient(fl.client.NumPyClient):
    """
    Flower client with ThreatAwarePrivacyScheduler.

    Key additions vs base repo:
      • privacy_scheduler tracks attack rate → adjusts ε
      • noise_multiplier applied to gradient clipping + Gaussian noise (DP-SGD sim)
      • Reports current_eps back to server in metrics
    """

    def __init__(self, cid: int, X: torch.Tensor, y: torch.Tensor,
                 attack_rate: float, mode: str = "dynamic"):
        self.cid          = cid
        self.X            = X
        self.y            = y
        self.attack_rate  = attack_rate
        self.mode         = mode           # "fixed_1.0" | "fixed_0.5" | "dynamic"

        self.model        = IDSNet(X.shape[1])
        self.privacy_sched = ThreatAwarePrivacyScheduler()
        self._set_eps_for_mode()

    def _set_eps_for_mode(self):
        if self.mode == "fixed_0.5":
            self.privacy_sched.eps = 0.5
        elif self.mode == "fixed_1.0":
            self.privacy_sched.eps = 1.0
        # "dynamic" starts at 1.0 (default)

    # ── Flower API ──────────────────────────────────────────

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        state = self.model.state_dict()
        for k, v, p in zip(state.keys(), params, self.model.parameters()):
            p.data = torch.tensor(v)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # ── Compute attack rate for this round ──────────────
        local_attack_rate = float(self.y.float().mean())

        # ── Update privacy budget ────────────────────────────
        if self.mode == "dynamic":
            current_eps = self.privacy_sched.update(local_attack_rate)
        else:
            current_eps = self.privacy_sched.eps   # fixed

        noise_mult = self.privacy_sched.noise_multiplier
        log.info(f"  Client {self.cid} | mode={self.mode} | attack_rate={local_attack_rate:.3f}"
                 f" | ε={current_eps:.3f} | noise_mult={noise_mult:.3f}")

        # ── Training ─────────────────────────────────────────
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()

                # ── DP-SGD: clip gradients + add Gaussian noise ──
                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * (noise_mult * MAX_GRAD_NORM)
                            param.grad += noise

                optimizer.step()

        return (
            self.get_parameters(config={}),
            len(self.X),
            {"eps": float(current_eps), "attack_rate": float(local_attack_rate)},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            preds  = logits.argmax(dim=1).numpy()
            labels = self.y.numpy()

        f1   = f1_score(labels, preds, average="macro", zero_division=0)
        acc  = float((preds == labels).mean())
        loss = float(nn.CrossEntropyLoss()(logits, self.y).item())

        return loss, len(self.X), {"f1": f1, "accuracy": acc}


# ─────────────────────────────────────────────────────────────
# 5. FLOWER SERVER STRATEGY  (FedAvg + logging)
# ─────────────────────────────────────────────────────────────
class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that logs F1, ε, attack_rate per round to a CSV."""

    def __init__(self, run_id: str, **kwargs):
        super().__init__(**kwargs)
        self.run_id   = run_id
        self.csv_path = RESULTS_DIR / f"metrics_{run_id}.csv"
        self.round_no = 0
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "mode", "f1", "accuracy", "mean_eps",
                             "mean_attack_rate", "attack_phase"])

    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        self.round_no = server_round

        if results:
            eps_vals    = [r.metrics.get("eps", 1.0)          for _, r in results]
            atk_vals    = [r.metrics.get("attack_rate", 0.0)  for _, r in results]
            mean_eps    = float(np.mean(eps_vals))
            mean_atk    = float(np.mean(atk_vals))
            attack_phase = int(server_round > ATTACK_ROUND)
            log.info(f"  [Server R{server_round}] mean_ε={mean_eps:.3f} | "
                     f"mean_attack_rate={mean_atk:.3f} | attack_phase={attack_phase}")
        return agg

    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)

        if results:
            f1_vals  = [r.metrics.get("f1", 0.0)       for _, r in results]
            acc_vals = [r.metrics.get("accuracy", 0.0) for _, r in results]
            mean_f1  = float(np.mean(f1_vals))
            mean_acc = float(np.mean(acc_vals))
            attack_phase = int(server_round > ATTACK_ROUND)

            # Grab ε from last fit round (approximation)
            mode_tag = self.run_id.split("_")[0]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    server_round, mode_tag, mean_f1, mean_acc,
                    None, None, attack_phase
                ])

        return agg


# ─────────────────────────────────────────────────────────────
# 6. RUNNER
# ─────────────────────────────────────────────────────────────
def run_simulation(mode: str = "dynamic", seed: int = SEED, num_rounds: int = NUM_ROUNDS):
    """Run a full FL simulation for one mode and seed."""
    log.info(f"\n{'='*60}")
    log.info(f"  Running simulation | mode={mode} | seed={seed}")
    log.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    df   = load_or_generate_data()
    run_id = f"{mode}_seed{seed}"

    # Split data across clients
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    client_dfs = [df.iloc[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

    all_metrics = []   # collect per-round metrics

    def client_fn(context: fl.common.Context):
        cid = int(context.node_id) % NUM_CLIENTS
        c_df = client_dfs[cid]

        # After round ATTACK_ROUND, inject surge attacks
        current_round = context.run_config.get("round", 0)
        attack_injected = (current_round > ATTACK_ROUND)

        X, y, atk_rate = prepare_tensors(c_df, attack_injected=attack_injected)
        return DynamicPrivacyClient(cid, X, y, atk_rate, mode=mode).to_client()

    # Use a closure strategy to capture metrics
    per_round_eps    = []
    per_round_atk    = []
    per_round_f1     = []

    class TrackingStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            agg = super().aggregate_fit(server_round, results, failures)
            if results:
                eps_vals = [r.metrics.get("eps", 1.0)        for _, r in results]
                atk_vals = [r.metrics.get("attack_rate", 0.) for _, r in results]
                per_round_eps.append(float(np.mean(eps_vals)))
                per_round_atk.append(float(np.mean(atk_vals)))
            return agg

        def aggregate_evaluate(self, server_round, results, failures):
            agg = super().aggregate_evaluate(server_round, results, failures)
            if results:
                f1_vals = [r.metrics.get("f1", 0.) for _, r in results]
                per_round_f1.append(float(np.mean(f1_vals)))
            return agg

    strategy = TrackingStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # ── In-process simulation (no sockets needed) ────────────
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 0.5, "num_gpus": 0},
    )

    # Pad shorter lists
    n = num_rounds
    per_round_f1  = per_round_f1[:n]  + [per_round_f1[-1]]  * max(0, n - len(per_round_f1))
    per_round_eps = per_round_eps[:n] + [per_round_eps[-1]] * max(0, n - len(per_round_eps))
    per_round_atk = per_round_atk[:n] + [per_round_atk[-1]] * max(0, n - len(per_round_atk))

    result_df = pd.DataFrame({
        "round":       range(1, n + 1),
        "mode":        mode,
        "seed":        seed,
        "f1":          per_round_f1,
        "mean_eps":    per_round_eps,
        "attack_rate": per_round_atk,
        "attack_phase": [int(r > ATTACK_ROUND) for r in range(1, n + 1)],
    })

    csv_path = RESULTS_DIR / f"metrics_{run_id}.csv"
    result_df.to_csv(csv_path, index=False)
    log.info(f"  Saved metrics to {csv_path}")
    return result_df


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEEDS = [42, 123, 999]
    MODES = ["fixed_1.0", "fixed_0.5", "dynamic"]

    all_results = []

    for seed in SEEDS:
        for mode in MODES:
            try:
                df_res = run_simulation(mode=mode, seed=seed, num_rounds=NUM_ROUNDS)
                all_results.append(df_res)
            except Exception as e:
                log.error(f"Simulation failed for mode={mode} seed={seed}: {e}")
                raise

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    log.info(f"\n✅  All simulations complete. Results in {RESULTS_DIR}/")
    log.info("   Run 'jupyter notebook experiments.ipynb' for plots.")
