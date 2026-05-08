# kdflow/opd_local/config.py
from dataclasses import dataclass


@dataclass
class LocalOPDConfig:
    enabled: bool = False

    # candidate detection
    num_full_rollouts: int = 4           # N_f
    num_candidates: int = 3              # K_c
    min_candidate_separation: int = 8    # Δ
    rollback_steps: int = 1              # R, 你后面可以改成 3/5/8

    # cheap probe
    probe_len: int = 20                  # L_p
    threshold_alpha: float = 2.0         # α
    threshold_eps: float = 1e-6

    # local continuation matching
    local_len: int = 20                  # L_c
    teacher_num_samples: int = 4         # N_t
    student_num_samples: int = 4         # N_s, 包含原始 suffix
    temperature: float = 1.0
    top_p: float = 1.0

    # local loss
    distance_type: str = "l1"            # "l1" / "l2"
    local_loss_weight: float = 1.0       # λ

    # misc
    max_divergence_points_per_traj: int = 1
    debug: bool = False