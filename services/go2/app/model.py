"""Go2 quadruped policy wrapper.

Checkpoint: wty-yy/go2_rl_gym, deploy/pre_train/go2/go2_cts_150k.pt
SHA-256: 3e102d0d5e69aebd25e071bcdd87637596561111f924b160ed5ceb0d5c70f400
Format: TorchScript (torch.jit.load), confirmed by loading and inspection.
Chosen over the go2_moe_cts_*.pt variants in the same repo because it is the
non-MoE architecture: single-tensor output (no expert-weights tuple to
unpack), smaller (1.86MB vs 5.2MB), and avoids a confirmed-broken checkpoint
reference in that repo's own deploy_real config (it points at a .pt file
that does not exist in the repo).

Architecture (verified by inspection):
  - A "student_encoder" (CTS -- contrastive teacher-student distillation)
    consumes a rolling window of the last 5 observations to produce a
    latent, which is concatenated with the current observation before the
    actor head.
  - That rolling window is `self.history`, shape (1, 5, 45) -- NOT a
    registered buffer (absent from named_buffers()), but a plain mutable
    TorchScript module attribute that forward() reassigns every call
    (`self.history = updated_window`). It is directly readable/writable
    via attribute access and persists across calls to the same module
    instance, confirmed empirically.
  - Same concurrency problem as G1's LSTM buffers: one shared model
    instance, multiple sessions. Handled the same way -- snapshot each
    session's `history` tensor, swap it into the model attribute right
    before forward(), capture the updated attribute right after, all
    under one lock.
  - No `reset_memory()` method on this checkpoint (confirmed absent); a
    fresh zero-filled (1, 5, 45) tensor is the canonical new-session state.

Observation (45-dim), action (12-dim), scaling -- from
deploy/deploy_mujoco/configs/go2.yaml (this service uses the mujoco/sim
scaling, since it runs as a pure cloud inference endpoint rather than
driving physical hardware; the deploy_real config uses different command
scales and a joint-index remap for the physical SDK, which is out of scope
here per the task's cloud-inference-only boundary):
  obs[0:3]   angular velocity (gyro)                 * ang_vel_scale (0.25)
  obs[3:6]   gravity-orientation vector
  obs[6:9]   velocity command [vx, vy, yaw_rate]      * cmd_scale ([2.0, 2.0, 0.25])
  obs[9:21]  joint pos (q - default_angles)           * dof_pos_scale (1.0)
  obs[21:33] joint vel                                * dof_vel_scale (0.05)
  obs[33:45] previous action (raw, unscaled)

Action scaling: target_dof_pos = default_angles + action * action_scale (0.25)
As with G1, this service returns the raw policy action; applying
default_angles/action_scale is the caller's responsibility.

Joint order (12 DOF, confirmed verbatim from go2.yaml's
`mujoco_joint_names`/`model_joint_names`, identical in both): per leg
[hip, thigh, calf], legs in order FL, FR, RL, RR.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import torch

CHECKPOINT_PATH = "/app/checkpoint/go2_cts_150k.pt"
CHECKPOINT_SHA256 = "3e102d0d5e69aebd25e071bcdd87637596561111f924b160ed5ceb0d5c70f400"
CHECKPOINT_ID = "go2-cts_150k-go2_rl_gym-" + CHECKPOINT_SHA256[:12]

OBS_DIM = 45
ACTION_DIM = 12
HISTORY_LEN = 5
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

SESSION_TTL_S = 600.0


@dataclass
class SessionState:
    history: torch.Tensor
    last_used: float


class Go2Policy:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self._model = torch.jit.load(checkpoint_path, map_location="cpu")
        self._model.eval()
        torch.set_num_threads(1)

        self._forward_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

        with torch.inference_mode():
            self._zero_history = torch.zeros_like(self._model.history)

        self.loaded = True

    def _new_session_locked(self, session_id: str) -> SessionState:
        state = SessionState(self._zero_history.clone(), time.time())
        self._sessions[session_id] = state
        return state

    def _purge_stale_locked(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_used > SESSION_TTL_S]
        for sid in stale:
            del self._sessions[sid]

    def predict(self, observation: list[float], session_id: str | None) -> tuple[list[float], float, str, bool]:
        is_new = False
        if session_id is None:
            session_id = uuid.uuid4().hex
            is_new = True

        with self._sessions_lock:
            self._purge_stale_locked()
            state = self._sessions.get(session_id)
            if state is None:
                state = self._new_session_locked(session_id)
                is_new = True

        x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        t0 = time.perf_counter()
        with self._forward_lock:
            with torch.inference_mode():
                self._model.history = state.history.clone()
                out = self._model(x)
                new_history = self._model.history.clone()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._sessions_lock:
            state.history = new_history
            state.last_used = time.time()

        actions = out.squeeze(0).tolist()
        return actions, latency_ms, session_id, is_new

    def reset_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
