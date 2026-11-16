"""G1 biped policy wrapper.

Checkpoint: unitreerobotics/unitree_rl_gym, deploy/pre_train/g1/motion.pt
SHA-256: cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d
Format: TorchScript (torch.jit.load), confirmed by loading and inspection.

Architecture (verified by inspection, not assumed):
  - PolicyExporterLSTM wrapper around an ActorCriticRecurrent actor.
  - forward(x) internally runs the LSTM and *mutates in place* two
    registered buffers, `hidden_state` (1,1,64) and `cell_state` (1,1,64),
    as the recurrent memory. There is no external hidden-state argument --
    the state lives inside the loaded module itself.
  - Because of that, a single shared model instance cannot naively serve
    multiple concurrent "robots"/sessions: calling forward() for session A
    would corrupt session B's memory. We handle this by snapshotting each
    session's (hidden_state, cell_state) tensors, swapping them into the
    model's buffers immediately before each forward call, and capturing the
    updated buffers immediately after -- all under a single lock so only
    one call touches the shared buffers at a time.
  - `reset_memory()` (present on the module, verified) zeroes both buffers;
    we use it once at startup to capture the canonical zero-init state used
    for brand new sessions.

Observation (47-dim), action (12-dim), scaling -- from
deploy/deploy_mujoco/configs/g1.yaml and deploy/deploy_real/configs/g1.yaml
(identical values in both):
  obs[0:3]   angular velocity (gyro)            * ang_vel_scale (0.25)
  obs[3:6]   gravity-orientation vector (from quaternion)
  obs[6:9]   velocity command [vx, vy, yaw_rate] * cmd_scale ([2.0, 2.0, 0.25])
  obs[9:21]  joint pos  (q - default_angles)      * dof_pos_scale (1.0)
  obs[21:33] joint vel                            * dof_vel_scale (0.05)
  obs[33:45] previous action (raw, unscaled)
  obs[45:47] [sin(phase), cos(phase)] for an assumed 0.8s gait period

This service does NOT compute the observation vector for the caller --
callers must supply it pre-scaled exactly as above (see schemas.py). This
keeps the service a pure policy-inference endpoint, consistent with the
task's cloud-inference-only scope (no low-level motor control implied).

Action scaling: target_dof_pos = default_angles + action * action_scale (0.25)
This service returns the raw policy action (pre-scaling); applying
default_angles/action_scale is the caller's responsibility if driving a
real/simulated robot, since that mapping is robot/hardware-specific.

Joint order (12 leg DOF only -- G1's 12-DOF leg variant; arms/waist are not
driven by this policy): per leg [hip_pitch, hip_roll, hip_yaw, knee,
ankle_pitch, ankle_roll], left leg then right leg. Order derived from the
deploy config's default_angles value sequence cross-referenced against the
standard Unitree G1 leg DOF convention -- verify against your own URDF
before using these labels for real hardware actuation.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import torch

CHECKPOINT_PATH = "/app/checkpoint/motion.pt"
CHECKPOINT_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
CHECKPOINT_ID = "g1-motion-unitree_rl_gym-" + CHECKPOINT_SHA256[:12]

OBS_DIM = 47
ACTION_DIM = 12
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
]

SESSION_TTL_S = 600.0


@dataclass
class SessionState:
    hidden_state: torch.Tensor
    cell_state: torch.Tensor
    last_used: float


class G1Policy:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self._model = torch.jit.load(checkpoint_path, map_location="cpu")
        self._model.eval()
        torch.set_num_threads(1)

        self._forward_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

        with torch.inference_mode():
            self._model.reset_memory()
            self._zero_hidden = self._model.hidden_state.clone()
            self._zero_cell = self._model.cell_state.clone()

        self.loaded = True

    def _new_session_locked(self, session_id: str) -> SessionState:
        state = SessionState(self._zero_hidden.clone(), self._zero_cell.clone(), time.time())
        self._sessions[session_id] = state
        return state

    def _purge_stale_locked(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_used > SESSION_TTL_S]
        for sid in stale:
            del self._sessions[sid]

    def predict(self, observation: list[float], session_id: str | None) -> tuple[list[float], float, str, bool]:
        """Returns (actions, inference_latency_ms, session_id, session_was_reset)."""
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

        # Model forward() expects a batched (1, OBS_DIM) input -- confirmed
        # empirically: an unbatched (OBS_DIM,) tensor raises inside the
        # TorchScript LSTM ("For unbatched 2-D input, hx and cx should also
        # be 2-D but got (3-D, 3-D) tensors"), since hidden/cell are (1,1,64).
        x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        t0 = time.perf_counter()
        with self._forward_lock:
            with torch.inference_mode():
                self._model.hidden_state.copy_(state.hidden_state)
                self._model.cell_state.copy_(state.cell_state)
                out = self._model(x)
                new_hidden = self._model.hidden_state.clone()
                new_cell = self._model.cell_state.clone()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._sessions_lock:
            state.hidden_state = new_hidden
            state.cell_state = new_cell
            state.last_used = time.time()

        actions = out.squeeze(0).tolist()
        return actions, latency_ms, session_id, is_new

    def reset_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
