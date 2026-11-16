import logging

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .model import ACTION_DIM, CHECKPOINT_ID, CHECKPOINT_SHA256, JOINT_NAMES, OBS_DIM, G1Policy, NonFiniteOutputError
from .schemas import MetadataResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("g1-service")

app = FastAPI(title="Unitree G1 Locomotion Inference")

_policy: G1Policy | None = None


@app.on_event("startup")
def load_model():
    global _policy
    logger.info("Loading G1 policy checkpoint...")
    _policy = G1Policy()
    logger.info("G1 policy loaded, checkpoint_id=%s", CHECKPOINT_ID)


@app.get("/health")
def health():
    if _policy is None or not getattr(_policy, "loaded", False):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    return MetadataResponse(
        robot_name="unitree_g1",
        checkpoint_id=CHECKPOINT_ID,
        checkpoint_source="unitreerobotics/unitree_rl_gym:deploy/pre_train/g1/motion.pt",
        checkpoint_sha256=CHECKPOINT_SHA256,
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        runtime_backend="pytorch-torchscript-cpu",
        requires_recurrent_state=True,
        joint_order=JOINT_NAMES,
        observation_layout={
            "0:3": "angular_velocity * 0.25",
            "3:6": "gravity_orientation_vector",
            "6:9": "velocity_command [vx, vy, yaw_rate] * [2.0, 2.0, 0.25]",
            "9:21": "(joint_pos - default_angles) * 1.0",
            "21:33": "joint_vel * 0.05",
            "33:45": "previous_action (raw)",
            "45:47": "[sin(phase), cos(phase)], 0.8s assumed gait period",
        },
        action_scaling={
            "formula": "target_dof_pos = default_angles + action * 0.25",
            "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
            "note": "This service returns the raw action; applying default_angles/action_scale is the caller's responsibility.",
        },
        input_schema={
            "observation": f"array of {OBS_DIM} floats, pre-scaled as above",
            "session_id": "optional string; omit to start a new zero-initialized LSTM session",
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    assert _policy is not None
    try:
        actions, latency_ms, session_id, was_reset = _policy.predict(req.observation, req.session_id)
    except NonFiniteOutputError as e:
        logger.warning("non-finite policy output: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    return PredictResponse(
        actions=actions,
        inference_latency_ms=latency_ms,
        checkpoint_id=CHECKPOINT_ID,
        session_id=session_id,
        session_was_reset=was_reset,
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    assert _policy is not None
    existed = _policy.reset_session(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "deleted", "session_id": session_id}
