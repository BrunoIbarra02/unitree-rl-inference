import math

from pydantic import BaseModel, Field, field_validator

from .model import ACTION_DIM, OBS_DIM


class PredictRequest(BaseModel):
    observation: list[float] = Field(
        ..., description=f"Exactly {OBS_DIM} floats. See /metadata for the required layout and scaling."
    )
    session_id: str | None = Field(
        None,
        description=(
            "Opaque id returned by a previous /predict call for this robot instance, used to continue "
            "the policy's 5-step observation-history window. Omit to start a brand new session (zero-filled history)."
        ),
    )

    @field_validator("observation")
    @classmethod
    def validate_observation(cls, v: list[float]) -> list[float]:
        if len(v) != OBS_DIM:
            raise ValueError(f"observation must have exactly {OBS_DIM} values, got {len(v)}")
        for x in v:
            if not math.isfinite(x):
                raise ValueError("observation contains a non-finite value (NaN/Inf)")
        return v


class PredictResponse(BaseModel):
    actions: list[float] = Field(..., description=f"{ACTION_DIM} raw policy actions, pre action_scale/default_angles.")
    inference_latency_ms: float = Field(..., description="Pure model forward-pass time, excludes HTTP overhead.")
    checkpoint_id: str
    session_id: str = Field(..., description="Pass this back on the next call to continue the history window.")
    session_was_reset: bool = Field(..., description="True if this call started a brand new (zero-init) session.")


class MetadataResponse(BaseModel):
    robot_name: str
    checkpoint_id: str
    checkpoint_source: str
    checkpoint_sha256: str
    observation_dim: int
    action_dim: int
    runtime_backend: str
    requires_recurrent_state: bool
    joint_order: list[str]
    observation_layout: dict
    action_scaling: dict
    input_schema: dict
