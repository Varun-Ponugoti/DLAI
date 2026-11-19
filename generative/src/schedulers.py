"""Noise schedules / samplers by name, for the M2 scheduler sweep.

All of these plug into pipeline.generate() unchanged: it only relies on the
common scheduler API (set_timesteps, init_noise_sigma, scale_model_input,
step(...).prev_sample), which every scheduler below implements. Only the
sampling algorithm changes; the training noise schedule is held fixed, which
is what makes the comparison fair.
"""
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
)

SCHEDULERS = {
    "pndm": PNDMScheduler,            # SD 1.5 default == your baseline reference
    "ddim": DDIMScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
}


def available():
    return list(SCHEDULERS)


def load_scheduler(name: str, model_id: str):
    key = name.lower()
    if key not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {name!r}. Options: {available()}")
    return SCHEDULERS[key].from_pretrained(model_id, subfolder="scheduler")