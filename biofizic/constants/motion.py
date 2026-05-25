"""
Motion and HAR-related caps.

HAR classes come from WISDM smartwatch training (STILL, SCROLL, HAND, WALK).
"""

# Maximum display arousal (0-10) per HAR class when motion caps physiology.
HAR_AROUSAL_CAP_BY_CLASS = {
    "STILL": 10,
    "SCROLL": 6,
    "HAND": 6,
    "WALK": 5,
}

HAR_CLASS_NAMES = ("STILL", "SCROLL", "HAND", "WALK")

# Activity context window at 1 Hz (seconds of acc_rms samples).
ACTIVITY_CONTEXT_WINDOW_SAMPLES = 10
