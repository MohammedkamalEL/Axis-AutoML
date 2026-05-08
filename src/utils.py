import os


def setup_dirs() -> dict:
    """Creates all required project directories and returns their paths."""
    dirs = {
        "data":         "data",
        "submissions":  "submissions",
        "models":       "model_registry",
        "experiments":  "spaceship_experiments",
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs
