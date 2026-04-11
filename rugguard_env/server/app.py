"""
FastAPI application for the RugGuard Environment.

Exposes RugGuardEnvironment over HTTP and WebSocket endpoints via create_app.
Each client session gets its own environment instance (SUPPORTS_CONCURRENT_SESSIONS=False).

Usage:
    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production / HF Spaces
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app

    from ..models import RugGuardAction, RugGuardObservation
    from .rugguard_environment import RugGuardEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import RugGuardAction, RugGuardObservation
    from server.rugguard_environment import RugGuardEnvironment


def _make_factory():
    """
    Build a singleton factory.

    The OpenEnv REST endpoints call env_factory() on every request (see
    http_server.py lines ~582/782/1134). Returning a fresh environment each
    time would wipe episode state between /reset and /step calls. Since this
    env has SUPPORTS_CONCURRENT_SESSIONS=False and max_concurrent_envs=1,
    we cache a single instance and hand it back on every call so episode
    state (task_queue, ep_step, ep_reward) persists across HTTP requests.
    """
    steps_per_task = int(os.getenv("RUGGUARD_STEPS_PER_TASK", "15"))
    seed_raw = os.getenv("RUGGUARD_SEED")
    seed = int(seed_raw) if seed_raw else None
    task_filter = os.getenv("RUGGUARD_TASK_FILTER") or None

    cached: dict = {}

    def factory() -> RugGuardEnvironment:
        if "env" not in cached:
            cached["env"] = RugGuardEnvironment(
                steps_per_task=steps_per_task,
                seed=seed,
                task_filter=task_filter,
            )
        return cached["env"]

    return factory


app = create_app(
    _make_factory(),
    RugGuardAction,
    RugGuardObservation,
    env_name="rugguard_env",
    max_concurrent_envs=1,
)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()