"""FastAPI application for the multi-agent ATC environment.

Exposes:
  OpenEnv standard surface (via create_app):
    POST /reset
    POST /step
    GET  /state
    GET  /health

  Extended multi-agent REST endpoints:
    POST /multi_agent/reset
    POST /multi_agent/step/bid
    POST /multi_agent/finalize
    POST /multi_agent/episode
    GET  /multi_agent/profiles
    GET  /multi_agent/status

  Demo (HF Space):
    GET  /demo/tasks
    GET  /demo/episode/stream  (SSE visual events)
    GET  /                      (built ``space_frontend/dist``)

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv is required. Install dependencies first."
    ) from exc

try:
    from ..atc_env.models import ATCAction, ATCObservation
    from ..atc_env.server.atc_environment import ATCEnvironment
    from ..multi_agent.environment import MultiAgentATCEnvironment
    from ..multi_agent.generator import ChallengeGenerator
    from ..multi_agent.supervisor import SupervisorAgent
    from ..domains import get_all_domain_tasks
    from ..multi_agent.inference import (
        API_BASE_URL,
        HF_TOKEN,
        MODEL_NAME,
        run_domain_episode,
        run_episode,
    )
    from ..multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )
    from ..tasks import task_catalog
    from .demo_tasks import VISUAL_DEMO_TASKS
except ImportError:
    from atc_env.models import ATCAction, ATCObservation
    from atc_env.server.atc_environment import ATCEnvironment
    from domains import get_all_domain_tasks
    from multi_agent.environment import MultiAgentATCEnvironment
    from multi_agent.generator import ChallengeGenerator
    from multi_agent.supervisor import SupervisorAgent
    from multi_agent.inference import (
        API_BASE_URL,
        HF_TOKEN,
        MODEL_NAME,
        run_domain_episode,
        run_episode,
    )
    from multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )
    from server.demo_tasks import VISUAL_DEMO_TASKS
    from tasks import task_catalog

# Shared multi-agent singletons (serialised via lock — one env per server process)
_ma_env        = MultiAgentATCEnvironment(seed=0)
_ma_generator  = ChallengeGenerator(seed=0)
_ma_supervisor = SupervisorAgent()
_MA_LOCK = asyncio.Lock()

app = create_app(
    ATCEnvironment,
    ATCAction,
    ATCObservation,
    env_name="atc_env",
    max_concurrent_envs=8,
)


# ── Multi-agent REST endpoints ─────────────────────────────────────────────────

class MAResetRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    supervisor_profile: Optional[str] = None


class MABidRequest(BaseModel):
    aman_action: Dict[str, Any]
    dman_action: Dict[str, Any]


class MAEpisodeRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    use_generator: bool = True
    use_heuristic: bool = True


@app.post("/multi_agent/reset", tags=["multi-agent"])
async def ma_reset(req: MAResetRequest) -> Dict[str, Any]:
    """Reset multi-agent environment, return AMAN and DMAN observations."""
    async with _MA_LOCK:
        try:
            profile = (
                SupervisorProfileName(req.supervisor_profile)
                if req.supervisor_profile
                else None
            )
            aman_obs, dman_obs = await asyncio.to_thread(
                _ma_env.reset,
                task_id=req.task_id,
                episode_id=req.episode_id,
                supervisor_profile=profile,
            )
            return {
                "aman_observation": json.loads(aman_obs.model_dump_json()),
                "dman_observation": json.loads(dman_obs.model_dump_json()),
                "atfm_deadlines": _ma_env._state.atfm_deadlines,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/step/bid", tags=["multi-agent"])
async def ma_step_bid(req: MABidRequest) -> Dict[str, Any]:
    """Submit BID-round actions from both agents."""
    async with _MA_LOCK:
        try:
            aman_action = AMANAction.model_validate(req.aman_action)
            dman_action  = DMANAction.model_validate(req.dman_action)
            aman_obs2, dman_obs2, reward, done = await asyncio.to_thread(
                _ma_env.step_bid, aman_action, dman_action
            )
            return {
                "aman_observation": json.loads(aman_obs2.model_dump_json()),
                "dman_observation": json.loads(dman_obs2.model_dump_json()),
                "partial_reward": reward,
                "done": done,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/finalize", tags=["multi-agent"])
async def ma_finalize() -> Dict[str, Any]:
    """Finalize episode, return full scored result."""
    async with _MA_LOCK:
        try:
            result: MultiAgentEpisodeResult = await asyncio.to_thread(_ma_env.finalize)
            return json.loads(result.model_dump_json())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/multi_agent/episode", tags=["multi-agent"])
async def ma_run_episode(req: MAEpisodeRequest) -> Dict[str, Any]:
    """Run a full multi-agent episode with heuristic or LLM agents."""
    async with _MA_LOCK:
        try:
            result = await asyncio.to_thread(
                run_episode,
                task_id=req.task_id,
                client=None,
                env=_ma_env,
                generator=_ma_generator if req.use_generator else None,
                supervisor=_ma_supervisor,
                episode_id=req.episode_id,
                use_generator=req.use_generator,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/multi_agent/profiles", tags=["multi-agent"])
async def ma_profiles() -> List[str]:
    """List available supervisor preference profiles."""
    return [p.value for p in SupervisorProfileName]


@app.get("/multi_agent/status", tags=["multi-agent"])
async def ma_status() -> Dict[str, Any]:
    """Current multi-agent environment state summary."""
    state = _ma_env._state
    if state is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "task_id": state.task.task_id if state.task else None,
        "round_number": state.round_number,
        "aman_slots_count": len(state.aman_slots),
        "dman_slots_count": len(state.dman_slots),
        "generator_difficulty": _ma_generator.difficulty_level,
        "supervisor_profile": (
            state.supervisor_profile.value if state.supervisor_profile else None
        ),
    }


# ── HF Space demo: static UI + visual episode stream ──────────────────────────

_SPACE_DIST = Path(__file__).resolve().parent.parent / "space_frontend" / "dist"
_SPACE_INDEX = _SPACE_DIST / "index.html"
_SPACE_ASSETS = _SPACE_DIST / "assets"


def _validate_stream_task(mode: str, task_id: str) -> None:
    if mode == "domain":
        if task_id not in get_all_domain_tasks():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown domain task_id={task_id!r}",
            )
    elif mode == "atc":
        if task_id not in task_catalog():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ATC task_id={task_id!r}",
            )
    else:
        raise HTTPException(status_code=400, detail="mode must be 'atc' or 'domain'")


async def _sse_episode_gen(
    task_id: str,
    mode: str,
    visual_profile: str,
    episode_id: int,
    use_generator: bool,
    use_llm: bool,
) -> AsyncIterator[str]:
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            from openai import OpenAI

            stream_env = MultiAgentATCEnvironment(seed=episode_id)
            stream_gen = ChallengeGenerator(seed=episode_id) if use_generator else None
            supervisor = SupervisorAgent()
            want_llm = bool(use_llm and HF_TOKEN)
            client = (
                OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
                if want_llm
                else None
            )

            def sink(ev: Dict[str, Any]) -> None:
                q.put(("ev", ev))

            if mode == "domain":
                run_domain_episode(
                    domain_task_id=task_id,
                    client=client,
                    env=stream_env,
                    supervisor=supervisor,
                    episode_id=episode_id,
                    model_name=MODEL_NAME,
                    visual_sink=sink,
                    visual_profile=visual_profile,
                )
            else:
                run_episode(
                    task_id=task_id,
                    client=client,
                    env=stream_env,
                    generator=stream_gen,
                    supervisor=supervisor,
                    episode_id=episode_id,
                    use_generator=use_generator,
                    model_name=MODEL_NAME,
                    visual_sink=sink,
                    visual_profile=visual_profile,
                )
        except Exception as exc:
            q.put(("ev", {"type": "error", "detail": str(exc)}))
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = await asyncio.to_thread(q.get)
        if kind == "done":
            break
        yield f"data: {json.dumps(payload, default=str)}\n\n"


@app.get("/demo/tasks", tags=["demo"])
async def demo_tasks() -> Dict[str, Any]:
    """Curated tasks for the Space picker (tier + ``visual_scene_key``)."""
    return {"tasks": list(VISUAL_DEMO_TASKS)}


@app.get("/demo/episode/stream", tags=["demo"])
async def demo_episode_stream(
    task_id: str = Query(..., min_length=1),
    mode: Literal["atc", "domain"] = Query("atc"),
    visual_profile: Literal["atc", "icu"] = Query("atc"),
    episode_id: int = Query(0, ge=0, le=1_000_000),
    use_generator: bool = Query(True),
    use_llm: bool = Query(False),
) -> StreamingResponse:
    """SSE stream of visual events for one episode (``run_episode`` / ``run_domain_episode``)."""
    _validate_stream_task(mode, task_id)
    prof = visual_profile
    if mode == "domain" and visual_profile == "atc":
        prof = "icu"

    return StreamingResponse(
        _sse_episode_gen(
            task_id=task_id,
            mode=mode,
            visual_profile=prof,
            episode_id=episode_id,
            use_generator=use_generator,
            use_llm=use_llm,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if _SPACE_ASSETS.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_SPACE_ASSETS)),
        name="space_frontend_assets",
    )


@app.get("/", tags=["demo"], include_in_schema=False)
async def space_frontend_index() -> FileResponse:
    if not _SPACE_INDEX.is_file():
        raise HTTPException(
            status_code=503,
            detail="space_frontend not built (run npm run build in space_frontend/)",
        )
    return FileResponse(str(_SPACE_INDEX))


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
