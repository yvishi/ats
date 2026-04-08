"""FastAPI application entrypoint for the ATC optimization environment."""

from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run this environment. Install dependencies first."
    ) from exc

try:
    from ..models import ATCOptimizationAction, ATCOptimizationObservation
    from .atc_environment import ATCOptimizationEnvironment
except ImportError:
    from models import ATCOptimizationAction, ATCOptimizationObservation
    from server.atc_environment import ATCOptimizationEnvironment


app = create_app(
    ATCOptimizationEnvironment,
    ATCOptimizationAction,
    ATCOptimizationObservation,
    env_name="atc_env",
    max_concurrent_envs=8,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ATC Optimization OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f172a; color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; padding: 40px 24px;
  }
  .container { max-width: 760px; margin: 0 auto; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #052e16; color: #4ade80;
    border: 1px solid #166534; border-radius: 999px;
    padding: 6px 16px; font-size: 14px; font-weight: 600;
    margin-bottom: 24px;
  }
  h1 { font-size: 2rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
  .subtitle { color: #94a3b8; font-size: 1rem; margin-bottom: 40px; }
  .section { margin-bottom: 32px; }
  .section-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin-bottom: 12px;
  }
  .task-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 14px 18px;
    margin-bottom: 8px; display: flex;
    justify-content: space-between; align-items: center;
  }
  .task-name { font-weight: 500; color: #e2e8f0; }
  .task-meta { color: #94a3b8; font-size: 0.85rem; }
  .badge-diff {
    font-size: 0.72rem; font-weight: 600; padding: 2px 10px;
    border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .easy { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .medium { background: #422006; color: #fb923c; border: 1px solid #9a3412; }
  .hard { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
  .note {
    color: #cbd5e1; line-height: 1.6; margin-top: 8px;
  }
  .endpoint {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 6px; padding: 10px 16px; margin-bottom: 6px;
    font-family: 'Courier New', monospace; font-size: 0.9rem; color: #7dd3fc;
  }
  .links { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
  a { color: #7dd3fc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .footer { margin-top: 48px; padding-top: 24px; border-top: 1px solid #1e293b;
    color: #475569; font-size: 0.82rem; }
</style>
</head>
<body>
<div class="container">
  <div class="badge">🟢 Live</div>
  <h1>ATC Optimization OpenEnv</h1>
  <p class="subtitle">
    Real-world air traffic disruption recovery environment for LLM agent training.<br>
    Built for the Meta × PyTorch × Scaler OpenEnv Hackathon.
  </p>

  <div class="section">
    <div class="section-title">Tasks</div>
    <div class="task-card">
      <div>
        <div class="task-name">Delhi Monsoon Recovery</div>
        <div class="task-meta">10 flights · 2 runways · weather disruption</div>
      </div>
      <span class="badge-diff easy">Easy</span>
    </div>
    <div class="task-card">
      <div>
        <div class="task-name">Mumbai Hub Bank Balance</div>
        <div class="task-meta">13 flights · 2 runways · airline equity</div>
      </div>
      <span class="badge-diff medium">Medium</span>
    </div>
    <div class="task-card">
      <div>
        <div class="task-name">Bengaluru IRROPS Recovery</div>
        <div class="task-meta">17 flights · 2 runways · emergency priority</div>
      </div>
      <span class="badge-diff hard">Hard</span>
    </div>
    <div class="task-card">
      <div>
        <div class="task-name">Hyderabad Cargo Crunch</div>
        <div class="task-meta">7 flights · 1 runway · capacity constraint</div>
      </div>
      <span class="badge-diff hard">Hard</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">About</div>
    <p class="note">This environment is designed for LLM agent training via the OpenEnv API</p>
  </div>

  <div class="section">
    <div class="section-title">API Endpoints</div>
    <div class="endpoint">POST /reset</div>
    <div class="endpoint">POST /step</div>
    <div class="endpoint">GET  /state</div>
  </div>

  <div class="section">
    <div class="section-title">Links</div>
    <div class="links">
      <a href="https://github.com/GTsingh600/ats" target="_blank">📁 GitHub Repo</a>
      <a href="https://huggingface.co/spaces/GTsingh12/ATS-openenv" target="_blank">🤗 HF Space</a>
    </div>
  </div>

  <div class="footer">
    ATC Optimization OpenEnv · Team Cognito · Meta × PyTorch × Scaler Hackathon 2026
  </div>
</div>
</body>
</html>"""
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server directly."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
