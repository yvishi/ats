import type { ScoreState } from "../types";

type Props = {
  score: ScoreState;
  running: boolean;
  agentThinking: "AMAN" | "DMAN" | null;
  taskLabel?: string;
};

function ScoreBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = `${Math.max(0, Math.min(1, value)) * 100}%`;
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: pct, background: color }} />
      </div>
      <span className="score-value mono">{value.toFixed(3)}</span>
    </div>
  );
}

function severityMeta(
  sev: ScoreState["severity"],
  running: boolean,
  done: boolean,
): { label: string; cls: string } {
  if (sev === "success") return { label: "SUCCESS", cls: "sev-success" };
  if (sev === "degraded") return { label: "DEGRADED", cls: "sev-degraded" };
  if (sev === "catastrophic") return { label: "CATASTROPHIC", cls: "sev-catastrophic" };
  if (running) return { label: "RUNNING", cls: "sev-running" };
  if (done) return { label: "DONE", cls: "sev-done" };
  return { label: "IDLE", cls: "sev-idle" };
}

export function ScorePanel({ score, running, agentThinking, taskLabel }: Props) {
  const { label: sevLabel, cls: sevCls } = severityMeta(
    score.severity,
    running,
    score.done,
  );

  const compositeColor =
    score.composite >= 0.7
      ? "var(--green)"
      : score.composite >= 0.4
        ? "var(--amber)"
        : score.composite > 0
          ? "var(--red)"
          : "var(--muted)";

  const hasScores = score.aman_reward > 0 || score.dman_reward > 0 || score.done;

  return (
    <div className="score-panel">
      {agentThinking && (
        <div className="agent-thinking">
          <span className="thinking-pulse" />
          <span className="thinking-text">
            {agentThinking} <span className="thinking-action">generating…</span>
          </span>
        </div>
      )}

      <div className="panel-section">
        <div className="panel-title">EPISODE SCORE</div>

        {taskLabel && !running && !score.done && (
          <div className="score-task-label">{taskLabel}</div>
        )}

        <div className="composite-display">
          <div
            className="composite-value mono"
            style={{ color: compositeColor }}
          >
            {score.composite > 0 ? score.composite.toFixed(3) : "—"}
          </div>

          <div className="composite-bar-wrap">
            <div
              className="composite-bar-fill"
              style={{
                width: `${score.composite * 100}%`,
                background: `linear-gradient(90deg, var(--accent-dim), ${compositeColor})`,
              }}
            />
          </div>

          <div className={`status-badge ${sevCls}`}>
            {running && !score.done && <span className="blink-dot">●&nbsp;</span>}
            {sevLabel}
          </div>
        </div>
      </div>

      {hasScores && (
        <div className="panel-section">
          <div className="panel-title">AGENTS</div>
          <ScoreBar label="AMAN" value={score.aman_reward} color="var(--accent)" />
          <ScoreBar label="DMAN" value={score.dman_reward} color="var(--green)" />
          {score.coordination > 0 && (
            <ScoreBar label="COORD" value={score.coordination} color="var(--amber)" />
          )}
        </div>
      )}

      {(score.done || score.negotiation_rounds > 0) && (
        <div className="panel-section">
          <div className="panel-title">METRICS</div>
          <div className="metrics-grid">
            <div className="metric-cell">
              <span className="metric-key">Conflicts</span>
              <span
                className={`metric-val mono ${score.cross_lane_conflicts > 0 ? "val-red" : "val-green"}`}
              >
                {score.cross_lane_conflicts}
              </span>
            </div>
            <div className="metric-cell">
              <span className="metric-key">ATFM Viol.</span>
              <span
                className={`metric-val mono ${score.atfm_violations > 0 ? "val-amber" : "val-green"}`}
              >
                {score.atfm_violations}
              </span>
            </div>
            <div className="metric-cell">
              <span className="metric-key">Neg. Rounds</span>
              <span className="metric-val mono val-accent">
                {score.negotiation_rounds}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
