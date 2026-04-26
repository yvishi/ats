export type RunResult = {
  id: number;
  taskLabel: string;
  taskId: string;
  domain: "atc" | "domain";
  episodeId: number;
  composite: number;
  aman_reward: number;
  dman_reward: number;
  severity: string;
  usedLlm: boolean;
  ts: number;
};

type Props = {
  history: RunResult[];
};

function SevChip({ sev }: { sev: string }) {
  const cls =
    sev === "success"
      ? "sev-success"
      : sev === "catastrophic"
        ? "sev-catastrophic"
        : sev === "degraded"
          ? "sev-degraded"
          : "sev-idle";
  return <span className={`status-badge ${cls} sev-chip`}>{sev || "—"}</span>;
}

function ScoreBar({ value }: { value: number }) {
  const col =
    value >= 0.7 ? "var(--green)" : value >= 0.4 ? "var(--amber)" : "var(--red)";
  return (
    <div className="rh-bar-track">
      <div
        className="rh-bar-fill"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: col }}
      />
    </div>
  );
}

export function RunHistory({ history }: Props) {
  if (history.length === 0) {
    return (
      <div className="rh-empty">
        No episodes run yet. Hit <strong>Run Episode</strong> to generate results.
      </div>
    );
  }

  return (
    <div className="run-history">
      <table className="rh-table">
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Composite</th>
            <th>AMAN</th>
            <th>DMAN</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {[...history].reverse().map((r) => (
            <tr key={r.id}>
              <td className="rh-task">
                <span className="rh-domain-icon">
                  {r.domain === "domain" ? "⊕" : "✈"}
                </span>
                <span className="rh-task-name">{r.taskLabel}</span>
                {r.usedLlm && <span className="rh-llm-tag">LLM</span>}
              </td>
              <td>
                <div className="rh-score-cell">
                  <span
                    className="mono rh-score-val"
                    style={{
                      color:
                        r.composite >= 0.7
                          ? "var(--green)"
                          : r.composite >= 0.4
                            ? "var(--amber)"
                            : "var(--red)",
                    }}
                  >
                    {r.composite.toFixed(3)}
                  </span>
                  <ScoreBar value={r.composite} />
                </div>
              </td>
              <td className="mono rh-score-sm">{r.aman_reward.toFixed(2)}</td>
              <td className="mono rh-score-sm">{r.dman_reward.toFixed(2)}</td>
              <td>
                <SevChip sev={r.severity} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
