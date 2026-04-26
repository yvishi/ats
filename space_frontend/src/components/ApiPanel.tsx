type Endpoint = {
  method: "GET" | "POST";
  path: string;
  desc: string;
  params?: string[];
};

const ENDPOINTS: Endpoint[] = [
  {
    method: "GET",
    path: "/demo/tasks",
    desc: "Task catalog with metadata (label, difficulty, flight count)",
  },
  {
    method: "GET",
    path: "/demo/episode/stream",
    desc: "SSE live episode stream — runs agents and emits visual events",
    params: ["task_id", "mode", "visual_profile", "episode_id", "use_llm", "model"],
  },
  {
    method: "POST",
    path: "/multi_agent/reset",
    desc: "Reset multi-agent environment for a given task",
  },
  {
    method: "POST",
    path: "/multi_agent/step/bid",
    desc: "Submit BID-round actions from AMAN and DMAN",
  },
  {
    method: "POST",
    path: "/multi_agent/finalize",
    desc: "Finalize episode and return full scored result",
  },
  {
    method: "POST",
    path: "/multi_agent/episode",
    desc: "Run a complete episode end-to-end (REST alternative to SSE)",
  },
  {
    method: "GET",
    path: "/multi_agent/status",
    desc: "Current multi-agent environment state summary",
  },
  {
    method: "GET",
    path: "/multi_agent/profiles",
    desc: "List available supervisor preference profiles",
  },
];

type Props = {
  activeUrl: string;
};

export function ApiPanel({ activeUrl }: Props) {
  const activePath = activeUrl ? new URL(activeUrl).pathname : "";

  return (
    <div className="api-panel">
      <div className="api-section-label">ACTIVE REQUEST</div>
      {activeUrl ? (
        <div className="api-active-url mono">{activeUrl.slice(activeUrl.indexOf("/"))}</div>
      ) : (
        <div className="api-none">No active request</div>
      )}

      <div className="api-section-label" style={{ marginTop: "0.75rem" }}>
        ENDPOINTS
      </div>
      <div className="api-endpoint-list">
        {ENDPOINTS.map((ep) => {
          const isActive = activePath === ep.path;
          return (
            <div key={ep.path} className={`api-endpoint ${isActive ? "api-endpoint-active" : ""}`}>
              <div className="api-endpoint-top">
                <span className={`api-method api-method-${ep.method.toLowerCase()}`}>
                  {ep.method}
                </span>
                <span className="api-path mono">{ep.path}</span>
                {isActive && <span className="api-live-dot" />}
              </div>
              <div className="api-desc">{ep.desc}</div>
              {ep.params && (
                <div className="api-params">
                  {ep.params.map((param) => (
                    <span key={param} className="api-param mono">
                      {param}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
