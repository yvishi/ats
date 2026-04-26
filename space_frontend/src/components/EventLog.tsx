import { useEffect, useRef } from "react";
import type { LogEntry } from "../types";

type Props = {
  entries: LogEntry[];
  /** When true, omit the duplicate section title (parent supplies heading). */
  embedded?: boolean;
};

const EVENT_COLORS: Record<string, string> = {
  scene_reset: "var(--accent)",
  adapt_scene: "var(--purple)",
  adapt_mapping: "#cc88ff",
  llm_started: "#ffdd44",
  llm_finished: "#cc9933",
  action_layout: "var(--green)",
  negotiation_tick: "#7799ff",
  score_update: "var(--green)",
  terminal: "var(--red)",
  error: "#ff2244",
};

function fmt(ts: number): string {
  return new Date(ts).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function EventLog({ entries, embedded = false }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevLen = useRef(0);

  useEffect(() => {
    if (entries.length !== prevLen.current) {
      prevLen.current = entries.length;
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [entries.length]);

  return (
    <div className={`event-log ${embedded ? "event-log--embedded" : ""}`}>
      {!embedded && <div className="panel-title">EVENT LOG</div>}
      <div className="log-entries">
        {entries.length === 0 && (
          <div className="log-empty">Awaiting episode…</div>
        )}
        {entries.map((e) => (
          <div key={e.id} className="log-entry">
            <div className="log-entry-head">
              <span className="log-ts mono">{fmt(e.ts)}</span>
              <span
                className="log-type mono"
                style={{ color: EVENT_COLORS[e.type] ?? "var(--text-dim)" }}
              >
                {e.type}
              </span>
            </div>
            {e.detail ? (
              <div className="log-detail">{e.detail}</div>
            ) : null}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
