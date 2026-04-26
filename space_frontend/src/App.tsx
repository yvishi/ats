import { useCallback, useEffect, useRef, useState } from "react";
import { PixelEngine } from "./engine/PixelEngine";
import { TaskBrowser } from "./components/TaskBrowser";
import { ScorePanel } from "./components/ScorePanel";
import { EventLog } from "./components/EventLog";
import { ProjectBrief } from "./components/ProjectBrief";
import { RunHistory, type RunResult } from "./components/RunHistory";
import { ApiPanel } from "./components/ApiPanel";
import type { DemoTask, LogEntry, ScoreState } from "./types";
import { EMPTY_SCORE } from "./types";

type FilterMode = "all" | "atc" | "domain";

const PRESET_MODELS = [
  "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen2.5-14B-Instruct",
  "meta-llama/Llama-3.1-8B-Instruct",
  "mistralai/Mistral-7B-Instruct-v0.3",
];

const DIFFICULTY_ORDER: Record<string, number> = {
  Introductory: 0, Easy: 1, Medium: 2, Hard: 3, Expert: 4,
};

function useDemoTasks() {
  const [tasks, setTasks] = useState<DemoTask[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetch("/demo/tasks")
      .then((r) => r.json())
      .then((j: { tasks?: DemoTask[] }) => {
        setTasks(Array.isArray(j.tasks) ? j.tasks : []);
      })
      .catch(() => setErr("Could not reach /demo/tasks — is the FastAPI server running?"));
  }, []);
  return { tasks, err };
}

export default function App() {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const engineRef   = useRef<PixelEngine | null>(null);
  const esRef       = useRef<EventSource | null>(null);
  const logIdRef    = useRef(0);
  const histIdRef   = useRef(0);

  const { tasks, err: tasksErr } = useDemoTasks();

  const [filter, setFilter]               = useState<FilterMode>("all");
  const [adaptActive, setAdaptActive]     = useState(false);
  const [selectedTask, setSelectedTask]   = useState<DemoTask | null>(null);
  const [running, setRunning]             = useState(false);
  const [score, setScore]                 = useState<ScoreState>(EMPTY_SCORE);
  const [logEntries, setLogEntries]       = useState<LogEntry[]>([]);
  const [agentThinking, setAgentThinking] = useState<"AMAN" | "DMAN" | null>(null);
  const [useLlm, setUseLlm]               = useState(false);
  const [modelName, setModelName]         = useState(PRESET_MODELS[0]);
  const [episodeId, setEpisodeId]         = useState(0);
  const [runHistory, setRunHistory]       = useState<RunResult[]>([]);
  const [activeStreamUrl, setActiveStreamUrl] = useState("");

  /* Init PixelEngine */
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const eng = new PixelEngine(c);
    engineRef.current = eng;
    return () => { eng.destroy(); engineRef.current = null; };
  }, []);

  /* Auto-select first task */
  useEffect(() => {
    if (tasks.length > 0 && !selectedTask) setSelectedTask(tasks[0]);
  }, [tasks, selectedTask]);

  /* Idle canvas: real flight/patient rows from ``task_preview`` (from /demo/tasks). */
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    if (running) return;
    if (!selectedTask) {
      eng.applyEvent({ type: "idle_clear" });
      return;
    }
    if (selectedTask.task_preview) {
      eng.applyEvent({
        type: "idle_preview",
        task_id: selectedTask.task_id,
        visual_profile: selectedTask.mode === "domain" ? "icu" : "atc",
        task: selectedTask.task_preview,
      });
    } else {
      eng.applyEvent({
        type: "scene_reset",
        task_id: selectedTask.task_id,
        visual_profile: selectedTask.mode === "domain" ? "icu" : "atc",
      });
    }
  }, [selectedTask, running]);

  const pushLog = useCallback((type: string, detail: string) => {
    setLogEntries((prev) => [
      ...prev.slice(-299),
      { id: logIdRef.current++, type, detail, ts: Date.now() },
    ]);
  }, []);

  const stopStream = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setRunning(false);
    setAgentThinking(null);
  }, []);

  const runStream = useCallback(() => {
    if (!selectedTask) return;
    stopStream();
    setScore(EMPTY_SCORE);
    setLogEntries([]);
    logIdRef.current = 0;
    setAgentThinking(null);

    const vProfile = selectedTask.mode === "domain" ? "icu" : "atc";
    const u = new URL("/demo/episode/stream", window.location.origin);
    u.searchParams.set("task_id",        selectedTask.task_id);
    u.searchParams.set("mode",           selectedTask.mode);
    u.searchParams.set("visual_profile", vProfile);
    u.searchParams.set("episode_id",     String(episodeId));
    u.searchParams.set("use_generator",  "true");
    u.searchParams.set("use_llm",        useLlm ? "true" : "false");
    if (useLlm && modelName) u.searchParams.set("model", modelName);

    const urlStr = u.toString();
    setActiveStreamUrl(urlStr);

    const es = new EventSource(urlStr);
    esRef.current = es;
    setRunning(true);

    const latestScore: Partial<ScoreState> = {};

    es.onmessage = (msg) => {
      let ev: Record<string, unknown>;
      try { ev = JSON.parse(msg.data) as Record<string, unknown>; }
      catch { return; }

      engineRef.current?.applyEvent(ev);
      const t = String(ev.type);

      if (t === "llm_started") {
        setAgentThinking(String(ev.role) === "DMAN" ? "DMAN" : "AMAN");
      } else if (t === "llm_finished") {
        setAgentThinking(null);
      } else if (t === "score_update") {
        const update: Partial<ScoreState> = {
          composite:   Number(ev.composite   ?? 0),
          aman_reward: Number(ev.aman_reward  ?? 0),
          dman_reward: Number(ev.dman_reward  ?? 0),
        };
        Object.assign(latestScore, update);
        setScore((prev) => ({ ...prev, ...update }));
      } else if (t === "terminal") {
        const sev = String(ev.severity ?? "") as ScoreState["severity"];
        const final: ScoreState = {
          composite:            Number(ev.composite            ?? latestScore.composite    ?? 0),
          aman_reward:          Number(ev.aman_reward           ?? latestScore.aman_reward  ?? 0),
          dman_reward:          Number(ev.dman_reward           ?? latestScore.dman_reward  ?? 0),
          coordination:         Number(ev.coordination          ?? 0),
          cross_lane_conflicts: Number(ev.cross_lane_conflicts  ?? 0),
          atfm_violations:      Number(ev.atfm_violations       ?? 0),
          negotiation_rounds:   Number(ev.negotiation_rounds    ?? 0),
          severity: sev,
          done: true,
        };
        setScore(final);
        setRunHistory((prev) => [
          ...prev.slice(-19),
          {
            id:          histIdRef.current++,
            taskLabel:   selectedTask.label,
            taskId:      selectedTask.task_id,
            domain:      selectedTask.mode,
            episodeId,
            composite:   final.composite,
            aman_reward: final.aman_reward,
            dman_reward: final.dman_reward,
            severity:    sev ?? "",
            usedLlm: useLlm,
            ts: Date.now(),
          },
        ]);
        setRunning(false);
        setAgentThinking(null);
        setActiveStreamUrl("");
      }

      let detail = "";
      switch (t) {
        case "score_update":     detail = `cmp=${Number(ev.composite ?? 0).toFixed(3)}  aman=${Number(ev.aman_reward ?? 0).toFixed(3)}  dman=${Number(ev.dman_reward ?? 0).toFixed(3)}`; break;
        case "terminal":         detail = `${ev.severity}  cmp=${Number(ev.composite ?? 0).toFixed(3)}`; break;
        case "scene_reset":      detail = String(ev.task_id ?? ""); break;
        case "llm_started":      detail = String(ev.role ?? ""); break;
        case "negotiation_tick": detail = `pass ${ev.pass ?? 0}`; break;
        case "action_layout":    detail = `arr=${(ev as {layout?: {aman_arrivals?: unknown[]}}).layout?.aman_arrivals?.length ?? 0}  dep=${(ev as {layout?: {dman_departures?: unknown[]}}).layout?.dman_departures?.length ?? 0}`; break;
        case "error":            detail = String(ev.detail ?? ""); break;
        case "adapt_mapping":    detail = String(ev.rationale_preview ?? "").slice(0, 55); break;
        default: break;
      }
      if (t !== "negotiation_tick") pushLog(t, detail);
    };

    es.onerror = () => {
      pushLog("error", "stream closed");
      stopStream();
      setActiveStreamUrl("");
    };
  }, [selectedTask, episodeId, useLlm, modelName, stopStream, pushLog]);

  useEffect(() => () => stopStream(), [stopStream]);

  function handleTaskSelect(task: DemoTask) {
    if (running) stopStream();
    setSelectedTask(task);
    setScore(EMPTY_SCORE);
    setLogEntries([]);
    logIdRef.current = 0;
    setAgentThinking(null);
  }

  const sortedDiffs = selectedTask?.difficulty
    ? Object.entries(DIFFICULTY_ORDER).sort((a, b) => a[1] - b[1]).map(([k]) => k)
    : [];

  return (
    <div className="app-layout">
      {/* ── Header (sticky) ──────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-glyph">◈</div>
          <div className="brand-text">
            <span className="brand-main">ADAPT</span>
            <span className="brand-sub">ATC Multi-Agent · GRPO RL</span>
          </div>
        </div>

        <div className="header-divider" />

        <div className={`header-badge ${running ? "running" : ""}`}>
          <span className="dot" />
          {running ? "LIVE" : "READY"}
        </div>

        {selectedTask && !running && (
          <div className="header-task-pill">
            <span className="header-task-mode">
              {selectedTask.mode === "domain" ? "⊕ ICU" : "✈ ATC"}
            </span>
            <span className="header-task-name">{selectedTask.label}</span>
          </div>
        )}

        <div className="header-spacer" />

        <div className="header-controls">
          <label className="llm-toggle">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            <span>LLM</span>
          </label>
          {useLlm && (
            <select
              className="model-select"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            >
              {PRESET_MODELS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          )}
        </div>
      </header>

      <div className="app-scroll">
        {/* Full-width scenario context — not squeezed beside the canvas */}
        {selectedTask && (
          <section className={`scenario-hero ${running ? "scenario-hero--compact" : ""}`} aria-labelledby="scenario-title">
            <div className="scenario-hero-inner">
              <div className="scenario-hero-top">
                <span className="scenario-hero-icon" aria-hidden>
                  {selectedTask.mode === "domain" ? "⊕" : "✈"}
                </span>
                <div className="scenario-hero-text">
                  <h1 id="scenario-title" className="scenario-hero-title">{selectedTask.label}</h1>
                  {!running && selectedTask.description && (
                    <p className="scenario-hero-desc">{selectedTask.description}</p>
                  )}
                </div>
                <div className="scenario-hero-badges">
                  <span className={`tier-badge tier-${selectedTask.tier}`}>T{selectedTask.tier}</span>
                  {selectedTask.difficulty && (
                    <span className={`diff-badge diff-${selectedTask.difficulty.toLowerCase()}`}>
                      {selectedTask.difficulty}
                    </span>
                  )}
                  <span className="scenario-domain-pill">
                    {selectedTask.mode === "domain" ? "ICU domain transfer" : "ATC scenario"}
                  </span>
                </div>
              </div>
              {!running && (
                <div className="scenario-hero-stats">
                  {selectedTask.flight_count !== undefined && (
                    <div className="scenario-stat-card">
                      <span className="scenario-stat-label">
                        {selectedTask.mode === "domain" ? "Patients" : "Flights"}
                      </span>
                      <span className="scenario-stat-value mono">{selectedTask.flight_count}</span>
                    </div>
                  )}
                  {selectedTask.runway_count !== undefined && (
                    <div className="scenario-stat-card">
                      <span className="scenario-stat-label">
                        {selectedTask.mode === "domain" ? "Beds / resources" : "Runways"}
                      </span>
                      <span className="scenario-stat-value mono">{selectedTask.runway_count}</span>
                    </div>
                  )}
                  {selectedTask.difficulty && (
                    <div className="scenario-diff-track">
                      <span className="scenario-diff-label">Difficulty curve</span>
                      <div className="td-diff-bar td-diff-bar--spaced">
                        {sortedDiffs.map((d) => (
                          <span
                            key={d}
                            className={`td-diff-step ${
                              (DIFFICULTY_ORDER[d] ?? 0) <=
                              (DIFFICULTY_ORDER[selectedTask.difficulty ?? ""] ?? 0)
                                ? "active"
                                : ""
                            }`}
                            title={d}
                          />
                        ))}
                      </div>
                      <span className="scenario-diff-current">{selectedTask.difficulty}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Three columns: scenarios · visualization · scores & tools */}
        <div className="app-columns">
          <aside className="sidebar-left">
            <ProjectBrief
              adaptActive={adaptActive}
              onToggleAdapt={() => {
                setAdaptActive((v) => {
                  const next = !v;
                  if (!next) setFilter("all");
                  return next;
                });
              }}
            />
            {tasksErr && (
              <div className="server-error-banner">{tasksErr}</div>
            )}
            <TaskBrowser
              tasks={tasks}
              selected={selectedTask}
              filter={filter}
              adaptActive={adaptActive}
              onFilterChange={setFilter}
              onSelect={handleTaskSelect}
            />
          </aside>

          <main className="center-panel">
            <div className="episode-controls">
              <label className="control-label">
                Episode&nbsp;seed
                <input
                  type="number"
                  className="episode-input mono"
                  min={0}
                  max={999}
                  value={episodeId}
                  onChange={(e) =>
                    setEpisodeId(Math.max(0, parseInt(e.target.value) || 0))
                  }
                />
              </label>

              <div className="control-spacer" />

              {running ? (
                <button className="btn-stop" type="button" onClick={stopStream}>■ Stop</button>
              ) : (
                <button
                  type="button"
                  className="btn-run"
                  disabled={!selectedTask}
                  onClick={runStream}
                >
                  ▶ Run Episode
                </button>
              )}
            </div>

            <div className="canvas-shell">
              <canvas ref={canvasRef} width={320} height={180} className="sim-canvas" />
              {!selectedTask && (
                <div className="idle-overlay">
                  <div className="idle-overlay-glyph">◈</div>
                  <div className="idle-overlay-text">Select a scenario</div>
                </div>
              )}
            </div>
          </main>

          <aside className="right-panel">
            <section className="insight-compartment">
              <header className="insight-compartment-head">Outcome</header>
              <div className="insight-compartment-body insight-compartment-body--tight">
                <ScorePanel
                  score={score}
                  running={running}
                  agentThinking={agentThinking}
                  taskLabel={selectedTask?.label}
                />
              </div>
            </section>

            <section className="insight-compartment">
              <header className="insight-compartment-head">
                Live log
                {logEntries.length > 0 && (
                  <span className="insight-compartment-meta mono">{logEntries.length}</span>
                )}
              </header>
              <div className="insight-compartment-body insight-compartment-body--fill">
                <EventLog entries={logEntries} embedded />
              </div>
            </section>

            <section className="insight-compartment">
              <header className="insight-compartment-head">
                Run history
                {runHistory.length > 0 && (
                  <span className="insight-compartment-meta mono">{runHistory.length}</span>
                )}
              </header>
              <div className="insight-compartment-body insight-compartment-body--fill">
                <RunHistory history={runHistory} />
              </div>
            </section>

            <section className="insight-compartment">
              <header className="insight-compartment-head">API reference</header>
              <div className="insight-compartment-body insight-compartment-body--scroll">
                <ApiPanel activeUrl={activeStreamUrl} />
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}
