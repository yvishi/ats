import { useCallback, useEffect, useRef, useState } from "react";
import { PixelEngine } from "./engine/PixelEngine";

type DemoTask = {
  mode: "atc" | "domain";
  task_id: string;
  tier: number;
  visual_scene_key: string;
  label: string;
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
      .catch(() => setErr("Could not load /demo/tasks"));
  }, []);
  return { tasks, err };
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<PixelEngine | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const { tasks, err: tasksErr } = useDemoTasks();
  const [taskId, setTaskId] = useState("mumbai_bank_balance_medium");
  const [mode, setMode] = useState<"atc" | "domain">("atc");
  const [visualProfile, setVisualProfile] = useState<"atc" | "icu">("atc");
  const [useLlm, setUseLlm] = useState(false);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string>("");

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }
    const eng = new PixelEngine(c);
    engineRef.current = eng;
    return () => {
      eng.destroy();
      engineRef.current = null;
    };
  }, []);

  const appendLog = useCallback((line: string) => {
    setLog((prev) => (prev + line + "\n").slice(-4000));
  }, []);

  const stopStream = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setRunning(false);
  }, []);

  const runStream = useCallback(() => {
    stopStream();
    setLog("");
    const u = new URL("/demo/episode/stream", window.location.origin);
    u.searchParams.set("task_id", taskId);
    u.searchParams.set("mode", mode);
    u.searchParams.set("visual_profile", visualProfile);
    u.searchParams.set("episode_id", "0");
    u.searchParams.set("use_generator", "true");
    u.searchParams.set("use_llm", useLlm ? "true" : "false");

    const es = new EventSource(u.toString());
    esRef.current = es;
    setRunning(true);

    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data) as Record<string, unknown>;
        engineRef.current?.applyEvent(ev);
        const t = String(ev.type);
        if (t !== "negotiation_tick") {
          appendLog(`${t}`);
        }
      } catch {
        appendLog(`(parse error) ${msg.data}`);
      }
    };
    es.onerror = () => {
      appendLog("(stream error / closed)");
      stopStream();
    };
  }, [appendLog, mode, stopStream, taskId, useLlm, visualProfile]);

  useEffect(() => () => stopStream(), [stopStream]);

  return (
    <div className="app-wrap">
      <h1>ADAPT · live episode</h1>
      <p className="sub">
        Pixel replay from server visual events. Catastrophic terminal outcome triggers the shared
        failure sequence.
      </p>
      {tasksErr && <p className="sub">{tasksErr}</p>}

      <div className="controls">
        <label>
          Task
          <select
            value={taskId}
            onChange={(e) => {
              const v = e.target.value;
              setTaskId(v);
              const row = tasks.find((t) => t.task_id === v);
              if (row) {
                setMode(row.mode);
                if (row.mode === "domain") {
                  setVisualProfile("icu");
                }
              }
            }}
          >
            {tasks.length === 0 && <option value={taskId}>{taskId}</option>}
            {tasks.map((t) => (
              <option key={`${t.mode}-${t.task_id}`} value={t.task_id}>
                {t.label} ({t.task_id})
              </option>
            ))}
          </select>
        </label>
        <label>
          Mode
          <select value={mode} onChange={(e) => setMode(e.target.value as "atc" | "domain")}>
            <option value="atc">ATC</option>
            <option value="domain">Domain (ICU)</option>
          </select>
        </label>
        <label>
          Visual profile
          <select
            value={visualProfile}
            onChange={(e) => setVisualProfile(e.target.value as "atc" | "icu")}
          >
            <option value="atc">ATC chrome</option>
            <option value="icu">ICU chrome</option>
          </select>
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: "0.35rem" }}>
          <input
            type="checkbox"
            checked={useLlm}
            onChange={(e) => setUseLlm(e.target.checked)}
          />
          HF router LLM
        </label>
        <button type="button" className="primary" disabled={running} onClick={runStream}>
          Run stream
        </button>
        <button type="button" onClick={stopStream} disabled={!running}>
          Stop
        </button>
      </div>

      <div className="canvas-shell">
        <canvas ref={canvasRef} width={320} height={180} />
      </div>
      <div className="log">{log || "…"}</div>
    </div>
  );
}
