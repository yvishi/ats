import type { DemoTask } from "../types";

type FilterMode = "all" | "atc" | "domain";

type Props = {
  tasks: DemoTask[];
  selected: DemoTask | null;
  filter: FilterMode;
  adaptActive: boolean;
  onFilterChange: (f: FilterMode) => void;
  onSelect: (t: DemoTask) => void;
};

function TierBadge({ tier }: { tier: number }) {
  return <span className={`tier-badge tier-${tier}`}>T{tier}</span>;
}

function DiffBadge({ difficulty }: { difficulty: string }) {
  return (
    <span className={`diff-badge diff-${difficulty.toLowerCase()}`}>
      {difficulty}
    </span>
  );
}

export function TaskBrowser({ tasks, selected, filter, adaptActive, onFilterChange, onSelect }: Props) {
  const pool = adaptActive ? tasks : tasks.filter((t) => t.mode === "atc");
  const visible = filter === "all" ? pool
    : filter === "domain" ? pool.filter((t) => t.mode === "domain")
    : pool.filter((t) => t.mode === "atc");

  const atcCount = pool.filter((t) => t.mode === "atc").length;
  const icuCount = pool.filter((t) => t.mode === "domain").length;

  return (
    <div className="task-browser">
      <div className="browser-header">
        <span className="browser-title">SCENARIOS</span>
        <span className="browser-count">{visible.length}</span>
      </div>

      <div className={`filter-tabs ${adaptActive ? "filter-tabs--adapt" : ""}`}>
        <button
          className={`filter-tab ${filter === "all" ? "active" : ""}`}
          onClick={() => onFilterChange("all")}
        >
          ALL <span className="tab-count">{pool.length}</span>
        </button>
        <button
          className={`filter-tab ${filter === "atc" ? "active atc" : ""}`}
          onClick={() => onFilterChange("atc")}
        >
          ATC <span className="tab-count">{atcCount}</span>
        </button>
        {adaptActive && (
          <button
            className={`filter-tab ${filter === "domain" ? "active icu" : ""}`}
            onClick={() => onFilterChange("domain")}
          >
            ICU <span className="tab-count">{icuCount}</span>
          </button>
        )}
      </div>

      <div className="task-list">
        {visible.map((task) => {
          const isAtc = task.mode === "atc";
          const isSelected = selected?.task_id === task.task_id;
          return (
            <button
              key={task.task_id}
              className={`task-card ${isSelected ? "selected" : ""} ${isAtc ? "atc" : "domain"}`}
              onClick={() => onSelect(task)}
            >
              <div className="task-card-top">
                <span className="task-icon">{isAtc ? "✈" : "⊕"}</span>
                <span className="task-label">{task.label}</span>
                <TierBadge tier={task.tier} />
              </div>

              {task.description && (
                <div className="task-desc">{task.description}</div>
              )}

              <div className="task-meta">
                {task.difficulty && <DiffBadge difficulty={task.difficulty} />}
                {task.flight_count !== undefined && (
                  <span className="meta-pill">
                    {task.flight_count}&nbsp;{isAtc ? "flt" : "pts"}
                  </span>
                )}
                {task.runway_count !== undefined && (
                  <span className="meta-pill">
                    {task.runway_count}&nbsp;{isAtc ? "rwy" : "bed"}
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
