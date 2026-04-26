type Props = {
  adaptActive: boolean;
  onToggleAdapt: () => void;
};

export function ProjectBrief({ adaptActive, onToggleAdapt }: Props) {
  return (
    <div className={`project-brief ${adaptActive ? "project-brief--active" : ""}`}>
      {/* Wordmark */}
      <div className="pb-wordmark">
        <span className="pb-wordmark-glyph">◈</span>
        <div>
          <div className="pb-wordmark-text">ADAPT</div>
          <div className="pb-wordmark-sub">Multi-Agent · GRPO-RL · Zero-Shot Transfer</div>
        </div>
      </div>

      {/* Description */}
      <p className="pb-desc">
        <strong>ADAPT</strong> proves that multi-agent intelligence is{" "}
        <em>domain-agnostic</em>. Two LLM agents — <strong>AMAN</strong>{" "}
        (arrivals) &amp; <strong>DMAN</strong> (departures) — negotiate slot
        assignments under hard constraints, trained with{" "}
        <strong>GRPO</strong> reinforcement learning on Air Traffic Control.
        The same trained agents transfer <em>zero-shot</em> to entirely
        different domains, showing that structured negotiation is a{" "}
        <strong>universal primitive</strong>.
      </p>

      {/* Tags */}
      <div className="pb-tags">
        <span className="pb-tag pb-tag--atc">✈ ATC</span>
        <span className="pb-tag pb-tag--grpo">◎ GRPO-RL</span>
        <span className="pb-tag pb-tag--llm">⬡ LLM Agents</span>
        <span className="pb-tag pb-tag--transfer">↗ Zero-Shot</span>
      </div>

      {/* ADAPT CTA */}
      <div className="pb-adapt-section">
        {!adaptActive && (
          <div className="pb-adapt-cta-hint">
            <span className="pb-adapt-cta-arrow">▼</span>
            <span>Click to unlock domain transfer</span>
            <span className="pb-adapt-cta-arrow">▼</span>
          </div>
        )}

        <button
          type="button"
          className={`pb-adapt-btn ${adaptActive ? "pb-adapt-btn--active" : ""}`}
          onClick={onToggleAdapt}
        >
          {/* Animated background sweep */}
          <span className="pb-adapt-btn-sweep" aria-hidden />

          <span className="pb-adapt-btn-inner">
            <span className="pb-adapt-btn-icon" aria-hidden>⊕</span>
            <span className="pb-adapt-btn-label">ADAPT</span>
            <span className="pb-adapt-btn-sub">
              {adaptActive ? "✓ Domain transfer ON — ICU unlocked" : "Enable domain transfer"}
            </span>
          </span>
        </button>

        {adaptActive && (
          <div className="pb-adapt-active-bar">
            <span className="pb-adapt-active-dot" />
            ICU scenarios now visible in the task list below
          </div>
        )}
      </div>
    </div>
  );
}
