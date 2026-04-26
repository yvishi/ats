/** Minimal flight row from ``serialize_task_snapshot`` (planes or ICU patients). */
export type TaskPreviewFlight = {
  flight_id: string;
  operation: string;
  wake: string;
  scheduled: number;
  earliest: number;
  latest: number;
  runways: string[];
  priority: string;
};

/** Task structure for idle canvas preview (real schedule data, no LLM). */
export type TaskPreview = {
  task_id: string;
  airport: string;
  runways: { runway_id: string; ops: string[] }[];
  flights: TaskPreviewFlight[];
};

export type DemoTask = {
  mode: "atc" | "domain";
  task_id: string;
  tier: number;
  visual_scene_key: string;
  label: string;
  description?: string;
  difficulty?: string;
  flight_count?: number;
  runway_count?: number;
  task_preview?: TaskPreview | null;
};

export type ScoreState = {
  composite: number;
  aman_reward: number;
  dman_reward: number;
  coordination: number;
  cross_lane_conflicts: number;
  atfm_violations: number;
  negotiation_rounds: number;
  severity: "success" | "degraded" | "catastrophic" | null;
  done: boolean;
};

export const EMPTY_SCORE: ScoreState = {
  composite: 0,
  aman_reward: 0,
  dman_reward: 0,
  coordination: 0,
  cross_lane_conflicts: 0,
  atfm_violations: 0,
  negotiation_rounds: 0,
  severity: null,
  done: false,
};

export type LogEntry = {
  id: number;
  type: string;
  detail: string;
  ts: number;
};
