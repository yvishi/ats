/** Scene pack id — keep aligned with ``server/demo_tasks.py`` ``visual_scene_key``. */

export type VisualSceneKey =
  | "bhopal"
  | "vadodara"
  | "pune"
  | "nagpur"
  | "delhi"
  | "mumbai"
  | "hyderabad"
  | "bengaluru"
  | "icu";

const TASK_TO_SCENE: Record<string, VisualSceneKey> = {
  bhopal_solo_dep_t0: "bhopal",
  vadodara_mixed_pair_t0: "vadodara",
  pune_wake_intro_t1: "pune",
  nagpur_dual_runway_t1: "nagpur",
  delhi_monsoon_recovery_easy: "delhi",
  mumbai_bank_balance_medium: "mumbai",
  hyderabad_cargo_crunch_medium_hard: "hyderabad",
  bengaluru_irrops_hard: "bengaluru",
  icu_normal_day: "icu",
  icu_flu_surge: "icu",
  icu_mass_casualty: "icu",
};

export function resolveSceneKey(
  taskId: string,
  mode: "atc" | "domain",
): VisualSceneKey {
  if (mode === "domain" || taskId.startsWith("icu_")) {
    return "icu";
  }
  return TASK_TO_SCENE[taskId] ?? "bengaluru";
}

export type ScenePalette = {
  skyTop: string;
  skyBot: string;
  ground: string;
  runway: string;
  runwayMark: string;
  accent: string;
  /** Micro-ambient hint for ``drawSceneDecor``. */
  decor: "none" | "rain" | "cranes" | "heat";
};

export const SCENE_PALETTES: Record<VisualSceneKey, ScenePalette> = {
  bhopal: {
    skyTop: "#6a9cbc",
    skyBot: "#b8d4e8",
    ground: "#3d5c3a",
    runway: "#2a2a2a",
    runwayMark: "#f0f0a0",
    accent: "#e8c86a",
    decor: "none",
  },
  vadodara: {
    skyTop: "#5a8ab8",
    skyBot: "#c8dce8",
    ground: "#4a6044",
    runway: "#252525",
    runwayMark: "#ddd",
    accent: "#88c4e8",
    decor: "none",
  },
  pune: {
    skyTop: "#4a6a98",
    skyBot: "#a8c0d8",
    ground: "#3d5238",
    runway: "#222",
    runwayMark: "#e0e080",
    accent: "#f0a040",
    decor: "heat",
  },
  nagpur: {
    skyTop: "#5878a8",
    skyBot: "#b0c8e0",
    ground: "#455040",
    runway: "#202020",
    runwayMark: "#fff8c0",
    accent: "#90d090",
    decor: "none",
  },
  delhi: {
    skyTop: "#4a5568",
    skyBot: "#8898a8",
    ground: "#3a4538",
    runway: "#1c1c1c",
    runwayMark: "#d0d0a0",
    accent: "#68b0d8",
    decor: "rain",
  },
  mumbai: {
    skyTop: "#4a6088",
    skyBot: "#98b0c8",
    ground: "#3a4840",
    runway: "#1a1a1a",
    runwayMark: "#f5e6a0",
    accent: "#e89840",
    decor: "cranes",
  },
  hyderabad: {
    skyTop: "#5a78a0",
    skyBot: "#b0c4d8",
    ground: "#4a5040",
    runway: "#232323",
    runwayMark: "#e8e0a0",
    accent: "#c8a060",
    decor: "heat",
  },
  bengaluru: {
    skyTop: "#3a5080",
    skyBot: "#8090b0",
    ground: "#2d3830",
    runway: "#181818",
    runwayMark: "#ffe080",
    accent: "#ff6060",
    decor: "rain",
  },
  icu: {
    skyTop: "#1a2838",
    skyBot: "#2a3848",
    ground: "#0f1820",
    runway: "#2a3540",
    runwayMark: "#4a90c8",
    accent: "#40c090",
    decor: "none",
  },
};
