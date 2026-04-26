import { catastrophicMotion, drawCatastrophicOverlay } from "./catastrophicFailure";
import {
  resolveSceneKey,
  SCENE_PALETTES,
  type ScenePalette,
  type VisualSceneKey,
} from "../tasks/registry";

const LW = 320;
const LH = 180;

type LayoutSlot = {
  flight_id: string;
  runway: string;
  assigned_minute: number;
  hold_minutes: number;
};

type ActionLayout = {
  aman_arrivals: LayoutSlot[];
  dman_departures: LayoutSlot[];
};

type VisualEvent = Record<string, unknown>;

export class PixelEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private raf = 0;

  private sceneKey: VisualSceneKey = "bhopal";
  private profile: "atc" | "icu" = "atc";
  private mode: "atc" | "domain" = "atc";
  private palette: ScenePalette = SCENE_PALETTES.bhopal;

  private taskSnapshot: Record<string, unknown> | null = null;
  private layout: ActionLayout | null = null;

  private thinking: "AMAN" | "DMAN" | null = null;
  private negPass = 0;
  private composite = 0;
  private statusLine = "";

  private catastrophicStart: number | null = null;
  private decorT = 0;

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d context required");
    }
    this.canvas = canvas;
    this.ctx = ctx;
    this.resize();
    window.addEventListener("resize", this.resize);
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
  }

  destroy(): void {
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.resize);
  }

  private resize = (): void => {
    const parent = this.canvas.parentElement;
    const pw = parent?.clientWidth ?? 640;
    const scale = Math.max(1, Math.floor(pw / LW));
    this.canvas.width = LW * scale;
    this.canvas.height = LH * scale;
    this.canvas.style.width = `${LW * scale}px`;
    this.canvas.style.height = `${LH * scale}px`;
    this.ctx.imageSmoothingEnabled = false;
  };

  applyEvent(ev: VisualEvent): void {
    const t = String(ev.type);
    switch (t) {
      case "scene_reset": {
        const tid = String(ev.task_id ?? "");
        this.profile =
          String((ev as { visual_profile?: string }).visual_profile ?? "atc") === "icu"
            ? "icu"
            : "atc";
        this.mode = tid.startsWith("icu_") ? "domain" : "atc";
        this.sceneKey = resolveSceneKey(tid, this.mode);
        this.palette = SCENE_PALETTES[this.sceneKey];
        this.taskSnapshot = (ev as { task?: Record<string, unknown> }).task ?? null;
        this.layout = null;
        this.thinking = null;
        this.negPass = 0;
        this.statusLine = `${tid}`;
        break;
      }
      case "adapt_scene":
        this.mode = "domain";
        this.sceneKey = "icu";
        this.palette = SCENE_PALETTES.icu;
        this.statusLine = `ADAPT · ${String(ev.domain_task_id ?? "")}`;
        break;
      case "adapt_mapping":
        this.statusLine = `Mapping · ${String((ev as { rationale_preview?: string }).rationale_preview ?? "").slice(0, 80)}`;
        break;
      case "llm_started":
        this.thinking = String(ev.role) === "DMAN" ? "DMAN" : "AMAN";
        this.statusLine = `${this.thinking} generating…`;
        break;
      case "llm_finished":
        this.thinking = null;
        break;
      case "action_layout":
      case "negotiation_tick": {
        const layout = (ev as { layout?: ActionLayout }).layout;
        if (layout) {
          this.layout = layout;
        }
        if (t === "negotiation_tick") {
          this.negPass = Number((ev as { pass?: number }).pass ?? 0);
          this.statusLine = `Negotiate pass ${this.negPass}`;
        }
        break;
      }
      case "score_update":
        this.composite = Number((ev as { composite?: number }).composite ?? 0);
        break;
      case "terminal": {
        const sev = String((ev as { severity?: string }).severity);
        this.statusLine = `Terminal · ${sev} · composite ${this.composite.toFixed(3)}`;
        if (sev === "catastrophic") {
          this.catastrophicStart = performance.now();
        }
        break;
      }
      case "error":
        this.statusLine = `Error: ${String((ev as { detail?: string }).detail ?? "")}`;
        break;
      default:
        break;
    }
  }

  private loop = (): void => {
    this.raf = requestAnimationFrame(this.loop);
    this.decorT += 1;
    const scale = this.canvas.width / LW;
    const ctx = this.ctx;

    let shakeX = 0;
    let shakeY = 0;
    let catElapsed = 0;
    let catActive = false;
    if (this.catastrophicStart !== null) {
      catElapsed = performance.now() - this.catastrophicStart;
      const m = catastrophicMotion(catElapsed);
      shakeX = m.shakeX;
      shakeY = m.shakeY;
      catActive = true;
    }

    ctx.save();
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.translate(shakeX, shakeY);
    this.drawWorld(ctx);
    ctx.restore();

    if (catActive) {
      ctx.save();
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.imageSmoothingEnabled = false;
      drawCatastrophicOverlay(ctx, LW, LH, catElapsed);
      ctx.restore();
      if (catastrophicMotion(catElapsed).done) {
        this.catastrophicStart = null;
      }
    }
  };

  private drawWorld(ctx: CanvasRenderingContext2D): void {
    const p = this.palette;
    const g = ctx.createLinearGradient(0, 0, 0, LH);
    g.addColorStop(0, p.skyTop);
    g.addColorStop(1, p.skyBot);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, LW, LH);

    this.drawDecor(ctx, p);

    ctx.fillStyle = p.ground;
    ctx.fillRect(0, 125, LW, 55);

    const rwLabel = this.profile === "icu" ? "beds" : "runways";
    ctx.fillStyle = "#101820";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(rwLabel.toUpperCase(), 6, 118);

    const runways = this.extractRunwayIds();
    let x0 = 20;
    for (const rw of runways) {
      this.drawRunwayStrip(ctx, x0, 132, 56, 10, rw, p);
      x0 += 68;
    }

    this.drawAircraftGlyphs(ctx, p);

    ctx.fillStyle = "#0a0a12";
    ctx.fillRect(0, 0, LW, 22);
    ctx.fillStyle = p.accent;
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(this.statusLine.slice(0, 52), 6, 15);

    if (this.thinking) {
      ctx.fillStyle = "#ffff88";
      ctx.fillText(`● ${this.thinking}`, LW - 86, 15);
    }

    ctx.fillStyle = "#8898a8";
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText(`cmp ${this.composite.toFixed(2)}`, LW - 70, LH - 6);
  }

  private extractRunwayIds(): string[] {
    const t = this.taskSnapshot as { runways?: { runway_id: string }[] } | null;
    if (t?.runways?.length) {
      return t.runways.map((r) => r.runway_id);
    }
    if (this.layout) {
      const s = new Set<string>();
      for (const a of this.layout.aman_arrivals ?? []) {
        s.add(a.runway);
      }
      for (const d of this.layout.dman_departures ?? []) {
        s.add(d.runway);
      }
      return [...s].sort();
    }
    return ["RWY1"];
  }

  private drawRunwayStrip(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    p: ScenePalette,
  ): void {
    ctx.fillStyle = p.runway;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = p.runwayMark;
    ctx.lineWidth = 1;
    for (let i = 6; i < w - 4; i += 10) {
      ctx.beginPath();
      ctx.moveTo(x + i, y + h * 0.5);
      ctx.lineTo(x + i + 5, y + h * 0.5);
      ctx.stroke();
    }
    ctx.fillStyle = p.runwayMark;
    ctx.font = "8px JetBrains Mono, monospace";
    ctx.fillText(label.slice(0, 8), x + 3, y + h - 2);
  }

  private drawAircraftGlyphs(ctx: CanvasRenderingContext2D, p: ScenePalette): void {
    if (!this.layout) {
      return;
    }
    const baseY = 108;
    let i = 0;
    for (const slot of this.layout.aman_arrivals ?? []) {
      const gx = 24 + (i % 6) * 48;
      const gy = baseY - (i % 2) * 8;
      this.drawEntity(ctx, gx, gy, slot.flight_id, "arr", p);
      i++;
    }
    for (const slot of this.layout.dman_departures ?? []) {
      const gx = 24 + (i % 6) * 48;
      const gy = baseY + 10 + (i % 2) * 6;
      this.drawEntity(ctx, gx, gy, slot.flight_id, "dep", p);
      i++;
    }
  }

  private drawEntity(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    fid: string,
    kind: "arr" | "dep",
    p: ScenePalette,
  ): void {
    if (this.profile === "icu") {
      ctx.fillStyle = kind === "arr" ? "#50a0e0" : "#40c080";
      ctx.fillRect(x, y, 10, 6);
      ctx.fillStyle = "#fff";
      ctx.font = "6px JetBrains Mono, monospace";
      ctx.fillText(fid.slice(0, 3), x + 12, y + 5);
      return;
    }
    ctx.fillStyle = kind === "arr" ? p.accent : "#e0e8ff";
    ctx.beginPath();
    ctx.moveTo(x, y + 4);
    ctx.lineTo(x + 14, y + 2);
    ctx.lineTo(x + 14, y + 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#111";
    ctx.font = "6px JetBrains Mono, monospace";
    ctx.fillText(fid.slice(0, 4), x + 16, y + 5);
  }

  private drawDecor(ctx: CanvasRenderingContext2D, p: ScenePalette): void {
    const d = p.decor;
    if (d === "rain") {
      ctx.strokeStyle = "rgba(200,220,255,0.35)";
      for (let i = 0; i < 40; i++) {
        const rx = ((i * 47 + this.decorT * 3) % LW) | 0;
        const ry = ((i * 19 + this.decorT * 6) % 90) | 0;
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(rx - 2, ry + 6);
        ctx.stroke();
      }
    } else if (d === "cranes") {
      ctx.strokeStyle = "rgba(40,40,50,0.5)";
      for (let c = 0; c < 3; c++) {
        const bx = 200 + c * 36;
        ctx.beginPath();
        ctx.moveTo(bx, 120);
        ctx.lineTo(bx, 70);
        ctx.lineTo(bx + 28, 58);
        ctx.stroke();
      }
    } else if (d === "heat") {
      ctx.fillStyle = "rgba(255,200,120,0.06)";
      ctx.fillRect(0, 40, LW, 60);
    }
  }
}
