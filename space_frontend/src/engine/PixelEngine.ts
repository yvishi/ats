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
  /** Synthetic layout from ``idle_preview`` (scheduled slots) until SSE sends real actions. */
  private previewLayout: ActionLayout | null = null;

  private thinking: "AMAN" | "DMAN" | null = null;
  private negPass = 0;
  private composite = 0;
  private statusLine = "";

  private catastrophicStart: number | null = null;

  private decorT = 0;
  private radarAngle = 0;
  private blinkT = 0;

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context required");
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
    const ph = parent?.clientHeight ?? 360;
    let scaleX = Math.max(1, Math.floor(pw / LW));
    let scaleY = Math.max(1, Math.floor(ph / LH));
    let scale = Math.min(scaleX, scaleY);
    if (pw >= 720) scale = Math.max(scale, 2);
    if (pw >= 1100) scale = Math.max(scale, 3);
    scale = Math.min(5, Math.max(1, scale));
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
        this.previewLayout = null;
        this.thinking = null;
        this.negPass = 0;
        this.composite = 0;
        this.statusLine = tid;
        break;
      }
      case "idle_preview": {
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
        this.previewLayout = this._layoutFromTaskPreview(
          (ev as { task?: { flights?: unknown[] } }).task,
        );
        this.thinking = null;
        this.negPass = 0;
        this.composite = 0;
        const n = this.previewLayout
          ? (this.previewLayout.aman_arrivals?.length ?? 0) +
            (this.previewLayout.dman_departures?.length ?? 0)
          : 0;
        this.statusLine =
          this.profile === "icu"
            ? `Preview · ${n} patients (scheduled)`
            : `Preview · ${n} flights (scheduled)`;
        break;
      }
      case "idle_clear": {
        this.previewLayout = null;
        this.taskSnapshot = null;
        this.layout = null;
        this.sceneKey = "bhopal";
        this.mode = "atc";
        this.profile = "atc";
        this.palette = SCENE_PALETTES.bhopal;
        this.statusLine = "Select a scenario";
        this.composite = 0;
        break;
      }
      case "adapt_scene":
        this.mode = "domain";
        this.profile = "icu";
        this.sceneKey = "icu";
        this.palette = SCENE_PALETTES.icu;
        this.statusLine = `ADAPT · ${String(ev.domain_task_id ?? "")}`;
        break;
      case "adapt_mapping":
        this.statusLine = `Mapping · ${String(
          (ev as { rationale_preview?: string }).rationale_preview ?? "",
        ).slice(0, 55)}`;
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
          this.previewLayout = null;
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
        this.statusLine = `${sev.toUpperCase()} · score ${this.composite.toFixed(3)}`;
        if (sev === "catastrophic") this.catastrophicStart = performance.now();
        break;
      }
      case "error":
        this.statusLine = `ERR: ${String((ev as { detail?: string }).detail ?? "")}`;
        break;
      default:
        break;
    }
  }

  private loop(): void {
    this.raf = requestAnimationFrame(this.loop);
    this.decorT += 1;
    this.radarAngle = (this.radarAngle + 1.4) % 360;
    this.blinkT = (this.blinkT + 1) % 90;

    const scale = this.canvas.width / LW;
    const ctx = this.ctx;

    let shakeX = 0,
      shakeY = 0,
      catElapsed = 0,
      catActive = false;
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
      if (catastrophicMotion(catElapsed).done) this.catastrophicStart = null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // drawWorld: dispatch on profile
  // ─────────────────────────────────────────────────────────────────────────

  private drawWorld(ctx: CanvasRenderingContext2D): void {
    if (this.profile === "icu") {
      this.drawICUWorld(ctx);
    } else {
      this.drawATCWorld(ctx);
    }
    this.drawStatusBar(ctx);
    this.drawScoreOverlay(ctx);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // ATC World
  // ─────────────────────────────────────────────────────────────────────────

  private drawATCWorld(ctx: CanvasRenderingContext2D): void {
    const p = this.palette;

    // Sky gradient
    const sky = ctx.createLinearGradient(0, 0, 0, 120);
    sky.addColorStop(0, p.skyTop);
    sky.addColorStop(1, p.skyBot);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, LW, 120);

    // Scene decor
    this.drawATCDecor(ctx, p);

    // Radar in sky
    this.drawRadarSweep(ctx, p);

    // Ground
    const gnd = ctx.createLinearGradient(0, 118, 0, LH);
    gnd.addColorStop(0, p.ground);
    gnd.addColorStop(1, this.shade(p.ground, 0.72));
    ctx.fillStyle = gnd;
    ctx.fillRect(0, 118, LW, LH - 118);

    // Taxiway line
    ctx.strokeStyle = "rgba(255,200,0,0.25)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(0, 132);
    ctx.lineTo(LW, 132);
    ctx.stroke();
    ctx.setLineDash([]);

    // Runways
    const runways = this.extractRunwayIds();
    const rw = 54,
      rg = 14;
    const totalW = runways.length * rw + (runways.length - 1) * rg;
    let x0 = Math.max(14, (LW - totalW) / 2);
    for (const id of runways) {
      this.drawRunway(ctx, x0, 125, rw, 13, id, p);
      x0 += rw + rg;
    }

    // Aircraft
    this.drawAircraftGlyphs(ctx, p);
  }

  private drawRadarSweep(ctx: CanvasRenderingContext2D, p: ScenePalette): void {
    const cx = LW * 0.78,
      cy = 68,
      r = 40;
    const ang = (this.radarAngle * Math.PI) / 180;
    const sweepW = 0.65;

    ctx.save();

    // Radar circles
    ctx.strokeStyle = `${p.accent}1a`;
    ctx.lineWidth = 0.5;
    for (let ri = r / 3; ri <= r; ri += r / 3) {
      ctx.beginPath();
      ctx.arc(cx, cy, ri, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cross-hairs
    ctx.strokeStyle = `${p.accent}10`;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.stroke();

    // Sweep fan (layered sectors)
    for (let i = 0; i < 10; i++) {
      const a0 = ang - sweepW + (i / 10) * sweepW;
      const a1 = ang - sweepW + ((i + 1) / 10) * sweepW;
      ctx.globalAlpha = ((i + 1) / 10) * 0.22;
      ctx.fillStyle = p.accent;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, a0, a1);
      ctx.closePath();
      ctx.fill();
    }

    // Leading edge line
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r);
    ctx.stroke();

    // Centre dot
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = p.accent;
    ctx.beginPath();
    ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Blips from layout slots
    const lay = this._activeLayout();
    if (lay) {
      const all = [
        ...(lay.aman_arrivals ?? []),
        ...(lay.dman_departures ?? []),
      ];
      all.slice(0, 6).forEach((slot, i) => {
        const h = (slot.flight_id.charCodeAt(0) * 17 + slot.assigned_minute * 3 + i * 31) % 100;
        const bx = cx + Math.cos((h / 100) * Math.PI * 2) * (r * 0.3 + (h % 12) * 2);
        const by = cy + Math.sin((h / 100) * Math.PI * 2) * (r * 0.3 + (h % 10));
        const age = (this.decorT + i * 17) % 80;
        if (age < 25) {
          ctx.globalAlpha = ((25 - age) / 25) * 0.85;
          ctx.fillStyle = p.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    ctx.globalAlpha = 1;
    ctx.restore();
  }

  private drawRunway(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    p: ScenePalette,
  ): void {
    // Surface
    ctx.fillStyle = p.runway;
    ctx.fillRect(x, y, w, h);

    // Threshold markings
    ctx.fillStyle = p.runwayMark;
    for (let i = 0; i < 4; i++) {
      ctx.fillRect(x + 4 + i * 5, y + 2, 2, 3);
      ctx.fillRect(x + 4 + i * 5, y + h - 5, 2, 3);
    }

    // Centreline dashes
    ctx.strokeStyle = p.runwayMark;
    ctx.lineWidth = 0.6;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x + 10, y + h / 2);
    ctx.lineTo(x + w - 10, y + h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Edge highlight
    ctx.strokeStyle = `${p.runwayMark}55`;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

    // Label below runway
    ctx.fillStyle = p.runwayMark;
    ctx.font = "6px JetBrains Mono, monospace";
    ctx.fillText(label.slice(0, 6), x + 3, y + h + 7);
  }

  private drawAircraftGlyphs(ctx: CanvasRenderingContext2D, p: ScenePalette): void {
    const lay = this._activeLayout();
    if (!lay) return;
    const arrivals = lay.aman_arrivals ?? [];
    const departures = lay.dman_departures ?? [];
    const total = arrivals.length + departures.length;
    if (total === 0) return;

    const spacing = Math.min(44, (LW - 28) / total);
    let i = 0;

    for (const slot of arrivals) {
      const gx = 14 + i * spacing;
      const gy = 108 - (i % 2) * 9;
      this.drawAircraft(ctx, gx, gy, slot.flight_id, "arr", p);
      i++;
    }
    for (const slot of departures) {
      const gx = 14 + i * spacing;
      const gy = 114 + (i % 2) * 7;
      this.drawAircraft(ctx, gx, gy, slot.flight_id, "dep", p);
      i++;
    }
  }

  private drawAircraft(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    fid: string,
    kind: "arr" | "dep",
    p: ScenePalette,
  ): void {
    const isArr = kind === "arr";
    ctx.save();
    ctx.translate(x, y);

    // Fuselage
    ctx.fillStyle = isArr ? p.accent : "#dde8ff";
    ctx.beginPath();
    if (isArr) {
      ctx.moveTo(14, 4);
      ctx.lineTo(0, 1);
      ctx.lineTo(0, 7);
      ctx.closePath();
    } else {
      ctx.moveTo(0, 4);
      ctx.lineTo(14, 1);
      ctx.lineTo(14, 7);
      ctx.closePath();
    }
    ctx.fill();

    // Wings
    const wc = isArr ? this.shade(p.accent, 0.65) : "#8898b8";
    ctx.fillStyle = wc;
    const wx = isArr ? 4 : 4;
    ctx.fillRect(wx, 0, 5, 2);
    ctx.fillRect(wx, 6, 5, 2);

    // Nav blink
    const blink = this.blinkT < 9 || (this.blinkT > 32 && this.blinkT < 41);
    if (blink) {
      ctx.fillStyle = isArr ? "#ff3030" : "#30ff80";
      ctx.fillRect(isArr ? 11 : 1, 3, 2, 2);
    }

    // Flight ID
    ctx.fillStyle = "rgba(200,220,245,0.85)";
    ctx.font = "5px JetBrains Mono, monospace";
    ctx.fillText(fid.slice(0, 4), 0, 14);

    ctx.restore();
  }

  private drawATCDecor(ctx: CanvasRenderingContext2D, p: ScenePalette): void {
    const d = p.decor;
    if (d === "rain") {
      ctx.strokeStyle = "rgba(180,210,255,0.3)";
      ctx.lineWidth = 0.7;
      for (let i = 0; i < 38; i++) {
        const rx = ((i * 51 + this.decorT * 3) % LW) | 0;
        const ry = ((i * 23 + this.decorT * 6) % 90) | 0;
        ctx.beginPath();
        ctx.moveTo(rx, ry + 20);
        ctx.lineTo(rx - 1, ry + 26);
        ctx.stroke();
      }
    } else if (d === "cranes") {
      ctx.strokeStyle = "rgba(30,35,50,0.5)";
      ctx.lineWidth = 1;
      for (let c = 0; c < 3; c++) {
        const bx = 188 + c * 34;
        ctx.beginPath();
        ctx.moveTo(bx, 120);
        ctx.lineTo(bx, 68);
        ctx.lineTo(bx + 26, 57);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx, 68);
        ctx.lineTo(bx - 4, 120);
        ctx.stroke();
        // Hook line
        ctx.beginPath();
        ctx.moveTo(bx + 14, 57);
        ctx.lineTo(bx + 14, 80);
        ctx.stroke();
      }
    } else if (d === "heat") {
      ctx.fillStyle = "rgba(255,170,60,0.04)";
      ctx.fillRect(0, 28, LW, 80);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // ICU World
  // ─────────────────────────────────────────────────────────────────────────

  private drawICUWorld(ctx: CanvasRenderingContext2D): void {
    const p = this.palette;

    // Hospital background
    ctx.fillStyle = "#070c18";
    ctx.fillRect(0, 0, LW, LH);

    // Subtle grid lines
    ctx.strokeStyle = "rgba(18,48,78,0.35)";
    ctx.lineWidth = 0.5;
    for (let gx = 0; gx < LW; gx += 20) {
      ctx.beginPath();
      ctx.moveTo(gx, 22);
      ctx.lineTo(gx, LH - 2);
      ctx.stroke();
    }
    for (let gy = 22; gy < LH; gy += 20) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(LW, gy);
      ctx.stroke();
    }

    // Ambient glow
    const glow = ctx.createRadialGradient(LW / 2, LH / 2, 10, LW / 2, LH / 2, 110);
    glow.addColorStop(0, "rgba(0,70,120,0.1)");
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, LW, LH);

    // Ward label
    ctx.fillStyle = "#1e4060";
    ctx.font = "7px JetBrains Mono, monospace";
    ctx.fillText("ICU WARD", 6, 33);

    // Central monitor line
    ctx.strokeStyle = "rgba(0,180,120,0.15)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, LH - 28);
    ctx.lineTo(LW, LH - 28);
    ctx.stroke();

    // Beds
    const beds = this.extractRunwayIds();
    const maxPerRow = Math.min(beds.length, 4);
    const bedW = 60,
      bedH = 32,
      padX = 8,
      padY = 10;
    const gridW = maxPerRow * bedW + (maxPerRow - 1) * padX;
    const startX = Math.max(6, (LW - gridW) / 2);
    const startY = 38;

    beds.forEach((bedId, i) => {
      const row = Math.floor(i / maxPerRow);
      const col = i % maxPerRow;
      const bx = startX + col * (bedW + padX);
      const by = startY + row * (bedH + padY);
      const arrN = this._activeLayout()?.aman_arrivals?.length ?? 0;
      const hasPatient = arrN > 0 && i < arrN;
      this.drawBedCell(ctx, bx, by, bedW, bedH, bedId, hasPatient, p);
    });

    // Patient queue below beds
    this.drawPatientQueue(ctx, p, startX, startY, beds.length, maxPerRow, bedW, bedH, padX, padY);
  }

  private drawBedCell(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    occupied: boolean,
    _p: ScenePalette,
  ): void {
    // Frame
    ctx.fillStyle = occupied ? "#0c1c30" : "#080f1c";
    ctx.strokeStyle = occupied ? "#1e5080" : "#102030";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();

    // Bed surface
    ctx.fillStyle = occupied ? "#142438" : "#0a1520";
    ctx.fillRect(x + 3, y + Math.floor(h * 0.46), w - 6, Math.floor(h * 0.44));

    if (occupied) {
      // Patient silhouette
      ctx.fillStyle = "#2870b8";
      ctx.fillRect(x + 4, y + Math.floor(h * 0.52), w - 18, Math.floor(h * 0.28));

      // Pillow
      ctx.fillStyle = "#c8dff0";
      ctx.fillRect(x + w - 12, y + Math.floor(h * 0.5), 8, Math.floor(h * 0.3));

      // IV drip indicator
      ctx.strokeStyle = "rgba(100,200,255,0.5)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(x + w - 4, y + 3);
      ctx.lineTo(x + w - 4, y + h - 5);
      ctx.stroke();
    }

    // ECG waveform
    ctx.strokeStyle = occupied
      ? "rgba(0,220,120,0.75)"
      : "rgba(0,120,80,0.35)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    const ecgY = y + 10;
    const ecgSteps = Math.floor((w - 6) / 1.4);
    for (let s = 0; s <= ecgSteps; s++) {
      const ex = x + 3 + s * 1.4;
      const phase = s * 0.38 + this.decorT * 0.06;
      const mod = phase % (Math.PI * 2);
      let ey: number;
      if (mod > 1.7 && mod < 1.9) ey = ecgY - 5;
      else if (mod > 1.9 && mod < 2.1) ey = ecgY + 3;
      else ey = ecgY + Math.sin(phase * 0.22) * 1.3;
      if (s === 0) ctx.moveTo(ex, ey);
      else ctx.lineTo(ex, ey);
    }
    ctx.stroke();

    // Bed label
    ctx.fillStyle = occupied ? "#3a80b8" : "#1a3850";
    ctx.font = "5px JetBrains Mono, monospace";
    const shortLabel = label.length > 5 ? label.slice(0, 5) : label;
    ctx.fillText(shortLabel, x + 3, y + h - 2);

    // Occupied dot
    if (occupied) {
      ctx.fillStyle = "#00e87a";
      ctx.beginPath();
      ctx.arc(x + w - 4, y + 4, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private drawPatientQueue(
    ctx: CanvasRenderingContext2D,
    _p: ScenePalette,
    startX: number,
    startY: number,
    bedCount: number,
    maxPerRow: number,
    _bedW: number,
    bedH: number,
    _padX: number,
    padY: number,
  ): void {
    const lay = this._activeLayout();
    if (!lay) return;
    const entities = [
      ...(lay.dman_departures ?? []),
    ];
    if (entities.length === 0) return;

    const rows = Math.ceil(bedCount / maxPerRow);
    const queueY = startY + rows * (bedH + padY) + 4;
    if (queueY > LH - 14) return;

    // Queue label
    ctx.fillStyle = "#1e4060";
    ctx.font = "6px JetBrains Mono, monospace";
    ctx.fillText("PENDING", startX, queueY + 7);

    entities.slice(0, 8).forEach((slot, i) => {
      const ex = startX + 44 + i * 28;
      if (ex > LW - 16) return;
      ctx.fillStyle = "#1a5080";
      ctx.strokeStyle = "#2a7ab8";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.rect(ex, queueY, 22, 10);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#6ab0e0";
      ctx.font = "5px JetBrains Mono, monospace";
      ctx.fillText(slot.flight_id.slice(0, 4), ex + 2, queueY + 7);
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Shared overlays
  // ─────────────────────────────────────────────────────────────────────────

  private drawStatusBar(ctx: CanvasRenderingContext2D): void {
    const p = this.palette;

    ctx.fillStyle = "rgba(4,6,12,0.93)";
    ctx.fillRect(0, 0, LW, 20);

    // Bottom border
    const borderColor = this.profile === "icu" ? "#1a4060" : `${p.accent}35`;
    ctx.fillStyle = borderColor;
    ctx.fillRect(0, 20, LW, 1);

    // Profile indicator chip
    ctx.fillStyle = this.profile === "icu" ? "#0e2840" : "#0a1e30";
    ctx.strokeStyle = this.profile === "icu" ? "#2a5a80" : `${p.accent}50`;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(3, 3, 24, 14);
    ctx.fillRect(3, 3, 24, 14);
    ctx.fillStyle = this.profile === "icu" ? "#40a0d0" : p.accent;
    ctx.font = "6px JetBrains Mono, monospace";
    ctx.fillText(this.profile.toUpperCase(), 6, 13);

    // Status text
    ctx.fillStyle = this.profile === "icu" ? "#50a8d8" : p.accent;
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillText(this.statusLine.slice(0, 36), 32, 13);

    // Thinking indicator
    if (this.thinking) {
      const blink = this.blinkT < 45;
      ctx.fillStyle = blink ? "#ffdd44" : "#664400";
      ctx.fillText(`● ${this.thinking}`, LW - 60, 13);
    }
  }

  private drawScoreOverlay(ctx: CanvasRenderingContext2D): void {
    if (this.composite <= 0) return;
    const col =
      this.composite >= 0.7
        ? "#00e87a"
        : this.composite >= 0.4
          ? "#ffaa30"
          : "#ff3355";
    ctx.fillStyle = "rgba(4,6,12,0.7)";
    ctx.fillRect(LW - 44, LH - 14, 44, 14);
    ctx.fillStyle = col;
    ctx.font = "8px JetBrains Mono, monospace";
    ctx.fillText(this.composite.toFixed(3), LW - 40, LH - 4);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Utilities
  // ─────────────────────────────────────────────────────────────────────────

  private _activeLayout(): ActionLayout | null {
    return this.layout ?? this.previewLayout;
  }

  /** Build AMAN/DMAN slot lists from scheduled times (idle preview). */
  private _layoutFromTaskPreview(task: unknown): ActionLayout | null {
    if (!task || typeof task !== "object") return null;
    const flights = (task as { flights?: unknown[] }).flights;
    if (!Array.isArray(flights) || flights.length === 0) return null;
    const aman_arrivals: LayoutSlot[] = [];
    const dman_departures: LayoutSlot[] = [];
    for (const raw of flights) {
      if (!raw || typeof raw !== "object") continue;
      const f = raw as Record<string, unknown>;
      const op = String(f.operation ?? "").toLowerCase();
      const runway = Array.isArray(f.runways) && f.runways.length
        ? String(f.runways[0])
        : "RWY1";
      const slot: LayoutSlot = {
        flight_id: String(f.flight_id ?? "?"),
        runway,
        assigned_minute: Number(f.scheduled ?? f.earliest ?? 0),
        hold_minutes: 0,
      };
      if (op === "arrival") aman_arrivals.push(slot);
      else if (op === "departure") dman_departures.push(slot);
    }
    if (aman_arrivals.length === 0 && dman_departures.length === 0) return null;
    return { aman_arrivals, dman_departures };
  }

  private extractRunwayIds(): string[] {
    const t = this.taskSnapshot as { runways?: { runway_id: string }[] } | null;
    if (t?.runways?.length) return t.runways.map((r) => r.runway_id);
    const lay = this._activeLayout();
    if (lay) {
      const s = new Set<string>();
      for (const a of lay.aman_arrivals ?? []) s.add(a.runway);
      for (const d of lay.dman_departures ?? []) s.add(d.runway);
      return [...s].sort();
    }
    return this.profile === "icu" ? ["BED-1", "BED-2", "BED-3"] : ["RWY1"];
  }

  private shade(hex: string, f: number): string {
    const n = parseInt(hex.replace("#", ""), 16);
    if (isNaN(n)) return hex;
    const r = Math.round(((n >> 16) & 0xff) * f);
    const g = Math.round(((n >> 8) & 0xff) * f);
    const b = Math.round((n & 0xff) * f);
    return `rgb(${r},${g},${b})`;
  }
}
