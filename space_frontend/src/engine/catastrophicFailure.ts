/**
 * Single shared full-screen FX for `terminal.severity === "catastrophic"`.
 */

const DURATION_MS = 2800;

export type CatastrophicMotion = {
  shakeX: number;
  shakeY: number;
  done: boolean;
};

export function catastrophicMotion(elapsedMs: number): CatastrophicMotion {
  const t = Math.min(1, elapsedMs / DURATION_MS);
  const shakeAmp = Math.max(0, 6 * (1 - t * 1.2));
  const shakeX = Math.round(
    (Math.sin(elapsedMs * 0.09) + Math.sin(elapsedMs * 0.031)) * shakeAmp,
  );
  const shakeY = Math.round(
    (Math.cos(elapsedMs * 0.084) + Math.cos(elapsedMs * 0.029)) * shakeAmp,
  );
  return { shakeX, shakeY, done: elapsedMs >= DURATION_MS };
}

/** Full-screen vignette + pixel burst (call in logical pixel space, 320×180). */
export function drawCatastrophicOverlay(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  elapsedMs: number,
): void {
  const t = Math.min(1, elapsedMs / DURATION_MS);
  const phase = t < 0.15 ? t / 0.15 : t < 0.55 ? (t - 0.15) / 0.4 : (t - 0.55) / 0.45;
  const overlayAlpha = t < 0.2 ? t * 2.2 : t < 0.65 ? 0.55 + phase * 0.25 : Math.max(0, 0.85 - (t - 0.65) * 2.2);

  const g = ctx.createRadialGradient(
    w * 0.5,
    h * 0.48,
    Math.max(w, h) * 0.05,
    w * 0.5,
    h * 0.5,
    Math.max(w, h) * 0.85,
  );
  g.addColorStop(0, `rgba(255,80,40,${0.12 * overlayAlpha})`);
  g.addColorStop(0.55, `rgba(120,0,20,${0.55 * overlayAlpha})`);
  g.addColorStop(1, `rgba(10,0,0,${0.92 * overlayAlpha})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  const seed = Math.floor(elapsedMs / 40);
  const n = 48;
  for (let i = 0; i < n; i++) {
    const a = ((i * 997 + seed * 13) % 1000) / 1000;
    const b = ((i * 541 + seed * 7) % 1000) / 1000;
    const ang = a * Math.PI * 2;
    const dist = (b * 0.4 + t * 0.55) * Math.max(w, h) * 0.55;
    const px = Math.floor(w * 0.5 + Math.cos(ang) * dist * (0.3 + t));
    const py = Math.floor(h * 0.45 + Math.sin(ang) * dist * (0.25 + t));
    const sz = 2 + ((i * 3) % 4);
    ctx.fillStyle = i % 3 === 0 ? "#ff4444" : i % 3 === 1 ? "#ffcc00" : "#ffffff";
    ctx.globalAlpha = Math.max(0, 1 - t * 0.95);
    ctx.fillRect(px, py, sz, sz);
  }
  ctx.globalAlpha = 1;
}
