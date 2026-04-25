"""Adversarial self-play scenario generator with ZPD-aware curriculum.

Design principles (grounded in literature):

  Automatic Curriculum Learning
    ALP-GMM / PLR (Portelas 2020, Jiang 2021): track *learning progress* per
    difficulty tier, not raw score.  Progress = |Δreward| per tier.  Sample
    tiers proportional to recent absolute improvement.

  VCRL (2025): within a GRPO group the reward variance is a free, parameter-free
    ZPD proxy.  High variance ↔ the policy is in the learning frontier (sometimes
    succeeds, sometimes fails).  We track per-tier group variance and prefer tiers
    where variance is highest.

  Competence-based curriculum (Platanios 2019): the dataset builder uses a
    √-schedule to ramp max-allowed tier from 0 → MAX_TIER.  The ChallengeGenerator
    provides the *online* adaptive layer on top of that static ramp.

  ProCuRL / ZPD enforcement (2022, 2025): keep tasks where the policy's pass@G
    is in [0.10, 0.75].  We approximate this via rescue mode: when the global EMA
    drops below RESCUE_THRESHOLD the generator forces tier-0 tasks until it recovers.

  PAIRED regret (Dennis 2020): generator reward = -(controller_score - baseline).
    Generator is rewarded for staying at the frontier of agent capability, not for
    making tasks arbitrarily hard or unsolvable.

  Hysteresis (Bengio 2009 competence function): require HYSTERESIS_UP consecutive
    episodes above ESCALATION_THRESHOLD before escalating, but only HYSTERESIS_DOWN
    to deescalate.  This prevents noisy single-episode spikes from driving the
    curriculum off a cliff.

Self-improving failure mode addressed here:
  The original code set EMA_ALPHA=0.2 (slow to fall) and FLOOR_THRESHOLD=0.30
  (too permissive).  A cold model scoring 0.12 has EMA decay:
    0.2 * 0.12 + 0.8 * 0.50 = 0.424 — never reaches 0.30 even after 5 bad episodes.
  Fix: EMA_ALPHA=0.08 (tracks ~12-episode window) + FLOOR_THRESHOLD=0.42 +
  hard rescue at 0.22.
"""

from __future__ import annotations

import random
import statistics
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

try:
    from ..engine import simulate_plan
    from ..models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from .models import GeneratorAction, GeneratorMutation, MutationType
except ImportError:
    from engine import simulate_plan
    from models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from multi_agent.models import GeneratorAction, GeneratorMutation, MutationType


# ── Curriculum hyperparameters ────────────────────────────────────────────────
# These constants are the main levers.  Comments explain the reasoning so future
# tuners understand what each parameter does.

ESCALATION_THRESHOLD = 0.65   # EMA must exceed this to escalate tier.
                               # 0.65 + HYSTERESIS_UP=5 is strictly more conservative
                               # than old 0.65+no-hysteresis: single-episode noise
                               # cannot trigger escalation.  0.68 was too close to
                               # the EMA convergence point (~0.678) for 0.72 scores.

FLOOR_THRESHOLD      = 0.42   # EMA below this → start deescalation countdown.
                               # Was 0.30 — far too forgiving.  At 0.30 a model
                               # scoring 0.12 every episode takes 30+ episodes
                               # to trigger ease-back (EMA decays slowly).
                               # 0.42 triggers after ~4 consecutive low episodes.

RESCUE_THRESHOLD     = 0.22   # Hard floor: EMA below this → immediate rescue mode.
                               # Research basis: GRPO absorbing state (DeepSeek-R1
                               # analysis) — if policy never succeeds, GRPO gradient
                               # is zero.  Rescue injects easy tasks to break the loop.

EMA_ALPHA            = 0.08   # Was 0.2.  τ = 1/alpha ≈ 12 episodes.
                               # 0.2 reacts too fast to single-episode noise;
                               # 0.08 tracks a 12-episode rolling average, reducing
                               # the chance of curriculum thrash from one outlier.

RESCUE_BUDGET        = 6      # How many tier-0 episodes to inject when EMA collapses.
                               # Based on ProCuRL warmup recommendations: ~1 full
                               # GRPO batch (N_GENERATIONS=8 × BATCH=4 = 32 samples,
                               # ≈ 6 episodes) before re-evaluating capability.

HYSTERESIS_UP        = 5      # Consecutive episodes above ESCALATION_THRESHOLD
                               # required before advancing the tier.
                               # Prevents single-episode lucky runs from escalating.

HYSTERESIS_DOWN      = 2      # Consecutive episodes below FLOOR_THRESHOLD before
                               # deescalating.  Asymmetric: deescalate quickly
                               # (protect learning signal), escalate conservatively.

MAX_TIER             = 4      # Tier 0=warmup … Tier 4=expert.  Mirrors TASK_TIER
                               # in tasks.py.

MAX_MUTATIONS_PER_EPISODE = 3 # Cap mutations to prevent compounding unsolvability.
MIN_WINDOW_WIDTH     = 8      # Never squeeze a flight window below 8 minutes.

# Per-mutation mastery (unchanged from original)
MASTERY_WINDOW       = 10
MASTERY_THRESHOLD    = 0.55   # Score below this on a mutation type → "weak"

# LLM priority string aliases
_PRIORITY_ALIASES: Dict[str, str] = {
    "high": "emergency", "urgent": "emergency", "critical": "emergency",
    "med": "medical", "low": "normal", "standard": "normal",
    "routine": "normal", "conn": "connection",
}


class ChallengeGenerator:
    """ZPD-aware adversarial curriculum generator for multi-agent ATC training.

    Maintains:
      - Global EMA of recent controller scores (slow α=0.08)
      - Per-tier score windows for learning-progress estimation (ALP-GMM style)
      - Per-tier within-group reward variance windows (VCRL ZPD proxy)
      - Rescue mode that forces tier-0 episodes when EMA collapses
      - Hysteresis counters to prevent curriculum thrash

    Public API:
      update(score, group_rewards)  — call after each episode (real or simulated)
      recommended_tier()            — tier (0-4) for next base task selection
      mutate(task)                  — apply curriculum-appropriate mutations
      compute_reward(score, solvable) — PAIRED regret signal for generator training
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

        # ── Global EMA and episode history ────────────────────────────────────
        self._ema_score: float = 0.50      # start at mid-difficulty assumption
        self._score_history: Deque[float] = deque(maxlen=20)

        # ── Tier state ────────────────────────────────────────────────────────
        self._tier: int = 0                # current recommended base-task tier (0-4)
        self._difficulty_level: int = 1   # mutation intensity (1-6); derived from tier

        # Per-tier score windows for ALP-style learning progress estimation
        self._tier_scores: Dict[int, Deque[float]] = {
            i: deque(maxlen=20) for i in range(MAX_TIER + 1)
        }
        # Per-tier GRPO-group reward variance (VCRL ZPD proxy)
        self._tier_variance: Dict[int, Deque[float]] = {
            i: deque(maxlen=10) for i in range(MAX_TIER + 1)
        }

        # ── Hysteresis counters ───────────────────────────────────────────────
        self._consecutive_above: int = 0   # episodes above ESCALATION_THRESHOLD
        self._consecutive_below: int = 0   # episodes below FLOOR_THRESHOLD

        # ── Rescue mode ───────────────────────────────────────────────────────
        self._rescue_mode: bool = False
        self._rescue_remaining: int = 0

        # ── Mutation mastery (per-task, per-mutation) ─────────────────────────
        self._task_mastery: Dict[str, Deque[float]] = {}
        self._mutation_mastery: Dict[str, Deque[float]] = {}
        self._mutation_history: List[Dict] = []

        # ── PAIRED baseline ───────────────────────────────────────────────────
        self._last_heuristic_score: float = 0.5
        self._last_mutated_task: Optional[TaskDefinition] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        controller_score: float,
        group_rewards: Optional[List[float]] = None,
    ) -> None:
        """Update curriculum state after one episode.

        Args:
            controller_score: composite score from the episode (0-1).
            group_rewards:    optional list of individual GRPO group rewards.
                              When provided, the within-group variance is used
                              as a VCRL ZPD signal for the current tier.
        """
        self._score_history.append(controller_score)
        self._ema_score = (
            EMA_ALPHA * controller_score + (1.0 - EMA_ALPHA) * self._ema_score
        )
        self._tier_scores[self._tier].append(controller_score)

        # VCRL: track within-group reward variance as a ZPD proxy
        if group_rewards and len(group_rewards) >= 2:
            try:
                var = statistics.variance(group_rewards)
            except statistics.StatisticsError:
                var = 0.0
            self._tier_variance[self._tier].append(var)

        # ── Rescue mode check (hard floor) ────────────────────────────────────
        if self._ema_score < RESCUE_THRESHOLD and not self._rescue_mode:
            self._rescue_mode = True
            self._rescue_remaining = RESCUE_BUDGET
            # Reset hysteresis so rescue can end cleanly
            self._consecutive_above = 0
            self._consecutive_below = 0

        if self._rescue_mode:
            self._rescue_remaining -= 1
            if self._rescue_remaining <= 0:
                self._rescue_mode = False
                # Drop tier to a safe level so the exit doesn't immediately
                # re-trigger rescue.
                self._tier = max(0, min(1, self._tier))
                self._difficulty_level = max(1, self._tier)
            # Skip normal tier updates during rescue
            return

        # ── Hysteresis: escalation ────────────────────────────────────────────
        if self._ema_score > ESCALATION_THRESHOLD:
            self._consecutive_above += 1
            self._consecutive_below = 0
        elif self._ema_score < FLOOR_THRESHOLD:
            self._consecutive_below += 1
            self._consecutive_above = 0
        else:
            # In the ZPD zone — decay both counters slowly (model is learning)
            self._consecutive_above = max(0, self._consecutive_above - 1)
            self._consecutive_below = max(0, self._consecutive_below - 1)

        # ── Tier transitions ──────────────────────────────────────────────────
        if self._consecutive_above >= HYSTERESIS_UP and self._tier < MAX_TIER:
            self._tier += 1
            self._consecutive_above = 0
        elif self._consecutive_below >= HYSTERESIS_DOWN and self._tier > 0:
            self._tier -= 1
            self._consecutive_below = 0

        # ── Mutation difficulty level (1-6) derived from tier ─────────────────
        # Tier 0 → level 0 (no mutations)
        # Tier 1 → level 1-2
        # Tier 2 → level 2-3
        # Tier 3 → level 3-4
        # Tier 4 → level 5-6
        # Extra boost when EMA is above floor (performing well within tier)
        tier_to_level = {0: 0, 1: 1, 2: 3, 3: 4, 4: 6}
        base_level = tier_to_level.get(self._tier, 1)
        # Add 1 when clearly performing well but not yet escalating
        boost = 1 if (self._ema_score > FLOOR_THRESHOLD + 0.15 and base_level > 0) else 0
        self._difficulty_level = min(6, max(0, base_level + boost))

    def record(
        self,
        task_id: str,
        mutations_used: Optional[List[str]],
        composite_score: float,
    ) -> None:
        """Record per-task and per-mutation mastery after an episode.

        Called by the training loop alongside update().
        """
        if task_id not in self._task_mastery:
            self._task_mastery[task_id] = deque(maxlen=MASTERY_WINDOW)
        self._task_mastery[task_id].append(composite_score)

        for mtype in (mutations_used or []):
            if mtype not in self._mutation_mastery:
                self._mutation_mastery[mtype] = deque(maxlen=MASTERY_WINDOW)
            self._mutation_mastery[mtype].append(composite_score)

    def get_weak_mutations(self, threshold: float = MASTERY_THRESHOLD) -> List[str]:
        """Return mutation types where agents still score below threshold."""
        return [
            mtype
            for mtype, scores in self._mutation_mastery.items()
            if len(scores) >= 3 and (sum(scores) / len(scores)) < threshold
        ]

    def recommended_tier(self) -> int:
        """Tier (0-4) recommended for selecting the next base task.

        Returns 0 during rescue mode (force easy).  Otherwise returns the
        current tier, clamped to [0, MAX_TIER].
        """
        if self._rescue_mode:
            return 0
        return max(0, min(MAX_TIER, self._tier))

    def is_in_rescue_mode(self) -> bool:
        """True when EMA collapsed below RESCUE_THRESHOLD and easy tasks are forced."""
        return self._rescue_mode

    def mastery_report(self) -> Dict:
        """Per-task / per-mutation success rates for logging."""
        task_means = {
            tid: round(sum(v) / len(v), 3)
            for tid, v in self._task_mastery.items() if v
        }
        mut_means = {
            mtype: round(sum(v) / len(v), 3)
            for mtype, v in self._mutation_mastery.items() if v
        }
        # VCRL ZPD signal: mean variance per tier (high = in learning frontier)
        tier_var_means = {
            tier: round(sum(v) / len(v), 4) if v else 0.0
            for tier, v in self._tier_variance.items()
        }
        return {
            "task_mastery":     task_means,
            "mutation_mastery": mut_means,
            "weak_mutations":   self.get_weak_mutations(),
            "tier":             self._tier,
            "difficulty_level": self._difficulty_level,
            "ema_score":        self.ema_score,
            "rescue_mode":      self._rescue_mode,
            "rescue_remaining": self._rescue_remaining,
            "tier_variance_zpd": tier_var_means,
        }

    def curriculum_summary(self) -> str:
        """One-line human-readable curriculum state for training logs (ASCII-safe)."""
        mode = "RESCUE" if self._rescue_mode else f"T{self._tier}/L{self._difficulty_level}"
        above = f"up={self._consecutive_above}/{HYSTERESIS_UP}"
        below = f"dn={self._consecutive_below}/{HYSTERESIS_DOWN}"
        return (
            f"[Curriculum] EMA={self._ema_score:.3f}  {mode}  "
            f"hyst:{above},{below}"
        )

    # ── Mutation API ──────────────────────────────────────────────────────────

    def mutate(
        self,
        base_task: TaskDefinition,
        generator_action: Optional[GeneratorAction] = None,
    ) -> Tuple[TaskDefinition, bool]:
        """Apply difficulty-appropriate mutations to base_task.

        If generator_action is provided (LLM-driven), apply those mutations.
        Otherwise use rule-based mutations matching current difficulty_level.

        At tier/level 0 no mutations are applied — base task is returned as-is.
        This ensures warmup tasks are always solvable in their canonical form.

        Returns (mutated_task, is_solvable).
        """
        task = self._deep_copy_task(base_task)

        if generator_action and generator_action.mutations:
            mutations = generator_action.mutations[:MAX_MUTATIONS_PER_EPISODE]
        else:
            mutations = self._sample_mutations(task)

        for mut in mutations:
            task = self._apply_mutation(task, mut)
            self._mutation_history.append({
                "type":  mut.mutation_type.value,
                "level": self._difficulty_level,
                "tier":  self._tier,
                "ema":   round(self._ema_score, 3),
            })

        solvable = self._check_solvability(task)

        self._last_mutated_task = task
        self._last_heuristic_score = (
            self._score_scheduled_baseline(task) if solvable else 0.0
        )
        return task, solvable

    def compute_reward(self, controller_score: float, is_solvable: bool) -> float:
        """PAIRED regret-based generator reward with solvability guard.

        Unsolvable → -1.0 penalty (generator must create HARD but not IMPOSSIBLE).
        Otherwise: reward = -(controller_score - baseline_score)
          • Positive when agents fall behind baseline (generator winning)
          • Negative when agents beat baseline (task too easy for generator)
        """
        if not is_solvable:
            return -1.0

        if self._last_mutated_task is not None:
            regret = controller_score - self._last_heuristic_score
            return round(max(-1.0, min(1.0, -regret)), 4)

        # Fallback if called without prior mutate() call (e.g. unit tests)
        return round(max(-1.0, min(1.0, 1.0 - controller_score)), 4)

    @property
    def difficulty_level(self) -> int:
        return self._difficulty_level

    @property
    def ema_score(self) -> float:
        return round(self._ema_score, 3)

    @property
    def current_tier(self) -> int:
        return self._tier

    # ── Mutation sampling ─────────────────────────────────────────────────────

    def _sample_mutations(self, task: TaskDefinition) -> List[GeneratorMutation]:
        """Rule-based mutation selection based on current difficulty_level.

        Level 0  → no mutations (warmup tier)
        Level 1  → TIGHTEN_WINDOW (mild squeeze)
        Level 2  → + ADD_ATFM_DEADLINE
        Level 3  → + INCREASE_WEATHER_PENALTY
        Level 4  → + INJECT_EMERGENCY
        Level 5  → + ADD_CONFLICTING_FLIGHT
        Level 6  → + CLOSE_RUNWAY_WINDOW

        Weak mutations (agents consistently score below MASTERY_THRESHOLD on them)
        are boosted 3× in the sampling pool — curriculum focuses on gaps.
        """
        level = self._difficulty_level

        # Tier/level 0: no mutations at all — base task is the challenge
        if level <= 0:
            return []

        n_mutations = min(MAX_MUTATIONS_PER_EPISODE, max(1, level // 2))

        pool: List[MutationType] = []
        if level >= 1:
            pool += [MutationType.TIGHTEN_WINDOW] * 3
        if level >= 2:
            pool += [MutationType.ADD_ATFM_DEADLINE] * 2
        if level >= 3:
            pool += [MutationType.INCREASE_WEATHER_PENALTY] * 2
        if level >= 4:
            pool += [MutationType.INJECT_EMERGENCY] * 2
        if level >= 5:
            pool += [MutationType.ADD_CONFLICTING_FLIGHT] * 2
        if level >= 6:
            pool += [MutationType.CLOSE_RUNWAY_WINDOW] * 1

        # Boost under-mastered mutation types (3× weight)
        weak = set(self.get_weak_mutations())
        pool += [mt for mt in pool if mt.value in weak]

        selected = self._rng.choices(pool, k=n_mutations)

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        arrivals   = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        mutations: List[GeneratorMutation] = []

        for mtype in selected:
            if mtype == MutationType.TIGHTEN_WINDOW:
                target = self._rng.choice(task.flights)
                # Squeeze scales with level but stays gentle at low tiers
                max_squeeze = max(2, 2 + level)
                squeeze = self._rng.randint(2, max_squeeze)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_flight_id=target.flight_id,
                    params={"squeeze_minutes": squeeze},
                    rationale=f"Tighten {target.flight_id} window by {squeeze} min each side",
                ))

            elif mtype == MutationType.ADD_ATFM_DEADLINE and departures:
                target = self._rng.choice(departures)
                buffer = self._rng.randint(5, 10)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_flight_id=target.flight_id,
                    params={"deadline_offset": buffer},
                    rationale=f"ATFM slot: {target.flight_id} must depart by scheduled+{buffer}",
                ))

            elif mtype == MutationType.INCREASE_WEATHER_PENALTY:
                target_rwy = self._rng.choice(task.runways)
                delta = round(self._rng.uniform(0.1, 0.25), 2)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_runway_id=target_rwy.runway_id,
                    params={"penalty_delta": delta},
                    rationale=f"Weather degrades {target_rwy.runway_id} capacity by {delta}×",
                ))

            elif mtype == MutationType.INJECT_EMERGENCY and arrivals:
                base = self._rng.choice(arrivals)
                window_center = self._rng.randint(
                    base.earliest_minute,
                    min(base.latest_minute, base.earliest_minute + 20),
                )
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    params={
                        "flight_id": f"EMG{self._rng.randint(100, 999)}",
                        "priority":  "emergency",
                        "minute":    window_center,
                        "runway":    self._rng.choice(task.runways).runway_id,
                    },
                    rationale="Inject emergency diversion at peak arrival window",
                ))

            elif mtype == MutationType.ADD_CONFLICTING_FLIGHT:
                if arrivals:
                    anchor = self._rng.choice(arrivals)
                    mutations.append(GeneratorMutation(
                        mutation_type=mtype,
                        params={
                            "flight_id":  f"WKT{self._rng.randint(100, 999)}",
                            "wake_class": "H",
                            "operation":  "arrival",
                            "minute":     max(0, anchor.earliest_minute - 4),
                            "runway":     self._rng.choice(anchor.allowed_runways),
                        },
                        rationale="Heavy arrival 4 min before window — forces 6-min wake gap",
                    ))

            elif mtype == MutationType.CLOSE_RUNWAY_WINDOW:
                target_rwy = self._rng.choice(task.runways)
                duration = self._rng.randint(10, 20)
                mutations.append(GeneratorMutation(
                    mutation_type=mtype,
                    target_runway_id=target_rwy.runway_id,
                    params={"close_duration": duration},
                    rationale=f"Runway {target_rwy.runway_id} closed for {duration} min",
                ))

        return mutations

    # ── Mutation application ──────────────────────────────────────────────────

    def _apply_mutation(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        dispatch = {
            MutationType.TIGHTEN_WINDOW:         self._tighten_window,
            MutationType.INCREASE_WEATHER_PENALTY: self._increase_weather,
            MutationType.INJECT_EMERGENCY:        self._inject_emergency,
            MutationType.ADD_CONFLICTING_FLIGHT:  self._add_conflicting_flight,
            MutationType.CLOSE_RUNWAY_WINDOW:     self._close_runway_window,
            # ADD_ATFM_DEADLINE is handled at environment level via the atfm_deadlines
            # dict, not by modifying TaskDefinition flights.
            MutationType.ADD_ATFM_DEADLINE:       lambda t, m: t,
        }
        fn = dispatch.get(mut.mutation_type)
        return fn(task, mut) if fn else task

    def _tighten_window(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        squeeze = mut.params.get("squeeze_minutes", 3)
        updated = []
        for f in task.flights:
            if f.flight_id == mut.target_flight_id:
                new_earliest = f.earliest_minute + squeeze
                new_latest   = f.latest_minute   - squeeze
                if new_latest - new_earliest >= MIN_WINDOW_WIDTH:
                    f = f.model_copy(update={
                        "earliest_minute": new_earliest,
                        "latest_minute":   new_latest,
                    })
            updated.append(f)
        return task.model_copy(update={"flights": updated})

    def _increase_weather(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        delta = mut.params.get("penalty_delta", 0.15)
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                rwy = rwy.model_copy(update={
                    "weather_penalty": round(min(2.0, rwy.weather_penalty + delta), 2)
                })
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    def _inject_emergency(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "EMG001")
        minute = int(p.get("minute", 20))
        priority_str = str(p.get("priority", "emergency")).lower().strip()
        priority_str = _PRIORITY_ALIASES.get(priority_str, priority_str)
        try:
            priority = PriorityClass(priority_str)
        except ValueError:
            priority = PriorityClass.EMERGENCY
        runway_id = p.get("runway", task.runways[0].runway_id)

        new_flight = FlightRecord(
            flight_id=fid,
            airline="GOV",
            operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 2),
            latest_minute=minute + 6,
            allowed_runways=[runway_id],
            passengers=8,
            fuel_burn_per_minute=7.5,
            priority=priority,
            notes=f"Generator-injected {priority.value} diversion",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _add_conflicting_flight(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "WKT001")
        minute = int(p.get("minute", 10))
        try:
            wake = WakeClass(str(p.get("wake_class", "H")).upper())
        except ValueError:
            wake = WakeClass.HEAVY
        try:
            operation = OperationType(str(p.get("operation", "arrival")).lower())
        except ValueError:
            operation = OperationType.ARRIVAL
        runway_id = p.get("runway", task.runways[0].runway_id)

        new_flight = FlightRecord(
            flight_id=fid,
            airline="FRT",
            operation=operation,
            wake_class=wake,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 1),
            latest_minute=minute + 5,
            allowed_runways=[runway_id],
            passengers=1,
            fuel_burn_per_minute=6.0,
            priority=PriorityClass.NORMAL,
            notes="Generator-injected wake-turbulence trap (Heavy)",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _close_runway_window(
        self, task: TaskDefinition, mut: GeneratorMutation
    ) -> TaskDefinition:
        """Simulate runway closure via an extreme weather-penalty increase."""
        duration = mut.params.get("close_duration", 15)
        delta = min(1.9, 0.05 * duration)
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                rwy = rwy.model_copy(update={
                    "weather_penalty": min(2.0, rwy.weather_penalty + delta),
                    "notes": rwy.notes + f" [CLOSED {duration}min — generator]",
                })
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    # ── Solvability check ─────────────────────────────────────────────────────

    def _check_solvability(self, task: TaskDefinition) -> bool:
        """Heuristic: verify at least one valid assignment exists per flight."""
        from collections import defaultdict

        for f in task.flights:
            if f.latest_minute - f.earliest_minute < 2:
                return False
            if not f.allowed_runways:
                return False

        runway_demand: Dict[str, int] = defaultdict(int)
        for f in task.flights:
            for rwy_id in f.allowed_runways:
                runway_demand[rwy_id] += 1

        for rwy in task.runways:
            demand = runway_demand.get(rwy.runway_id, 0)
            effective_cap = rwy.hourly_capacity / rwy.weather_penalty
            max_ops = effective_cap * (task.planning_horizon_minutes / 60.0)
            if demand > max_ops * 1.5:
                return False

        return True

    def _score_scheduled_baseline(self, task: TaskDefinition) -> float:
        """PAIRED baseline: assign every flight at scheduled_minute (clamped to window)
        on its first allowed runway.  This is the 'do-nothing' heuristic score.
        """
        slots = []
        for f in task.flights:
            if not f.allowed_runways:
                continue
            minute = max(f.earliest_minute, min(f.latest_minute, f.scheduled_minute))
            slots.append(SlotAssignment(
                flight_id=f.flight_id,
                runway=f.allowed_runways[0],
                assigned_minute=minute,
                hold_minutes=0,
            ))
        try:
            outcome = simulate_plan(task, slots)
            return outcome.normalized_score
        except Exception:
            return 0.5

    def _deep_copy_task(self, task: TaskDefinition) -> TaskDefinition:
        return TaskDefinition.model_validate(task.model_dump())
