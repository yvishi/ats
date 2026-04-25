"""Scenario catalog for the ATC optimization benchmark.

Tasks are organized into five learning tiers:

  Tier 0 — Warmup     3-4 flights, 1 runway, all MEDIUM wake, 40-60 min windows.
                       A cold 1.5B model that can barely parse JSON should score ≥ 0.50
                       by placing flights close to their scheduled times.

  Tier 1 — Beginner   5-7 flights, 1-2 runways, all three wake classes introduced,
                       1 connection/medical priority. Naive FCFS scoring ≈ 0.30-0.50;
                       correct wake-aware ordering scores ≈ 0.65+.

  Tier 2 — Intermediate  8-14 flights, 2 runways, 1 emergency, LIGHT aircraft.
                          (Delhi EASY + Hyderabad HARD-layout-but-medium-volume)

  Tier 3 — Advanced   14-18 flights, 2 runways, multiple priorities, ATFM, tighter
                       windows.  (Mumbai MEDIUM)

  Tier 4 — Expert     18+ flights, complex IRROPS, multiple emergencies, strictest
                       windows.  (Bengaluru HARD)

Research grounding:
  - Bengio (2009): ordering easy→hard improves final model quality, not just speed.
  - Platanios (2019): root-schedule competence function c(t)=√(t/T) outperforms linear.
  - VCRL (2025): reward variance within GRPO group is a free ZPD proxy — target variance >0.
  - ProCuRL (2022): keep tasks where pass@1 ∈ [0.10, 0.75] for maximum learning signal.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

try:
    from .models import (
        Difficulty,
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        TaskDefinition,
        WakeClass,
    )
except ImportError:
    from models import (
        Difficulty,
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        TaskDefinition,
        WakeClass,
    )


def _f(
    flight_id: str,
    airline: str,
    operation: OperationType,
    wake_class: WakeClass,
    scheduled: int,
    earliest: int,
    latest: int,
    runways: List[str],
    passengers: int,
    burn: float,
    priority: PriorityClass = PriorityClass.NORMAL,
    connection_risk: float = 0.0,
    notes: str = "",
) -> FlightRecord:
    return FlightRecord(
        flight_id=flight_id,
        airline=airline,
        operation=operation,
        wake_class=wake_class,
        scheduled_minute=scheduled,
        earliest_minute=earliest,
        latest_minute=latest,
        allowed_runways=runways,
        passengers=passengers,
        fuel_burn_per_minute=burn,
        priority=priority,
        connection_risk=connection_risk,
        notes=notes,
    )


TASKS: List[TaskDefinition] = [

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 0 — WARMUP  (3-4 flights, 1 runway, all MEDIUM, 40-60 min windows)
    # Goal: teach the model basic JSON output and that flights must stay inside
    # [earliest, latest].  Naive scheduled-minute assignment scores ≥ 0.55.
    # ═════════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────────
    # T0-A: Bhopal Solo Departures
    # Three MEDIUM departures on one runway, generous windows, no other
    # constraints.  The only rule that matters: 3-min M→M separation.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="bhopal_solo_dep_t0",
        title="Bhopal Solo Departures (Warmup)",
        difficulty=Difficulty.EASY,
        airport="VABP",
        description=(
            "A quiet morning at Raja Bhoj Airport.  Three domestic jets need departure "
            "slots on the single active runway.  No weather, no emergencies — just "
            "sequence them safely and on time."
        ),
        objective=(
            "Assign departure slots for all three flights within their windows, "
            "maintaining the 3-minute minimum M→M wake separation."
        ),
        grading_focus=[
            "All flights assigned within their windows",
            "No wake-turbulence conflicts (M→M ≥ 3 min)",
            "Minimal total delay",
        ],
        planning_horizon_minutes=90,
        max_steps=3,
        delay_budget=200,
        fuel_budget=400.0,
        fairness_tolerance=20.0,
        runways=[
            RunwaySpec(
                runway_id="32",
                allowed_operations=[OperationType.DEPARTURE],
                hourly_capacity=20,
                weather_penalty=1.0,
                notes="Single active departure runway, clear conditions.",
            ),
        ],
        flights=[
            _f("BPL01", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  5,  5, 55, ["32"], 120, 2.2),
            _f("BPL02", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 15, 15, 65, ["32"], 115, 2.3),
            _f("BPL03", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 75, ["32"], 110, 2.1),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # T0-B: Vadodara Mixed Pair
    # Two arrivals + two departures on one runway, all MEDIUM, wide windows.
    # Challenge: interleave arrivals and departures on the same runway while
    # keeping 3-minute separation between any two movements.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="vadodara_mixed_pair_t0",
        title="Vadodara Mixed Pair (Warmup)",
        difficulty=Difficulty.EASY,
        airport="VABO",
        description=(
            "Vadodara Airport has four flights — two arriving and two departing — "
            "all on the same runway.  No special priorities, clear skies."
        ),
        objective=(
            "Interleave arrivals and departures without conflicts, keeping all "
            "assignments within their time windows."
        ),
        grading_focus=[
            "No runway conflicts (any consecutive pair ≥ 3 min)",
            "All flights inside their windows",
        ],
        planning_horizon_minutes=90,
        max_steps=3,
        delay_budget=200,
        fuel_budget=500.0,
        fairness_tolerance=20.0,
        runways=[
            RunwaySpec(
                runway_id="08",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=20,
                weather_penalty=1.0,
                notes="Single runway, mixed operations, clear conditions.",
            ),
        ],
        flights=[
            _f("VDR01", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM,  8,  8, 58, ["08"], 130, 4.5),
            _f("VDR02", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 14, 14, 64, ["08"], 125, 2.3),
            _f("VDR03", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 24, 24, 74, ["08"], 120, 4.4),
            _f("VDR04", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 32, 32, 82, ["08"], 115, 2.2),
        ],
    ),

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 1 — BEGINNER  (5-7 flights, mixed wake classes, 1-2 runways)
    # Goal: teach wake-turbulence asymmetry (H→L≥6 but L→H only 3) and basic
    # priority handling.  Naive FCFS scores ≈ 0.30-0.50; optimal ≈ 0.65+.
    # ═════════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────────
    # T1-A: Pune Wake Intro
    # Single runway, three wake classes.  The key insight: placing the LIGHT
    # aircraft BEFORE the HEAVY saves 3 minutes vs placing it after
    # (H→L=6 min gap vs L→H=3 min gap).  Proves wake asymmetry has real value.
    # One CONNECTION flight tests priority awareness.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="pune_wake_intro_t1",
        title="Pune Wake Turbulence Introduction (Beginner)",
        difficulty=Difficulty.EASY,
        airport="VAPO",
        description=(
            "Six flights on a single runway at Pune Airport introduce all three "
            "wake classes.  A heavy wide-body arrives early, followed by a light "
            "piston, two medium jets, and a connection-critical departure.  "
            "Smart ordering of the LIGHT aircraft relative to the HEAVY saves "
            "real runway time."
        ),
        objective=(
            "Sequence all six flights with no separation conflicts. Exploit wake "
            "asymmetry to keep the connection flight on time."
        ),
        grading_focus=[
            "H→L gap ≥ 6 min; L→H gap ≥ 3 min (asymmetry rewarded)",
            "Connection departure protected",
            "No window violations",
        ],
        planning_horizon_minutes=90,
        max_steps=4,
        delay_budget=150,
        fuel_budget=700.0,
        fairness_tolerance=18.0,
        runways=[
            RunwaySpec(
                runway_id="14",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=20,
                weather_penalty=1.0,
                notes="Single runway, mixed operations.",
            ),
        ],
        flights=[
            # Heavy arrives first: following LIGHT needs ≥6 min (H→L)
            _f("PNQ01", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,   8,  8, 40, ["14"], 260, 6.2),
            # Light piston: if after H, pad 6 min; if before H, only 3 min → smart agent moves it early
            _f("PNQ02", "SPJ", OperationType.ARRIVAL,   WakeClass.LIGHT,  16, 10, 46, ["14"],  18, 1.0),
            # Medium arrivals
            _f("PNQ03", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 22, 22, 52, ["14"], 165, 5.0),
            _f("PNQ04", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 30, 30, 60, ["14"], 170, 5.1),
            # Connection departure — must stay close to its window
            _f("PNQ05", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 35, 35, 62, ["14"], 180, 2.4,
               PriorityClass.CONNECTION, 0.7, "Bank connections at risk if delayed past t=62."),
            # Normal medium departure
            _f("PNQ06", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 44, 44, 74, ["14"], 175, 2.3),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # T1-B: Nagpur Dual Runway
    # Two runways (one arr-only, one mixed).  Seven flights with a heavy, a
    # light, and a MEDICAL departure.  Primary challenge: which runway gets
    # which flight, and medical priority awareness.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="nagpur_dual_runway_t1",
        title="Nagpur Dual Runway Intro (Beginner)",
        difficulty=Difficulty.EASY,
        airport="VANP",
        description=(
            "Two runways at Nagpur: 12L accepts arrivals only, 12R handles mixed "
            "operations.  A heavy freighter must use 12R; a light cargo feeder "
            "can use either runway.  A time-sensitive medical cargo departure "
            "needs priority handling."
        ),
        objective=(
            "Assign all seven flights to the correct runways without conflicts, "
            "protecting the medical departure's window."
        ),
        grading_focus=[
            "Medical departure protected (≤ 10 min delay)",
            "Runway restrictions respected (no departures on 12L)",
            "Wake separation maintained across both runways",
        ],
        planning_horizon_minutes=90,
        max_steps=4,
        delay_budget=150,
        fuel_budget=800.0,
        fairness_tolerance=18.0,
        runways=[
            RunwaySpec(
                runway_id="12L",
                allowed_operations=[OperationType.ARRIVAL],
                hourly_capacity=18,
                weather_penalty=1.0,
                notes="Arrival-only runway.",
            ),
            RunwaySpec(
                runway_id="12R",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.0,
                notes="Mixed-ops runway; heavy freighter requires 12R.",
            ),
        ],
        flights=[
            # Arrivals: can use 12L or 12R
            _f("NGP01", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM,  6,  6, 50, ["12L", "12R"], 155, 4.8),
            # Heavy freighter: 12R only
            _f("NGP02", "FTR", OperationType.ARRIVAL,   WakeClass.HEAVY,  12, 12, 45, ["12R"],          310, 6.5,
               PriorityClass.NORMAL, 0.0, "Heavy freighter — 12R only."),
            # Light feeder: either runway
            _f("NGP03", "CRG", OperationType.ARRIVAL,   WakeClass.LIGHT,  20, 14, 54, ["12L", "12R"],    15, 1.0,
               PriorityClass.NORMAL, 0.0, "Light feeder — exploit L→M asymmetry."),
            _f("NGP04", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 28, 28, 58, ["12L", "12R"], 160, 4.9),
            # Medical departure: 12R only (no dep on 12L)
            _f("NGP05", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 18, 42, ["12R"],          150, 2.5,
               PriorityClass.MEDICAL, 0.3, "Medical cargo — must depart within window."),
            # Normal departures: 12R only
            _f("NGP06", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 26, 26, 62, ["12R"],          170, 2.4),
            _f("NGP07", "IGO", OperationType.DEPARTURE, WakeClass.HEAVY,  38, 38, 68, ["12R"],          285, 5.8),
        ],
    ),

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 2 — INTERMEDIATE  (existing EASY + the Hyderabad scenario)
    # ═════════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────────
    # EASY — Delhi Monsoon Departure Recovery
    # Changes from v1: added 2 LIGHT aircraft (SPJ01, CRJ90) to exercise
    # untested H/M↔L wake separations.  Tightened 3 padded normal-flight
    # windows so every flight presents a real scheduling constraint.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="delhi_monsoon_recovery_easy",
        title="Delhi Monsoon Departure Recovery",
        difficulty=Difficulty.EASY,
        airport="VIDP",
        description=(
            "A short convective weather burst reduced departure acceptance rates for "
            "the morning bank. The controller must resequence mixed arrivals and "
            "departures — including a charter light piston and a regional feeder — "
            "while protecting a medical flight."
        ),
        objective=(
            "Restore a safe runway sequence, keep the medical departure close to "
            "schedule, correctly space light aircraft around heavier traffic, and "
            "minimize total passenger delay."
        ),
        grading_focus=[
            "Zero runway conflicts including light-aircraft wake spacing",
            "Medical flight protected",
            "Low total delay and fuel burn",
        ],
        planning_horizon_minutes=100,
        max_steps=4,
        delay_budget=105,
        fuel_budget=730.0,
        fairness_tolerance=14.0,
        runways=[
            RunwaySpec(
                runway_id="27L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.15,
                notes="Usable for mixed operations with reduced wet-runway spacing.",
            ),
            RunwaySpec(
                runway_id="27R",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.15,
                notes="Parallel runway, but departures should be metered carefully.",
            ),
        ],
        flights=[
            # ── Original flights (3 windows tightened) ───────────────────────
            _f("IGO601", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  8,  8, 55, ["27L", "27R"], 180, 2.6),
            _f("AIC221", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 60, ["27L", "27R"], 168, 2.9, PriorityClass.CONNECTION, 0.6, "Tight onward domestic bank."),
            _f("VTI404", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 15, 15, 48, ["27L"],         156, 4.8, PriorityClass.CONNECTION, 0.7, "Inbound hub connections at risk."),
            _f("AKJ118", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 18, 18, 38, ["27R"],         150, 2.4),    # was [18,70] — tightened
            _f("IGO882", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 20, 20, 56, ["27L", "27R"], 174, 5.2, PriorityClass.CONNECTION, 0.5, "Arrival bank should not absorb long holding."),
            _f("AIC911", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 24, 24, 68, ["27L", "27R"], 132, 2.8, PriorityClass.MEDICAL,     0.2, "Medical cargo onboard."),
            _f("VTI721", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 52, ["27L", "27R"], 188, 2.7),    # was [27,84] — tightened
            _f("GOW330", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 55, ["27R"],         186, 2.5),   # was [30,88] — tightened
            # ── NEW: LIGHT aircraft ──────────────────────────────────────────
            # SPJ01 departs 27L only.  Any MEDIUM departure immediately behind a
            # LIGHT needs a 4-minute gap (M←L); any HEAVY needs 6 minutes.
            _f("SPJ01",  "SPJ", OperationType.DEPARTURE, WakeClass.LIGHT,  10, 10, 30, ["27L"],          22, 1.1, PriorityClass.NORMAL, 0.0, "Charter light piston — must be slotted before heavier departures back up 27L."),
            # CRJ90 arrives 27R only.  A HEAVY arriving before it requires 6 min
            # separation; arriving before a HEAVY only 3 min — asymmetry matters.
            _f("CRJ90",  "CRJ", OperationType.ARRIVAL,   WakeClass.LIGHT,  22, 22, 42, ["27R"],          48, 1.8, PriorityClass.NORMAL, 0.0, "Regional feeder — light aircraft wake rules apply on 27R."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # MEDIUM — Mumbai Hub Bank Balance
    # Changes from v1: added DKT22 (LIGHT cargo feeder, rwy 14 only) and
    # JET05 (LIGHT corporate arrival, rwy 27 only).  Tightened IGO774 and
    # AKJ901 windows so the back-of-bank flights are real constraints.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="mumbai_bank_balance_medium",
        title="Mumbai Hub Bank Balance",
        difficulty=Difficulty.MEDIUM,
        airport="VABB",
        description=(
            "A runway inspection removed one preferred departure flow during the "
            "mid-morning hub bank.  A light cargo feeder and a corporate jet join "
            "the queue alongside international heavies.  Controllers need to protect "
            "fuel-sensitive arrivals and keep delay distribution fair across airlines."
        ),
        objective=(
            "Produce a conflict-free schedule that limits missed bank connections, "
            "correctly spaces light aircraft around heavier traffic, keeps a "
            "fuel-sensitive arrival near the front, and avoids over-penalizing "
            "a single carrier."
        ),
        grading_focus=[
            "Arrival fuel protection",
            "Light-aircraft wake separation correctness",
            "Fairness across airlines",
            "Capacity-aware sequencing",
        ],
        planning_horizon_minutes=120,
        max_steps=4,
        delay_budget=175,
        fuel_budget=1280.0,
        fairness_tolerance=12.0,
        runways=[
            RunwaySpec(
                runway_id="27",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=20,
                weather_penalty=1.2,
                notes="Primary runway with wet braking action advisory.",
            ),
            RunwaySpec(
                runway_id="14",
                allowed_operations=[OperationType.DEPARTURE],
                hourly_capacity=14,
                weather_penalty=1.1,
                notes="Departure-only flow after inspection release.",
            ),
        ],
        flights=[
            # ── NEW: LIGHT cargo feeder on dep-only rwy 14 ───────────────────
            # A MEDIUM departure behind DKT22 on 14 needs 4 min (M←L).
            _f("DKT22",  "DKT", OperationType.DEPARTURE, WakeClass.LIGHT,   8,  8, 28, ["14"],           18, 1.0, PriorityClass.NORMAL,     0.0, "Cargo feeder — light aircraft on dep-only rwy 14; sequence before MEDIUM wave."),
            # ── Original flights (2 windows tightened) ───────────────────────
            _f("IGO202", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  5,  5, 70, ["14", "27"],     186, 2.4),
            _f("AIC540", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  10, 10, 44, ["27"],           264, 6.1, PriorityClass.CONNECTION, 0.8, "Long-haul inbound with large connection bank."),
            _f("VTI313", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 13, 13, 78, ["14"],           176, 2.5, PriorityClass.CONNECTION, 0.7),
            _f("AKJ440", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 17, 17, 82, ["14", "27"],    172, 2.4),
            _f("IGO558", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 19, 19, 51, ["27"],           181, 5.0, PriorityClass.NORMAL,     0.4),
            _f("AIX117", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 22, 22, 88, ["14"],           164, 2.6),
            _f("AIC612", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 92, ["14", "27"],    158, 2.9, PriorityClass.CONNECTION, 0.6),
            _f("VTI880", "VTI", OperationType.ARRIVAL,   WakeClass.MEDIUM, 28, 28, 55, ["27"],           170, 5.6, PriorityClass.MEDICAL,    0.4, "Special handling passenger onboard."),
            _f("IGO774", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 65, ["14", "27"],    190, 2.5),  # was [31,100] — tightened
            _f("GOW410", "GOW", OperationType.DEPARTURE, WakeClass.MEDIUM, 35, 35, 102, ["14"],          186, 2.3),
            _f("AIC845", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  38, 38, 66, ["27"],           248, 6.3, PriorityClass.CONNECTION, 0.7, "Fuel-sensitive after holding south of field."),
            _f("AKJ901", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 42, 42, 72, ["14", "27"],    168, 2.4),  # was [42,108] — tightened
            # ── NEW: LIGHT corporate arrival on rwy 27 ────────────────────────
            # If arriving AFTER AIC845 (HEAVY), spacing must be 6 min (H→L).
            # If arriving BEFORE AIC845, only 3 min needed (L→H) — placing JET05
            # early is significantly more efficient, rewarding smart sequencing.
            _f("JET05",  "JET", OperationType.ARRIVAL,   WakeClass.LIGHT,  33, 33, 55, ["27"],            12, 1.2, PriorityClass.NORMAL,     0.0, "Corporate jet arrival — LIGHT, rwy 27 only; heavy-wake separation critical."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # HARD — Bengaluru IRROPS Recovery Command
    # Changes from v1: MED208 window fixed [31,85]→[31,55] (organ transport
    # must be operationally credible).  GOW099 (padding flight) removed.
    # IGO702 window tightened [30,108]→[30,65].  VFR01 (LIGHT arrival)
    # added to force early-slot ordering before HEAVY waves.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="bengaluru_irrops_hard",
        title="Bengaluru IRROPS Recovery Command",
        difficulty=Difficulty.HARD,
        airport="VOBL",
        description=(
            "A rapidly evolving IRROPS event combines wet-runway spacing, "
            "an emergency inbound, a time-critical organ-transport departure, and a "
            "heavily peaking departure bank.  A light VFR aircraft on a special permit "
            "must be resolved before the heavy wave arrives.  The controller must "
            "sequence mixed traffic with both safety and network impact in mind."
        ),
        objective=(
            "Protect the emergency arrival and medical departure within their strict "
            "windows, slot the light VFR arrival before it is blocked by heavy traffic, "
            "keep all assignments conflict-free under reduced capacity, minimize system "
            "delay, and distribute unavoidable delay fairly across participating airlines."
        ),
        grading_focus=[
            "Emergency and medical traffic prioritized within strict windows",
            "VFR01 slotted before heavy-wave blockage",
            "Zero separation and capacity conflicts",
            "Balanced delay distribution under IRROPS",
        ],
        planning_horizon_minutes=140,
        max_steps=6,
        delay_budget=220,
        fuel_budget=1680.0,
        fairness_tolerance=8.5,
        runways=[
            RunwaySpec(
                runway_id="09L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=18,
                weather_penalty=1.3,
                notes="Mixed-mode runway with longer heavy-to-medium spacing.",
            ),
            RunwaySpec(
                runway_id="09R",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=16,
                weather_penalty=1.25,
                notes="Secondary runway handling overflow and tactical recovery.",
            ),
        ],
        flights=[
            _f("IGO110", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM,  4,  4, 78,  ["09L", "09R"], 184, 2.4),
            _f("AIC781", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,   7,  7, 34,  ["09L"],         252, 6.5, PriorityClass.CONNECTION, 0.8),
            # ── NEW: LIGHT VFR on special permit ─────────────────────────────
            # VFR01 must land before the MEDIUM/HEAVY wave builds after t=14.
            # Placed after AIC781 (HEAVY): requires 6-min gap (H→L).
            # Placed before AIC781:        only 3-min gap (L→H) and no blockage.
            # Smart agents will slot it early; naive agents will conflict it.
            _f("VFR01",  "VFR", OperationType.ARRIVAL,   WakeClass.LIGHT,   9,  9, 20,  ["09L", "09R"],  6, 1.0, PriorityClass.NORMAL, 0.0, "Light VFR on special permit — must clear before heavy wave; strict window."),
            _f("VTI220", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM,  9,  9, 86,  ["09R"],         170, 2.6),
            _f("AKJ662", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 12, 12, 90,  ["09L", "09R"], 168, 2.3),
            _f("GOW510", "GOW", OperationType.ARRIVAL,   WakeClass.MEDIUM, 14, 14, 46,  ["09L", "09R"], 174, 5.3, PriorityClass.NORMAL, 0.4),
            _f("AIC110", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 16, 16, 92,  ["09L", "09R"], 162, 2.7, PriorityClass.CONNECTION, 0.7),
            _f("IGO455", "IGO", OperationType.ARRIVAL,   WakeClass.MEDIUM, 18, 18, 44,  ["09L"],         176, 5.5, PriorityClass.CONNECTION, 0.6),
            _f("VTI918", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 21, 21, 94,  ["09R"],         188, 2.5),
            _f("MED001", "GOV", OperationType.ARRIVAL,   WakeClass.MEDIUM, 23, 23, 28,  ["09L", "09R"],   6, 7.0, PriorityClass.EMERGENCY, 0.1, "Emergency medical diversion with priority handling."),
            _f("AKJ303", "AKJ", OperationType.DEPARTURE, WakeClass.MEDIUM, 25, 25, 98,  ["09L", "09R"], 178, 2.4),
            _f("AIX770", "AIX", OperationType.DEPARTURE, WakeClass.MEDIUM, 27, 27, 102, ["09R"],         164, 2.6),
            _f("IGO702", "IGO", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 65,  ["09L", "09R"], 192, 2.4),  # was [30,108] — tightened
            _f("MED208", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 31, 31, 55,  ["09R"],         154, 2.8, PriorityClass.MEDICAL, 0.4, "Time-critical organ transport departure — 24-minute window is firm."),  # was [31,85]
            _f("AIC605", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  33, 33, 60,  ["09L"],         244, 6.4, PriorityClass.CONNECTION, 0.7),
            _f("VTI640", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 36, 36, 112, ["09R"],         182, 2.5),
            _f("AKJ818", "AKJ", OperationType.ARRIVAL,   WakeClass.MEDIUM, 42, 42, 68,  ["09L", "09R"], 171, 5.4, PriorityClass.CONNECTION, 0.5),
            _f("AIC990", "AIC", OperationType.ARRIVAL,   WakeClass.HEAVY,  45, 45, 72,  ["09L"],         246, 6.6, PriorityClass.CONNECTION, 0.8, "Second fuel-sensitive long-haul inbound wave."),
        ],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # NEW — Hyderabad Cargo Crunch (Medium-Hard)
    # Single runway (09R bird-strike inspection closes 09L).
    # All three wake classes present: HEAVY freighters, MEDIUM passenger jets,
    # LIGHT cargo feeders.  No emergency flight — difficulty comes entirely
    # from runway scarcity and wake turbulence asymmetry.
    # ─────────────────────────────────────────────────────────────────────────
    TaskDefinition(
        task_id="hyderabad_cargo_crunch_medium_hard",
        title="Hyderabad Single-Runway Cargo Crunch",
        difficulty=Difficulty.HARD,
        airport="VOHS",
        description=(
            "A bird-strike inspection grounds runway 09R at Rajiv Gandhi International "
            "during the peak mango-export cargo wave.  All traffic — international "
            "heavy freighters, domestic passenger jets, and light cargo feeders — must "
            "be sequenced through runway 09L alone.  Wake turbulence between all three "
            "aircraft classes is the central constraint."
        ),
        objective=(
            "Sequence all seven aircraft through the single available runway without "
            "separation conflicts, exploiting wake-class asymmetry to maximize "
            "throughput while protecting connection passengers on AIC300."
        ),
        grading_focus=[
            "Zero conflicts on a single runway",
            "Wake-class asymmetry exploited (L→H cheaper than H→L)",
            "Connection passengers on AIC300 protected",
            "Minimal total delay given severe capacity constraint",
        ],
        planning_horizon_minutes=80,
        max_steps=4,
        delay_budget=90,
        fuel_budget=720.0,
        fairness_tolerance=15.0,
        runways=[
            RunwaySpec(
                runway_id="09L",
                allowed_operations=[OperationType.ARRIVAL, OperationType.DEPARTURE],
                hourly_capacity=16,
                weather_penalty=1.1,
                notes="Only active runway — 09R closed for bird-strike inspection.",
            ),
        ],
        flights=[
            # ── Slot ordering puzzle ─────────────────────────────────────────
            # FTR001 (HEAVY arr) → SPJ100 (LIGHT dep): 6-min gap required (H→L)
            # SPJ100 (LIGHT dep) → AIC300 (MEDIUM arr): 3-min gap (L→M) — asymmetry!
            # FTR002 (HEAVY dep) → CRG55 (LIGHT arr): 6-min gap (H→L)
            # CRG55  (LIGHT arr) → VTI200 (MEDIUM dep): only 3-min (L→M) — efficient
            # Optimal ordering saves ~6 minutes of total delay vs naive FCFS.
            _f("FTR001", "FTR", OperationType.ARRIVAL,   WakeClass.HEAVY,  10, 10, 28, ["09L"],  312, 6.8, PriorityClass.NORMAL,      0.0, "Heavy freighter — sets 6-min gap for any following LIGHT traffic."),
            _f("SPJ100", "SPJ", OperationType.DEPARTURE, WakeClass.LIGHT,  14, 14, 32, ["09L"],   20, 1.1, PriorityClass.NORMAL,      0.0, "Light cargo feeder — if after HEAVY arr, needs 6 min; if before, only 3."),
            _f("AIC300", "AIC", OperationType.ARRIVAL,   WakeClass.MEDIUM, 17, 17, 35, ["09L"],  198, 5.4, PriorityClass.CONNECTION,  0.8, "Connection bank closes at t=40; keep delays minimal."),
            _f("FTR002", "FTR", OperationType.DEPARTURE, WakeClass.HEAVY,  21, 21, 42, ["09L"],  285, 5.9, PriorityClass.NORMAL,      0.0, "Outbound freighter — after departure, following LIGHT needs 6-min gap."),
            _f("VTI200", "VTI", OperationType.DEPARTURE, WakeClass.MEDIUM, 23, 23, 44, ["09L"],  182, 2.6, PriorityClass.CONNECTION,  0.5, "Domestic passenger departure — tight window in single-runway crunch."),
            _f("CRG55",  "CRG", OperationType.ARRIVAL,   WakeClass.LIGHT,  27, 27, 46, ["09L"],   16, 1.0, PriorityClass.NORMAL,      0.0, "Light cargo arrival — exploit L→M asymmetry to save slots for VTI200."),
            _f("AIC480", "AIC", OperationType.DEPARTURE, WakeClass.MEDIUM, 30, 30, 52, ["09L"],  175, 2.5, PriorityClass.NORMAL,      0.3, "Second passenger departure — last slot in the crunch window."),
        ],
    ),
]


# ── Tier catalog ─────────────────────────────────────────────────────────────
# Explicit tier assignment for every task.
#
# Tier 0 — Warmup      naive-FCFS scores ≥ 0.55 (teaches JSON format)
# Tier 1 — Beginner    naive ≈ 0.30-0.50, optimal ≈ 0.65 (wake asymmetry)
# Tier 2 — Intermediate naive ≈ 0.20-0.40, optimal ≈ 0.55 (LIGHT + emergency)
# Tier 3 — Advanced    naive ≈ 0.15-0.35, optimal ≈ 0.50 (ATFM, 14 flights)
# Tier 4 — Expert      naive ≈ 0.10-0.25, optimal ≈ 0.45 (IRROPS, 18 flights)
#
# Source evidence: engine.py penalty structure — conflicts multiply the final
# score; a 20-flight task with even 1 unresolved conflict drops score by ~5%.
# Fairness tolerance 8.5 (Bengaluru) vs 20.0 (warmup) is a 2.4× harder target.

TASK_TIER: Dict[str, int] = {
    "bhopal_solo_dep_t0":                   0,
    "vadodara_mixed_pair_t0":               0,
    "pune_wake_intro_t1":                   1,
    "nagpur_dual_runway_t1":                1,
    "delhi_monsoon_recovery_easy":          2,
    "hyderabad_cargo_crunch_medium_hard":   2,  # compact but all-wake-classes puzzle
    "mumbai_bank_balance_medium":           3,
    "bengaluru_irrops_hard":                4,
}


def task_catalog() -> Dict[str, TaskDefinition]:
    """Return all tasks keyed by task_id."""
    return {t.task_id: t for t in TASKS}


def ordered_tasks() -> Iterable[TaskDefinition]:
    """Return tasks ordered by tier then task_id (easy → hard)."""
    return sorted(TASKS, key=lambda t: (TASK_TIER.get(t.task_id, 2), t.task_id))


def tasks_by_tier(tier: int) -> List[TaskDefinition]:
    """Return tasks belonging to exactly this tier (0-4)."""
    return [t for t in TASKS if TASK_TIER.get(t.task_id, 2) == tier]


def tasks_up_to_tier(tier: int) -> List[TaskDefinition]:
    """Return all tasks at or below this tier — the eligible pool for a given competence level.

    Always returns at least the tier-0 tasks so the dataset builder never gets an
    empty pool (guards against off-by-one during the first episode).
    """
    pool = [t for t in TASKS if TASK_TIER.get(t.task_id, 2) <= tier]
    if not pool:
        pool = tasks_by_tier(0)
    return pool


MAX_TASK_TIER: int = max(TASK_TIER.values())  # 4


def render_task_briefing(task: TaskDefinition) -> str:
    """Generate the ATC situation briefing shown to agents.

    Produces a concise, structured text that describes the scenario, runway
    configuration, flight list (with priorities and wake classes), and the
    scoring focus areas.  This is the primary context LLMs receive.
    """
    lines: List[str] = []

    lines.append(f"# ATC BRIEFING — {task.title}")
    lines.append(f"Airport: {task.airport}   Difficulty: {task.difficulty.value.upper()}")
    lines.append(f"Planning horizon: {task.planning_horizon_minutes} min   "
                 f"Delay budget: {task.delay_budget} min   "
                 f"Fuel budget: {task.fuel_budget:.0f} kg")
    lines.append("")
    lines.append(f"## Situation\n{task.description}")
    lines.append("")
    lines.append(f"## Objective\n{task.objective}")
    lines.append("")

    lines.append("## Runways")
    for rwy in task.runways:
        ops = "/".join(op.value for op in rwy.allowed_operations)
        lines.append(
            f"  {rwy.runway_id}: {ops}, capacity {rwy.hourly_capacity}/hr, "
            f"wx_penalty×{rwy.weather_penalty}. {rwy.notes}"
        )
    lines.append("")

    lines.append("## Wake Turbulence Separation (minutes between consecutive movements)")
    lines.append("  Leader→Follower : H→H=4  H→M=5  H→L=6  M→H=3  M→M=3  M→L=4  L→H=3  L→M=3  L→L=3")
    lines.append("")

    lines.append("## Flights (flight_id | airline | op | wake | sched | window | runways | priority | risk | notes)")
    for f in task.flights:
        r_str = ",".join(f.allowed_runways)
        risk = f"-risk={f.connection_risk:.1f}" if f.connection_risk > 0 else ""
        notes = f"  [{f.notes}]" if f.notes else ""
        lines.append(
            f"  {f.flight_id:<8} {f.airline:<4} {f.operation.value:<3} "
            f"{f.wake_class.value:<3} sched={f.scheduled_minute:>3} "
            f"[{f.earliest_minute},{f.latest_minute}] "
            f"rwy={r_str}  {f.priority.value}{risk}{notes}"
        )
    lines.append("")

    lines.append("## Grading Focus")
    for item in task.grading_focus:
        lines.append(f"  • {item}")

    return "\n".join(lines)

