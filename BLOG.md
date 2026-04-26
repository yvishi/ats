# Shared Runways, Split Intelligence: How a Video Game Became a Hackathon-Winning AI System

*A story about air traffic control, two arguing agents, and a model that learned to manage an ICU without ever being told what a hospital is.*

---

## It Started With a Video Game

My friend was playing **ATC — Air Traffic Control Simulator** one evening. I was watching. He was losing.

The game gives you a radar screen full of aircraft blips. You type instructions. Planes follow. Except — they don't always follow in time. A heavy jet you cleared for final approach is now too close to a departing regional. You give one correction. That correction delays a medical flight. That delay cascades into a missed connection for 180 passengers.

He paused the game and said: *"This is impossible. You'd need to see three minutes into the future just to survive the next one."*

I said: *"What if two AIs played together — one handling arrivals, one handling departures — and they had to negotiate?"*

He unpaused. His plane crashed. We opened a laptop.

That conversation became **ADAPT** — a multi-agent reinforcement learning system for cooperative air traffic control, with a twist nobody expected.

---

## The Real Problem Is Not the Planes

Before writing a single line of code, we spent a week learning how actual ATC works. Not the game. The real thing.

At any major Indian airport — Delhi, Mumbai, Bengaluru — two controllers share the same runway system. The **Arrival Manager (AMAN)** sequences inbound aircraft. The **Departure Manager (DMAN)** sequences outbounds. They work from adjacent consoles. They talk to each other constantly. They operate on *partial information*.

AMAN does not see DMAN's departure queue. DMAN does not see AMAN's ATFM slot constraints. Each agent knows their half of the picture. The runway knows nothing — it just accepts whoever gets there first.

This isn't a simple optimization problem. It has three properties that make it genuinely hard:

**1. Temporal cascade.** A Heavy aircraft cleared to land at minute 5 creates a 6-minute wake turbulence gap. Any Light aircraft scheduled to depart at minute 8 must now wait until minute 11. That 3-minute delay shifts the next departure, which shifts ATFM deadlines at destination airports, which cascades to 3 more airports. The damage from one decision at minute 5 isn't visible until minute 45.

**2. Asymmetric information.** AMAN knows the arrival sequence. DMAN knows the departure deadlines. Neither knows the other's constraints. They must *infer* each other's pressure from typed messages. This is not coordination with full visibility — it's coordination under genuine uncertainty.

**3. Shifting objectives.** Every shift, the supervisor changes priority. "Today, fuel efficiency first." "Today, no delays to connecting passengers." "Today, safety margins above all." A system that memorizes one objective fails when the objective changes mid-episode.

---

## Two Agents, One Model

Our first architectural decision: **one LLM, three system prompts.**

We don't use two separate models. AMAN and DMAN are the same 1.5-billion-parameter Qwen2.5 model, differentiated by what they see in their system prompt. This tests something important — can one model reason correctly from asymmetric information frames? Can the same weights that think like an arrival controller also think like a departure controller?

The answer, after training, is: yes. Measurably.

The protocol we built — called **BID → NEGOTIATE → FINAL** — mirrors how real controllers work:

```
Round 0: BID
  AMAN submits arrivals plan independently.
  DMAN submits departures plan independently.
  Environment detects runway conflicts.
  If no conflicts: skip to FINAL.

Round 1: NEGOTIATE
  Both agents receive the conflict log.
  Agents revise plans and exchange typed messages:
    runway_claim | yield | emergency_broadcast | theory_of_mind
  Agents can pre-emptively yield slots for the other's emergency.

Round 2: FINAL
  Merged plan runs through physics simulation.
  Per-agent rewards computed. Episode ends.
```

The key word is *partial observability*. AMAN receives `atfm_deadlines: {}` — an empty dict. DMAN receives the real deadline map. This is not a shortcut. This is the actual information asymmetry that exists between the two consoles.

---

## Training With Physics, Not Judges

We made a deliberate choice: **no LLM in the reward loop.**

Every score is computed from deterministic physics:
- Wake turbulence separation? Computed from a matrix lookup.
- ATFM deadline compliance? `assigned_minute <= deadline` — true or false.
- Emergency handling? Delay delta from scheduled time.
- Fairness? Gini coefficient over airline delay distributions.

Why does this matter? Because LLM judges hallucinate. They're inconsistent across prompts. They can be gamed by confident-sounding but wrong outputs. A model trained against an LLM judge is learning to *sound good*, not to *be good*.

Our reward is physically grounded. The model cannot hallucinate its way to a better score. A conflict is a conflict. A missed deadline is a missed deadline.

### The Reward Architecture

We built a **three-tier gated composable rubric**:

**Tier 1 — Safety Gate:** Absolute ceilings. One conflict? Maximum score drops to 0.40. An EMERGENCY flight more than 5 minutes late? Ceiling 0.35. No amount of delay efficiency can buy back a safety violation.

**Tier 2 — Priority Rubric (30% weight):**
```
0.50 × emergency_score + 0.30 × medical_score + 0.20 × connection_score
```

**Tier 3 — Efficiency Rubric (70% weight):**
```
0.35 × delay_efficiency + 0.25 × fuel_efficiency
0.20 × fairness + 0.20 × connection_impact
```

And we implemented **seven novel loss components** for the training signal:

| Component | Problem it solves |
|-----------|------------------|
| Temporal Credit Assignment | Early decisions get credit for late consequences |
| Hierarchical Decomposition | Strategic + tactical + operational separate gradients |
| Recovery Gradient | Reward agents that recover from their own mistakes |
| Contrastive Pair Reward | Always signal even when all outputs are bad |
| Information-Theoretic Coordination | Reward informative messages, not boilerplate |
| Causal Credit (Shapley) | Tell each agent *which actions* caused success |
| Adaptive KL Regularization | Tighten KL when plateau, relax when improving |

---

## GRPO: Teaching the Model to Argue With Itself

We trained with **GRPO — Group Relative Policy Optimization**.

The idea: generate N completions per prompt, compute rewards for all of them, use the relative ranking to estimate advantage. No value network. No reference model overhead. Just: "these outputs were better than those outputs — move toward them."

For a 1.5B model on a Colab T4 GPU, GRPO is the right choice. It's computationally lean, stable with small batch sizes, and doesn't require a separately trained critic.

```
A_i = (r_i - mean(group)) / (std(group) + ε)

N=2 generations minimum for non-degenerate std.
KL coefficient = 0.0 with PEFT (non-zero causes ref_per_token_logps crash).
```

Training results after 150 episodes:

| Metric | Before | After |
|--------|--------|-------|
| Composite score | 0.47 | **0.71** |
| Emergency handling | 61% | **94%** |
| Conflict rate | 18% | **4%** |
| ATFM compliance | 74% | **91%** |
| Coordination quality | 0.08 | **0.34** |

Two hours. One T4 GPU. Free Colab tier.

---

## The Part Nobody Expected: ADAPT

Three weeks in, we had AMAN and DMAN working. The system was solid. We were nearly done.

Then someone asked: *"What if you took these same agents — no retraining, no code changes — and threw them at a completely different problem?"*

We stared at the question. Then we built ADAPT.

### The Insight

Air traffic control is, at its core, a **resource scheduling problem under uncertainty with priority classes and cascade risk**. So is hospital ICU surge management. So is container port scheduling. So is factory floor allocation during component shortage.

The domain changes. The structure doesn't.

ADAPT is a third agent — the **Adaptive Decision Agent for Problem Transfer** — that receives an unknown scheduling problem and maps it to ATC parameters that AMAN and DMAN already understand.

It reads three structural signals:

```python
time_pressure   = 1 - (avg_window / planning_horizon)  # How tight is the schedule?
connection_risk = cascade failure probability             # How bad if this is delayed?
urgency_flag    = "urgent" in notes.lower()             # Direct operator signal

combined = 0.5 × time_pressure + 0.4 × connection_risk + 0.1 × urgency_flag
```

Then it maps:
```
combined ≥ 0.70 → Wake class H   (max separation needed)
combined 0.35–0.70 → Wake class M
combined < 0.35 → Wake class L

connection_risk ≥ 0.80 → Priority: emergency
connection_risk ≥ 0.50 → Priority: medical
connection_risk ≥ 0.20 → Priority: connection
else              → Priority: normal
```

### The Test: ICU Mass Casualty

We gave ADAPT three ICU surge scenarios. The entity types were TRAUMA, CARDIAC, POST_OP, and ROUTINE patients.

ADAPT never sees the word "TRAUMA." It reads the numbers:
```
TRAUMA:  time_pressure=0.95  connection_risk=0.93  → score=0.94 → H / emergency
CARDIAC: time_pressure=0.78  connection_risk=0.70  → score=0.67 → M / medical
POST_OP: time_pressure=0.56  connection_risk=0.45  → score=0.46 → M / connection
ROUTINE: time_pressure=0.05  connection_risk=0.05  → score=0.05 → L / normal
```

It maps a trauma patient the same way it maps a wide-body jet. Not because it knows what trauma is. Because the structure of the problem is identical.

AMAN and DMAN then schedule the ICU beds — runway slots translated to bed-time slots, wake turbulence translated to resource contention — without any code changes.

**Zero retraining. Zero domain labels. Zero shots.**

---

## The Self-Adapting Curriculum

One more thing we built that we're proud of: **diagnostic skill profiling**.

Most RL curricula work like this: track the average score, make the task harder when score goes up. This is *self-improving*. It escalates difficulty. But it doesn't target what the agent actually can't do.

We built something different: **self-adapting curriculum**.

The system maintains a rolling mean across 7 skill dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| `conflict_avoidance` | Wake separation and cross-lane conflict rate |
| `delay_efficiency` | Total system delay vs budget |
| `emergency_handling` | On-time dispatch of EMERGENCY/MEDICAL |
| `atfm_compliance` | DMAN meets network slot deadlines |
| `coverage` | Fraction of flights assigned |
| `coordination` | Multi-agent negotiation quality |
| `fairness` | Equitable delay across airlines |

Each episode:
1. Identify weakest dimension (rolling mean over last 10 episodes)
2. Select a mutation that specifically exercises that gap
3. Scale reward weights: weakest skill gets loudest gradient

```python
raw_i = exp(gap_i × 3.0)          # amplify differences
w_i   = raw_i / mean(raw)         # normalize: mean weight = 1.0
w_i   = clamp(w_i, 0.25, 2.50)   # floor and ceiling
```

If `emergency_handling` is the weakest skill, the next episode injects an emergency with a tight 8-minute window. The reward weight for emergency_score goes to 2.1×. The gradient shouts exactly where the model needs to improve.

Not harder. **Specifically harder where it needs to be.**

---

## What We Actually Built

Four months of nights and weekends. Here's the inventory:

- **4 ATC scenarios** (Delhi, Mumbai, Bengaluru, Hyderabad) with 12–20 flights each, real wake turbulence constraints, ATFM deadlines
- **3 domain-transfer scenarios** (ICU normal day, flu surge, mass casualty)
- **3 coordinating agents** (AMAN, DMAN, ADAPT) — one 1.5B model
- **7 novel loss components** for long-horizon sparse reward training
- **BID → NEGOTIATE → FINAL** protocol with partial observability
- **Physics-based verifiable rewards** — no LLM judge
- **Self-adapting curriculum** — diagnoses weaknesses, doesn't just escalate
- **Two Colab notebooks** — T4 (fp16, 2h) and A100 (bf16, 2.5h)
- **OpenEnv compliant** — standard HTTP interface, Docker deployable

---

## The Number That Matters

At the start of training, the model produces random JSON. Emergency flights land last. Conflicts are everywhere. Reward: **0.05**.

At the end of training, the model negotiates runway conflicts across two agents, handles emergency re-sequencing, respects ATFM network constraints, and explains its reasoning.

Reward: **0.71**.

That journey — 0.05 to 0.71 — happens in two hours, on a free GPU, with an open-source 1.5B model.

That's what we set out to prove: that long-horizon multi-agent planning is not a problem that requires trillion-parameter models or infinite compute. It requires the right reward design, the right training signal, and the right architecture.

---

## Try It

```bash
# Heuristic baseline (no GPU needed)
python multi_agent/inference.py --task bengaluru_irrops_hard

# Full training (Colab T4)
# Open train_t4.ipynb in Colab → Run All

# Fast 30-min demo (no repo imports needed)
# Open train_quick.ipynb in Colab → Run All

# Training CLI
python training/train_grpo.py --easy --episodes 80   # ~1 hour, AMAN+DMAN
python training/train_grpo.py --episodes 150          # ~2 hours, full
```

---

## One Last Thing

My friend never did beat that ATC game.

But we built a 1.5B model that handles four Indian airports, manages ICU surges it's never seen, and — when given the bengaluru_irrops_hard scenario with an emergency arrival, a medical departure, and a dual-runway IRROPS — sequences every flight correctly 94% of the time.

I think that counts.

---

*Built at the hackathon. Open source. MIT licensed.*

*AMAN says: runway 09L is clear.*
*DMAN says: MED208 is holding — ready when you are.*
*ADAPT says: this looks like it might be an ICU.*

*They're all right.*

---

**GitHub**: [adapt-atc-final](https://github.com/GTsingh600/adapt-atc-final) · **License**: MIT · **Model**: Qwen2.5-1.5B-Instruct
