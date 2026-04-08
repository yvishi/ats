# Benchmark Results — ATC Optimization OpenEnv

## Setup
- Hardware: 2 vCPU / 8 GB RAM
- Model: heuristic-baseline
- Run date: 2026-04-08

## Scores

| Task | Difficulty | Random Baseline | Our Agent | Improvement |
|------|------------|----------------|-----------|-------------|
| Delhi Monsoon Recovery | Easy | 0.21 | 0.9446 | 0.7346 |
| Mumbai Bank Balance | Medium | 0.18 | 0.9900 | 0.8100 |
| Bengaluru IRROPS | Hard | 0.12 | 0.8615 | 0.7415 |
| Hyderabad Cargo Crunch | Hard | 0.15 | 0.8576 | 0.7076 |
| **Average** | | **0.165** | **0.9134** | **0.7484** |

## Why Random Baseline Scores Low

A random agent assigns slots without respecting:
- Wake turbulence separation (Heavy/Medium/Light class constraints)
- Emergency flight priority overrides
- Airline equity / bank balance fairness
- Single-runway sequencing under capacity constraints

The grader uses a 3-layer gated composite score. An agent must pass all separation constraints before partial credit is given for efficiency metrics. This makes the environment non-trivial: random guessing cannot get above ~0.22 even on the easy task.

## Runtime

Total inference.py runtime: 11.69 seconds
All 4 tasks complete well within the 20-minute limit on 2 vCPU / 8 GB RAM.
