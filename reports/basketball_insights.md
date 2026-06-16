# Basketball Insights — Expected Shot Value (2024-25)

**Project:** CourtVision ML  
**Phase:** 8 — Expected Shot Value and Player Evaluation  
**Season:** 2024-25  
**Model:** GRU spatial_sequence v3 (Challenger+)  
**GRU run ID:** 40fe1b8851f7423f831a77fce30b770d  
**Report date:** 2026-06-16  
**Evaluation shots:** 43,819

---

## Overview

This report translates shot-level expected value results into basketball-facing takeaways from the
held-out evaluation period. Positive **points above expected** means a player, team, or zone scored
more than the model anticipated given shot difficulty and context.

---

## Findings

### 1. Top players by total points above expected

Among players with at least 100 evaluation shots, these players added the most total scoring value relative to model expectation:

- **Kawhi Leonard (`202695`)** — 259 attempts, +44.8 total points above expected (+17.30 per 100 shots).
- **Zach LaVine (`203897`)** — 264 attempts, +40.0 total points above expected (+15.15 per 100 shots).
- **Nikola Jokić (`203999`)** — 209 attempts, +39.2 total points above expected (+18.73 per 100 shots).
- **Austin Reaves (`1630559`)** — 258 attempts, +38.6 total points above expected (+14.98 per 100 shots).
- **Ty Jerome (`1629660`)** — 110 attempts, +37.7 total points above expected (+34.26 per 100 shots).

### 2. Top players by points above expected per 100 shots

Efficiency leaders on the evaluation split (minimum 100 attempts):

- **Ty Jerome (`1629660`)** — +34.26 per 100 shots on 110 attempts (+37.7 total).
- **Keon Ellis (`1631165`)** — +29.88 per 100 shots on 106 attempts (+31.7 total).
- **Bogdan Bogdanović (`203992`)** — +27.58 per 100 shots on 119 attempts (+32.8 total).
- **Jarace Walker (`1641716`)** — +23.72 per 100 shots on 101 attempts (+24.0 total).
- **Isaiah Joe (`1630198`)** — +23.44 per 100 shots on 141 attempts (+33.0 total).

### 3. Teams with the best points above expected

Team shot-making value relative to expectation on the held-out test games:

- **LAC (`1610612746`)** — +132.3 total (+9.63 per 100 shots) across 1,374 team shots.
- **SAC (`1610612758`)** — +111.6 total (+7.14 per 100 shots) across 1,564 team shots.
- **MIL (`1610612749`)** — +85.4 total (+5.98 per 100 shots) across 1,427 team shots.
- **OKC (`1610612760`)** — +82.3 total (+5.56 per 100 shots) across 1,481 team shots.
- **PHX (`1610612756`)** — +67.3 total (+4.76 per 100 shots) across 1,414 team shots.

### 4. Teams with the best average expected shot value

Teams whose shot mix generated the highest average expected points per attempt from the model:

- **MIN (`1610612750`)** — average expected shot value 1.160 on 1,334 shots (-9.4 total above expected).
- **ATL (`1610612737`)** — average expected shot value 1.130 on 1,450 shots (+45.1 total above expected).
- **PHI (`1610612755`)** — average expected shot value 1.130 on 1,517 shots (-116.8 total above expected).
- **CHI (`1610612741`)** — average expected shot value 1.127 on 1,466 shots (+34.2 total above expected).
- **NYK (`1610612752`)** — average expected shot value 1.127 on 1,502 shots (-60.0 total above expected).

### 5. Best and worst zones by points above expected

Zone combinations with at least 100 evaluation shots:

**Best zones**

- **Mid-Range / Left Side Center(LC) / 16-24 ft.** — +8.94 per 100 shots (499 attempts).
- **In The Paint (Non-RA) / Right Side(R) / 8-16 ft.** — +7.11 per 100 shots (520 attempts).
- **In The Paint (Non-RA) / Center(C) / 8-16 ft.** — +6.03 per 100 shots (2,728 attempts).

**Worst zones**

- **Mid-Range / Right Side(R) / 16-24 ft.** — -8.63 per 100 shots (176 attempts).
- **Mid-Range / Right Side Center(RC) / 16-24 ft.** — -6.51 per 100 shots (502 attempts).
- **Above the Break 3 / Right Side Center(RC) / 24+ ft.** — -3.80 per 100 shots (4,838 attempts).

### 6. Player monthly trend

Monthly trend scan using at least 40 shots in each month:

- **Largest improvement:** Harrison Barnes (`203084`) moved from 2025-03 to 2025-04 at +41.60 per 100 shots (85 → 84 attempts; ending month rate +33.92 per 100).
- **Largest decline:** Mike Conley (`201144`) moved from 2025-03 to 2025-04 at -70.45 per 100 shots (42 → 43 attempts).

---

## Limitations

- **Held-out test split only:** Findings come from the latest ~20% of 2024-25 games by date, not the full season.
- **Model-driven expectation:** Expected shot value uses raw make probabilities from GRU spatial_sequence v3 (Challenger+) without a separate calibration pass.
- **Sample-size filters:** Player insights require at least 100 evaluation shots; zone insights require at least 100. Tiny zones (for example backcourt heaves) are excluded from zone rankings.
- **Feature scope:** Scoring features include shot geometry, game context, and rolling player/team form — not matchup-specific scouting, player tracking, lineup, or defender-distance features.
- **Evaluation volume:** This report is based on 43,819 evaluation shots for season `2024-25`.

