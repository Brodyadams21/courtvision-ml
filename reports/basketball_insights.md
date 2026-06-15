# Basketball Insights — Expected Shot Value (2024-25)

**Project:** CourtVision ML  
**Phase:** 8 — Expected Shot Value and Player Evaluation  
**Season:** 2024-25  
**Model:** `courtvision-shot-make-model` (`Candidate`)  
**Report date:** 2026-06-15  
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

- **Brice Sensabaugh (`1641729`)** — 158 attempts, +42.1 total points above expected (+26.66 per 100 shots).
- **Austin Reaves (`1630559`)** — 258 attempts, +41.9 total points above expected (+16.24 per 100 shots).
- **Zach LaVine (`203897`)** — 264 attempts, +41.7 total points above expected (+15.78 per 100 shots).
- **Kawhi Leonard (`202695`)** — 259 attempts, +41.0 total points above expected (+15.84 per 100 shots).
- **Ty Jerome (`1629660`)** — 110 attempts, +39.1 total points above expected (+35.56 per 100 shots).

### 2. Top players by points above expected per 100 shots

Efficiency leaders on the evaluation split (minimum 100 attempts):

- **Ty Jerome (`1629660`)** — +35.56 per 100 shots on 110 attempts (+39.1 total).
- **Keon Ellis (`1631165`)** — +31.00 per 100 shots on 106 attempts (+32.9 total).
- **Bogdan Bogdanović (`203992`)** — +28.18 per 100 shots on 119 attempts (+33.5 total).
- **Brice Sensabaugh (`1641729`)** — +26.66 per 100 shots on 158 attempts (+42.1 total).
- **Christian Braun (`1631128`)** — +25.77 per 100 shots on 143 attempts (+36.9 total).

### 3. Teams with the best points above expected

Team shot-making value relative to expectation on the held-out test games:

- **LAC (`1610612746`)** — +136.4 total (+9.92 per 100 shots) across 1,374 team shots.
- **SAC (`1610612758`)** — +116.8 total (+7.47 per 100 shots) across 1,564 team shots.
- **MIL (`1610612749`)** — +100.3 total (+7.03 per 100 shots) across 1,427 team shots.
- **OKC (`1610612760`)** — +86.1 total (+5.82 per 100 shots) across 1,481 team shots.
- **BOS (`1610612738`)** — +77.3 total (+5.42 per 100 shots) across 1,426 team shots.

### 4. Teams with the best average expected shot value

Teams whose shot mix generated the highest average expected points per attempt from the model:

- **MIN (`1610612750`)** — average expected shot value 1.140 on 1,334 shots (+17.7 total above expected).
- **ATL (`1610612737`)** — average expected shot value 1.131 on 1,450 shots (+44.0 total above expected).
- **NYK (`1610612752`)** — average expected shot value 1.131 on 1,502 shots (-66.5 total above expected).
- **CHI (`1610612741`)** — average expected shot value 1.115 on 1,466 shots (+52.1 total above expected).
- **PHI (`1610612755`)** — average expected shot value 1.110 on 1,517 shots (-87.1 total above expected).

### 5. Best and worst zones by points above expected

Zone combinations with at least 100 evaluation shots:

**Best zones**

- **Mid-Range / Left Side Center(LC) / 16-24 ft.** — +8.10 per 100 shots (499 attempts).
- **In The Paint (Non-RA) / Right Side(R) / 8-16 ft.** — +7.68 per 100 shots (520 attempts).
- **Left Corner 3 / Left Side(L) / 24+ ft.** — +4.88 per 100 shots (2,499 attempts).

**Worst zones**

- **Mid-Range / Right Side(R) / 16-24 ft.** — -7.34 per 100 shots (176 attempts).
- **Mid-Range / Right Side Center(RC) / 16-24 ft.** — -5.59 per 100 shots (502 attempts).
- **Mid-Range / Left Side(L) / 8-16 ft.** — -1.73 per 100 shots (856 attempts).

### 6. Player monthly trend

Monthly trend scan using at least 40 shots in each month:

- **Largest improvement:** Harrison Barnes (`203084`) moved from 2025-03 to 2025-04 at +45.44 per 100 shots (85 → 84 attempts; ending month rate +39.07 per 100).
- **Largest decline:** Mike Conley (`201144`) moved from 2025-03 to 2025-04 at -69.74 per 100 shots (42 → 43 attempts).

---

## Limitations

- **Held-out test split only:** Findings come from the latest ~20% of 2024-25 games by date, not the full season.
- **Model-driven expectation:** Expected shot value uses raw LightGBM make probabilities from `courtvision-shot-make-model` (`Candidate`) without a separate calibration pass.
- **Sample-size filters:** Player insights require at least 100 evaluation shots; zone insights require at least 100. Tiny zones (for example backcourt heaves) are excluded from zone rankings.
- **Feature scope:** The registered LightGBM Candidate uses shot geometry, game context, and rolling player/team form — not matchup-specific scouting, player tracking, lineup, or defender-distance features.
- **Evaluation volume:** This report is based on 43,819 evaluation shots for season `2024-25`.

