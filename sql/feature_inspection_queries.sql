-- CourtVision ML Phase 4 feature inspection (PostgreSQL)
-- Run after build_features.py --load (and optionally --export):
--
--   Get-Content sql/feature_inspection_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
--
-- Edit season / feature_set_version / train_fraction in each params CTE to match your run.
-- Null rolling features early in the season are expected (no prior games yet).

-- =============================================================================
-- 1. Row count (gold vs shots)
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '1_row_count' AS check_name,
    p.season,
    p.feature_set_version,
    (SELECT COUNT(*)
     FROM shots AS s
     INNER JOIN games AS gl ON gl.game_id = s.game_id
     WHERE gl.season = p.season) AS shots_row_count,
    (SELECT COUNT(*)
     FROM gold_shot_features AS g
     WHERE g.season = p.season
       AND g.feature_set_version = p.feature_set_version) AS gold_row_count,
    (SELECT COUNT(*)
     FROM gold_shot_features AS g
     WHERE g.season = p.season
       AND g.feature_set_version = p.feature_set_version)
        - (SELECT COUNT(*)
           FROM shots AS s
           INNER JOIN games AS gl ON gl.game_id = s.game_id
           WHERE gl.season = p.season) AS row_count_delta
FROM params AS p;

-- =============================================================================
-- 2. Duplicate shot_id
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '2_duplicate_shot_id' AS check_name,
    g.season,
    g.feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT g.shot_id) AS distinct_shot_ids,
    COUNT(*) - COUNT(DISTINCT g.shot_id) AS duplicate_shot_id_rows
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- Duplicate shot_id detail (should return 0 rows when healthy)
WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    g.shot_id,
    COUNT(*) AS row_count
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.shot_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC, g.shot_id
LIMIT 20;

-- =============================================================================
-- 3. Missing target (shot_made_flag)
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '3_missing_target' AS check_name,
    g.season,
    g.feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE g.shot_made_flag IS NULL) AS missing_shot_made_flag,
    COUNT(*) FILTER (WHERE g.shot_made_flag IS NOT NULL) AS with_shot_made_flag
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- =============================================================================
-- 4. shot_value only 2 or 3
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '4_shot_value_distribution' AS check_name,
    g.season,
    g.feature_set_version,
    g.shot_value,
    COUNT(*) AS row_count
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version, g.shot_value
ORDER BY g.shot_value;

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '4_invalid_shot_value' AS check_name,
    g.season,
    g.feature_set_version,
    COUNT(*) FILTER (WHERE g.shot_value NOT IN (2, 3)) AS invalid_shot_value_rows
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- =============================================================================
-- 5. Date range
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '5_date_range' AS check_name,
    g.season,
    g.feature_set_version,
    MIN(g.game_date) AS min_game_date,
    MAX(g.game_date) AS max_game_date,
    COUNT(DISTINCT g.game_id) AS distinct_games,
    COUNT(*) AS total_rows
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- =============================================================================
-- 6. Train / test date split (mirrors build_features.py 80/20 by game date)
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version,
        0.8::numeric AS train_fraction
),
game_dates AS (
    SELECT
        g.game_id,
        MIN(g.game_date) AS game_date
    FROM gold_shot_features AS g
    CROSS JOIN params AS p
    WHERE g.season = p.season
      AND g.feature_set_version = p.feature_set_version
    GROUP BY g.game_id
),
ranked_games AS (
    SELECT
        game_id,
        game_date,
        ROW_NUMBER() OVER (ORDER BY game_date, game_id) AS game_rn,
        COUNT(*) OVER () AS total_games
    FROM game_dates
),
split_games AS (
    SELECT
        rg.game_id,
        rg.game_date,
        CASE
            WHEN rg.game_rn <= GREATEST(
                1,
                LEAST(
                    rg.total_games - 1,
                    FLOOR(rg.total_games * p.train_fraction)::integer
                )
            )
            THEN 'train'
            ELSE 'test'
        END AS split_set
    FROM ranked_games AS rg
    CROSS JOIN params AS p
),
split_shots AS (
    SELECT
        sg.split_set,
        g.shot_id,
        g.game_id,
        g.game_date
    FROM gold_shot_features AS g
    INNER JOIN split_games AS sg ON sg.game_id = g.game_id
    CROSS JOIN params AS p
    WHERE g.season = p.season
      AND g.feature_set_version = p.feature_set_version
)
SELECT
    '6_train_test_split' AS check_name,
    split_set,
    COUNT(DISTINCT game_id) AS games,
    COUNT(*) AS shot_rows,
    MIN(game_date) AS min_game_date,
    MAX(game_date) AS max_game_date
FROM split_shots
GROUP BY split_set
ORDER BY split_set;

-- Train/test boundary sanity: overlapping game count (expect 0)
WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version,
        0.8::numeric AS train_fraction
),
game_dates AS (
    SELECT g.game_id, MIN(g.game_date) AS game_date
    FROM gold_shot_features AS g
    CROSS JOIN params AS p
    WHERE g.season = p.season
      AND g.feature_set_version = p.feature_set_version
    GROUP BY g.game_id
),
ranked_games AS (
    SELECT
        game_id,
        ROW_NUMBER() OVER (ORDER BY game_date, game_id) AS game_rn,
        COUNT(*) OVER () AS total_games
    FROM game_dates
),
split_games AS (
    SELECT
        rg.game_id,
        CASE
            WHEN rg.game_rn <= GREATEST(
                1,
                LEAST(
                    rg.total_games - 1,
                    FLOOR(rg.total_games * p.train_fraction)::integer
                )
            )
            THEN 'train'
            ELSE 'test'
        END AS split_set
    FROM ranked_games AS rg
    CROSS JOIN params AS p
)
SELECT
    '6_train_test_overlap' AS check_name,
    COUNT(*) AS overlapping_game_ids
FROM split_games AS train_g
INNER JOIN split_games AS test_g
    ON train_g.game_id = test_g.game_id
   AND train_g.split_set = 'train'
   AND test_g.split_set = 'test';

-- =============================================================================
-- 7. Missing rolling feature counts
--    (nulls early in season are expected — no prior games for player/team yet)
-- =============================================================================

WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '7_missing_rolling_features' AS check_name,
    g.season,
    g.feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE g.player_recent_fg_pct_5 IS NULL) AS missing_player_recent_fg_pct_5,
    COUNT(*) FILTER (WHERE g.player_recent_fg3_pct_5 IS NULL) AS missing_player_recent_fg3_pct_5,
    COUNT(*) FILTER (WHERE g.player_recent_fga_5 IS NULL) AS missing_player_recent_fga_5,
    COUNT(*) FILTER (WHERE g.player_recent_fg3a_5 IS NULL) AS missing_player_recent_fg3a_5,
    COUNT(*) FILTER (WHERE g.player_recent_minutes_5 IS NULL) AS missing_player_recent_minutes_5,
    COUNT(*) FILTER (WHERE g.player_recent_points_5 IS NULL) AS missing_player_recent_points_5,
    COUNT(*) FILTER (WHERE g.team_recent_off_eff_proxy_5 IS NULL) AS missing_team_recent_off_eff_proxy_5,
    COUNT(*) FILTER (WHERE g.team_recent_pace_proxy_5 IS NULL) AS missing_team_recent_pace_proxy_5,
    COUNT(*) FILTER (WHERE g.team_recent_fg_pct_5 IS NULL) AS missing_team_recent_fg_pct_5,
    COUNT(*) FILTER (WHERE g.team_recent_three_point_rate_5 IS NULL) AS missing_team_recent_three_point_rate_5,
    COUNT(*) FILTER (WHERE g.team_recent_fga_5 IS NULL) AS missing_team_recent_fga_5,
    COUNT(*) FILTER (WHERE g.team_recent_points_5 IS NULL) AS missing_team_recent_points_5,
    COUNT(*) FILTER (WHERE g.team_recent_turnovers_5 IS NULL) AS missing_team_recent_turnovers_5,
    COUNT(*) FILTER (WHERE g.opp_recent_points_allowed_5 IS NULL) AS missing_opp_recent_points_allowed_5,
    COUNT(*) FILTER (WHERE g.opp_recent_fg_pct_allowed_5 IS NULL) AS missing_opp_recent_fg_pct_allowed_5,
    COUNT(*) FILTER (WHERE g.opp_recent_three_point_rate_allowed_5 IS NULL)
        AS missing_opp_recent_three_point_rate_allowed_5,
    COUNT(*) FILTER (WHERE g.opp_recent_pace_proxy_5 IS NULL) AS missing_opp_recent_pace_proxy_5,
    COUNT(*) FILTER (WHERE g.opp_recent_fga_allowed_5 IS NULL) AS missing_opp_recent_fga_allowed_5
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- Rolling fill rates (percent non-null)
WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '7_rolling_fill_rates_pct' AS check_name,
    g.season,
    g.feature_set_version,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE g.player_recent_fg_pct_5 IS NOT NULL) / NULLIF(COUNT(*), 0),
        1
    ) AS player_recent_fg_pct_5_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE g.team_recent_fg_pct_5 IS NOT NULL) / NULLIF(COUNT(*), 0),
        1
    ) AS team_recent_fg_pct_5_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE g.opp_recent_fg_pct_allowed_5 IS NOT NULL) / NULLIF(COUNT(*), 0),
        1
    ) AS opp_recent_fg_pct_allowed_5_pct
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY g.season, g.feature_set_version;

-- Missing player rolling by month (early-season null concentration)
WITH params AS (
    SELECT
        '2024-25'::varchar AS season,
        'base_v1'::varchar AS feature_set_version
)
SELECT
    '7_missing_player_rolling_by_month' AS check_name,
    DATE_TRUNC('month', g.game_date)::date AS game_month,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE g.player_recent_fg_pct_5 IS NULL) AS missing_player_recent_fg_pct_5,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE g.player_recent_fg_pct_5 IS NULL) / NULLIF(COUNT(*), 0),
        1
    ) AS missing_player_recent_fg_pct_5_pct
FROM gold_shot_features AS g
CROSS JOIN params AS p
WHERE g.season = p.season
  AND g.feature_set_version = p.feature_set_version
GROUP BY DATE_TRUNC('month', g.game_date)
ORDER BY game_month;
