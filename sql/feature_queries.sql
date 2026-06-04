-- CourtVision ML gold shot feature inspection (PostgreSQL)
-- Run after build_features.py --load:
--
--   Get-Content sql/feature_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
--
-- Set season / version to match your load (defaults match build_features.py).

-- =============================================================================
-- 1. Row counts: gold vs shots (same season)
-- =============================================================================

SELECT
    g.season,
    g.feature_set_version,
    (SELECT COUNT(*) FROM shots s
     INNER JOIN games gl ON gl.game_id = s.game_id
     WHERE gl.season = g.season) AS shots_row_count,
    COUNT(*) AS gold_row_count,
    COUNT(*) - (SELECT COUNT(*) FROM shots s
                INNER JOIN games gl ON gl.game_id = s.game_id
                WHERE gl.season = g.season) AS row_count_delta
FROM gold_shot_features AS g
GROUP BY g.season, g.feature_set_version
ORDER BY g.season, g.feature_set_version;

-- =============================================================================
-- 2. shot_id uniqueness (per feature_set_version)
-- =============================================================================

SELECT
    season,
    feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT shot_id) AS distinct_shot_ids,
    COUNT(*) - COUNT(DISTINCT shot_id) AS duplicate_shot_ids
FROM gold_shot_features
GROUP BY season, feature_set_version
ORDER BY season, feature_set_version;

-- =============================================================================
-- 3. shot_value must be 2 or 3
-- =============================================================================

SELECT
    season,
    feature_set_version,
    shot_value,
    COUNT(*) AS row_count
FROM gold_shot_features
GROUP BY season, feature_set_version, shot_value
ORDER BY season, feature_set_version, shot_value;

SELECT
    season,
    feature_set_version,
    COUNT(*) FILTER (WHERE shot_value NOT IN (2, 3)) AS invalid_shot_value_rows
FROM gold_shot_features
GROUP BY season, feature_set_version;

-- =============================================================================
-- 4. Required fields not missing
-- =============================================================================

SELECT
    season,
    feature_set_version,
    COUNT(*) FILTER (WHERE shot_made_flag IS NULL) AS missing_shot_made_flag,
    COUNT(*) FILTER (WHERE opponent_team_id IS NULL) AS missing_opponent_team_id,
    COUNT(*) FILTER (WHERE is_home IS NULL) AS missing_is_home,
    COUNT(*) AS total_rows
FROM gold_shot_features
GROUP BY season, feature_set_version;

-- =============================================================================
-- 5. is_home sanity vs games
-- =============================================================================

SELECT
    g.season,
    g.feature_set_version,
    COUNT(*) FILTER (
        WHERE g.is_home IS TRUE AND g.team_id <> gl.home_team_id
    ) AS is_home_true_but_away,
    COUNT(*) FILTER (
        WHERE g.is_home IS FALSE AND g.team_id <> gl.away_team_id
    ) AS is_home_false_but_home,
    COUNT(*) AS total_rows
FROM gold_shot_features AS g
INNER JOIN games AS gl ON gl.game_id = g.game_id
GROUP BY g.season, g.feature_set_version;

-- =============================================================================
-- 6. Sample rows
-- =============================================================================

SELECT
    shot_id,
    game_id,
    team_id,
    opponent_team_id,
    shot_value,
    shot_made_flag,
    is_home,
    is_corner_three,
    shot_distance,
    period,
    seconds_remaining_game
FROM gold_shot_features
ORDER BY game_date, game_id, game_event_id
LIMIT 10;

-- =============================================================================
-- 7. Team rolling features (previous 5 games, excludes current game)
-- =============================================================================

SELECT
    season,
    feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE team_recent_fg_pct_5 IS NOT NULL) AS with_team_fg_pct,
    COUNT(*) FILTER (WHERE team_recent_fg_pct_5 IS NULL) AS missing_team_fg_pct,
    ROUND(AVG(team_recent_off_eff_proxy_5)::numeric, 3) AS avg_off_eff_proxy,
    ROUND(AVG(team_recent_pace_proxy_5)::numeric, 2) AS avg_pace_proxy,
    ROUND(AVG(team_recent_three_point_rate_5)::numeric, 3) AS avg_three_point_rate
FROM gold_shot_features
GROUP BY season, feature_set_version
ORDER BY season, feature_set_version;

-- =============================================================================
-- 8. Opponent rolling features (what defense allowed, prior 5 games)
-- =============================================================================

SELECT
    season,
    feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE opp_recent_fg_pct_allowed_5 IS NOT NULL) AS with_opp_fg_pct,
    COUNT(*) FILTER (WHERE opp_recent_fg_pct_allowed_5 IS NULL) AS missing_opp_fg_pct,
    ROUND(AVG(opp_recent_points_allowed_5)::numeric, 2) AS avg_points_allowed,
    ROUND(AVG(opp_recent_pace_proxy_5)::numeric, 2) AS avg_pace_allowed,
    ROUND(AVG(opp_recent_fga_allowed_5)::numeric, 2) AS avg_fga_allowed
FROM gold_shot_features
GROUP BY season, feature_set_version
ORDER BY season, feature_set_version;

-- =============================================================================
-- 9. Score margin coverage (optional; nulls expected when PBP scores missing)
-- =============================================================================

SELECT
    season,
    feature_set_version,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE score_margin IS NOT NULL) AS with_score_margin,
    COUNT(*) FILTER (WHERE score_margin IS NULL) AS missing_score_margin,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE score_margin IS NOT NULL) / NULLIF(COUNT(*), 0),
        1
    ) AS score_margin_pct,
    MIN(score_margin) AS min_margin,
    MAX(score_margin) AS max_margin
FROM gold_shot_features
GROUP BY season, feature_set_version
ORDER BY season, feature_set_version;
