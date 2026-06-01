-- CourtVision ML Phase 2 inspection queries (PostgreSQL)
-- Run after load_data.py and compare row counts to loader logs.
--
--   Get-Content sql/inspection_queries.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
--
-- Expected 2024-25 ballpark counts (after clean + load):
--   teams 30 | players ~570 | games ~1230 | shots ~220k
--   player_game_logs ~26k | team_game_logs ~2.5k | play_by_play ~574k

-- =============================================================================
-- 1. Row counts
-- =============================================================================

SELECT 'teams' AS table_name, COUNT(*) AS row_count FROM teams
UNION ALL SELECT 'players', COUNT(*) FROM players
UNION ALL SELECT 'games', COUNT(*) FROM games
UNION ALL SELECT 'shots', COUNT(*) FROM shots
UNION ALL SELECT 'player_game_logs', COUNT(*) FROM player_game_logs
UNION ALL SELECT 'team_game_logs', COUNT(*) FROM team_game_logs
UNION ALL SELECT 'play_by_play', COUNT(*) FROM play_by_play
ORDER BY table_name;


-- =============================================================================
-- 2. Missing values (key columns)
-- =============================================================================

-- shots: game_id, game_event_id, player_id, team_id, shot_made_flag, game_date
SELECT
    'shots' AS table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) AS missing_game_id,
    COUNT(*) FILTER (WHERE game_event_id IS NULL) AS missing_game_event_id,
    COUNT(*) FILTER (WHERE player_id IS NULL) AS missing_player_id,
    COUNT(*) FILTER (WHERE team_id IS NULL) AS missing_team_id,
    COUNT(*) FILTER (WHERE shot_made_flag IS NULL) AS missing_shot_made_flag,
    COUNT(*) FILTER (WHERE game_date IS NULL) AS missing_game_date,
    COUNT(*) AS total_rows
FROM shots;

-- player_game_logs: game_id, player_id, team_id, game_date
SELECT
    'player_game_logs' AS table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) AS missing_game_id,
    COUNT(*) FILTER (WHERE player_id IS NULL) AS missing_player_id,
    COUNT(*) FILTER (WHERE team_id IS NULL) AS missing_team_id,
    COUNT(*) FILTER (WHERE game_date IS NULL) AS missing_game_date,
    COUNT(*) AS total_rows
FROM player_game_logs;

-- team_game_logs: game_id, team_id, game_date
SELECT
    'team_game_logs' AS table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) AS missing_game_id,
    COUNT(*) FILTER (WHERE team_id IS NULL) AS missing_team_id,
    COUNT(*) FILTER (WHERE game_date IS NULL) AS missing_game_date,
    COUNT(*) AS total_rows
FROM team_game_logs;

-- play_by_play: game_id, action_number, period, game_clock
SELECT
    'play_by_play' AS table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) AS missing_game_id,
    COUNT(*) FILTER (WHERE action_number IS NULL) AS missing_action_number,
    COUNT(*) FILTER (WHERE period IS NULL) AS missing_period,
    COUNT(*) FILTER (WHERE game_clock IS NULL OR BTRIM(game_clock) = '') AS missing_game_clock,
    COUNT(*) AS total_rows
FROM play_by_play;


-- =============================================================================
-- 3. Duplicate keys
-- =============================================================================

-- shots: game_id + game_event_id + player_id
SELECT
    'shots' AS table_name,
    COUNT(*) AS duplicate_key_groups,
    COALESCE(SUM(row_count - 1), 0)::BIGINT AS extra_duplicate_rows
FROM (
    SELECT COUNT(*) AS row_count
    FROM shots
    GROUP BY game_id, game_event_id, player_id
    HAVING COUNT(*) > 1
) AS dupes;

-- player_game_logs: game_id + player_id
SELECT
    'player_game_logs' AS table_name,
    COUNT(*) AS duplicate_key_groups,
    COALESCE(SUM(row_count - 1), 0)::BIGINT AS extra_duplicate_rows
FROM (
    SELECT COUNT(*) AS row_count
    FROM player_game_logs
    GROUP BY game_id, player_id
    HAVING COUNT(*) > 1
) AS dupes;

-- team_game_logs: game_id + team_id
SELECT
    'team_game_logs' AS table_name,
    COUNT(*) AS duplicate_key_groups,
    COALESCE(SUM(row_count - 1), 0)::BIGINT AS extra_duplicate_rows
FROM (
    SELECT COUNT(*) AS row_count
    FROM team_game_logs
    GROUP BY game_id, team_id
    HAVING COUNT(*) > 1
) AS dupes;

-- play_by_play: game_id + action_number
SELECT
    'play_by_play' AS table_name,
    COUNT(*) AS duplicate_key_groups,
    COALESCE(SUM(row_count - 1), 0)::BIGINT AS extra_duplicate_rows
FROM (
    SELECT COUNT(*) AS row_count
    FROM play_by_play
    GROUP BY game_id, action_number
    HAVING COUNT(*) > 1
) AS dupes;

-- Sample duplicate natural keys (if any)
SELECT 'shots sample dupes' AS label, game_id, game_event_id, player_id, COUNT(*) AS row_count
FROM shots
GROUP BY game_id, game_event_id, player_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC
LIMIT 10;

SELECT 'player_game_logs sample dupes' AS label, game_id, player_id, COUNT(*) AS row_count
FROM player_game_logs
GROUP BY game_id, player_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC
LIMIT 10;

SELECT 'team_game_logs sample dupes' AS label, game_id, team_id, COUNT(*) AS row_count
FROM team_game_logs
GROUP BY game_id, team_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC
LIMIT 10;

SELECT 'play_by_play sample dupes' AS label, game_id, action_number, COUNT(*) AS row_count
FROM play_by_play
GROUP BY game_id, action_number
HAVING COUNT(*) > 1
ORDER BY row_count DESC
LIMIT 10;


-- =============================================================================
-- 4. Date coverage
-- =============================================================================

-- games: min game_date, max game_date, count distinct games
SELECT
    'games' AS table_name,
    MIN(game_date) AS min_game_date,
    MAX(game_date) AS max_game_date,
    COUNT(DISTINCT game_id) AS distinct_games,
    COUNT(*) AS total_rows
FROM games;

-- shots: min game_date, max game_date
SELECT
    'shots' AS table_name,
    MIN(game_date) AS min_game_date,
    MAX(game_date) AS max_game_date,
    COUNT(DISTINCT game_id) AS distinct_games,
    COUNT(*) AS total_rows
FROM shots;

-- player_game_logs: min game_date, max game_date
SELECT
    'player_game_logs' AS table_name,
    MIN(game_date) AS min_game_date,
    MAX(game_date) AS max_game_date,
    COUNT(DISTINCT game_id) AS distinct_games,
    COUNT(*) AS total_rows
FROM player_game_logs;

-- team_game_logs: min game_date, max game_date
SELECT
    'team_game_logs' AS table_name,
    MIN(game_date) AS min_game_date,
    MAX(game_date) AS max_game_date,
    COUNT(DISTINCT game_id) AS distinct_games,
    COUNT(*) AS total_rows
FROM team_game_logs;

-- Season breakdown (games)
SELECT season, season_type, COUNT(*) AS game_count
FROM games
GROUP BY season, season_type
ORDER BY season, season_type;
