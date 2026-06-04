-- CourtVision ML database schema (PostgreSQL)
-- Run once on setup or after schema changes:
--   Get-Content sql/schema.sql | docker compose exec -T postgres psql -U courtvision_user -d courtvision_ml
--
-- Day-to-day reloads: use load_data.py (it truncates loaded tables before insert).
--
-- Load order (respect foreign keys):
--   1. teams
--   2. players
--   3. games          -- home/away from team_game_logs MATCHUP (see games table comment)
--   4. shots
--   5. player_game_logs
--   6. team_game_logs
--   7. play_by_play    -- person_id is not FK-constrained (coaches, officials, missing players)
--   ML tables (gold_shot_features, shot_features, shot_predictions, player_evaluation)
--   stay empty until later phases.

-- ---------------------------------------------------------------------------
-- 1. Drop old tables
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS shot_predictions CASCADE;
DROP TABLE IF EXISTS gold_shot_features CASCADE;
DROP TABLE IF EXISTS shot_features CASCADE;
DROP TABLE IF EXISTS player_evaluation CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS play_by_play CASCADE;
DROP TABLE IF EXISTS player_game_logs CASCADE;
DROP TABLE IF EXISTS team_game_logs CASCADE;
DROP TABLE IF EXISTS shots CASCADE;
DROP TABLE IF EXISTS games CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS teams CASCADE;

-- ---------------------------------------------------------------------------
-- 2. Identity tables
-- ---------------------------------------------------------------------------

CREATE TABLE teams (
    team_id         INTEGER PRIMARY KEY,
    abbreviation    VARCHAR(3)  NOT NULL,
    full_name       VARCHAR(80) NOT NULL,
    nickname        VARCHAR(40),
    city            VARCHAR(40),
    state           VARCHAR(40),
    year_founded    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,
    full_name       VARCHAR(80) NOT NULL,
    first_name      VARCHAR(40),
    last_name       VARCHAR(40),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE games (
    game_id         VARCHAR(10) PRIMARY KEY,
    season          VARCHAR(7)  NOT NULL,
    season_type     VARCHAR(32) NOT NULL DEFAULT 'Regular Season',
    game_date       DATE        NOT NULL,
    home_team_id    INTEGER     NOT NULL REFERENCES teams (team_id),
    away_team_id    INTEGER     NOT NULL REFERENCES teams (team_id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT games_distinct_teams CHECK (home_team_id <> away_team_id)
);

COMMENT ON TABLE games IS
    'Derive home_team_id and away_team_id from team_game_logs before insert. '
    'MATCHUP uses the row team as the first tricode: "LAL vs. BOS" => LAL home, BOS away; '
    '"LAL @ BOS" => LAL away, BOS home.';

-- ---------------------------------------------------------------------------
-- 3. Basketball data tables
-- ---------------------------------------------------------------------------

CREATE TABLE shots (
    shot_id                 BIGSERIAL PRIMARY KEY,
    game_id                 VARCHAR(10) NOT NULL REFERENCES games (game_id),
    game_event_id           INTEGER     NOT NULL,
    player_id               INTEGER     NOT NULL REFERENCES players (player_id),
    team_id                 INTEGER     NOT NULL REFERENCES teams (team_id),
    period                  SMALLINT    NOT NULL,
    minutes_remaining       SMALLINT    NOT NULL,
    seconds_remaining       SMALLINT    NOT NULL,
    event_type              VARCHAR(40),
    action_type             VARCHAR(80),
    shot_type               VARCHAR(40),
    shot_zone_basic         VARCHAR(40),
    shot_zone_area          VARCHAR(40),
    shot_zone_range         VARCHAR(40),
    shot_distance           SMALLINT,
    loc_x                   SMALLINT,
    loc_y                   SMALLINT,
    shot_attempted_flag     BOOLEAN     NOT NULL DEFAULT TRUE,
    shot_made_flag          BOOLEAN     NOT NULL,
    game_date               DATE        NOT NULL,
    home_team_abbreviation  VARCHAR(3),
    away_team_abbreviation  VARCHAR(3),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT shots_natural_key UNIQUE (game_id, game_event_id, player_id)
);

COMMENT ON TABLE shots IS
    'Natural key (game_id, game_event_id, player_id) matches collect.py dedupe keys. '
    'ML tables reference shot_id.';

CREATE TABLE player_game_logs (
    season_year             VARCHAR(7)  NOT NULL,
    game_id                 VARCHAR(10) NOT NULL REFERENCES games (game_id),
    player_id               INTEGER     NOT NULL REFERENCES players (player_id),
    team_id                 INTEGER     NOT NULL REFERENCES teams (team_id),
    game_date               DATE        NOT NULL,
    matchup                 VARCHAR(20) NOT NULL,
    win_loss                CHAR(1)     NOT NULL,
    minutes                 NUMERIC(5, 1),
    field_goals_made        SMALLINT,
    field_goals_attempted   SMALLINT,
    field_goal_pct          NUMERIC(5, 3),
    three_pointers_made     SMALLINT,
    three_pointers_attempted SMALLINT,
    three_point_pct         NUMERIC(5, 3),
    free_throws_made        SMALLINT,
    free_throws_attempted   SMALLINT,
    free_throw_pct          NUMERIC(5, 3),
    offensive_rebounds      SMALLINT,
    defensive_rebounds      SMALLINT,
    rebounds                SMALLINT,
    assists                 SMALLINT,
    turnovers               SMALLINT,
    steals                  SMALLINT,
    blocks                  SMALLINT,
    blocked_att             SMALLINT,
    personal_fouls          SMALLINT,
    fouls_drawn             SMALLINT,
    points                  SMALLINT,
    plus_minus              SMALLINT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, player_id),
    CONSTRAINT player_game_logs_win_loss CHECK (win_loss IN ('W', 'L'))
);

CREATE TABLE team_game_logs (
    season_year             VARCHAR(7)  NOT NULL,
    game_id                 VARCHAR(10) NOT NULL REFERENCES games (game_id),
    team_id                 INTEGER     NOT NULL REFERENCES teams (team_id),
    game_date               DATE        NOT NULL,
    matchup                 VARCHAR(20) NOT NULL,
    win_loss                CHAR(1)     NOT NULL,
    minutes                 NUMERIC(5, 1),
    field_goals_made        SMALLINT,
    field_goals_attempted   SMALLINT,
    field_goal_pct          NUMERIC(5, 3),
    three_pointers_made     SMALLINT,
    three_pointers_attempted SMALLINT,
    three_point_pct         NUMERIC(5, 3),
    free_throws_made        SMALLINT,
    free_throws_attempted   SMALLINT,
    free_throw_pct          NUMERIC(5, 3),
    offensive_rebounds      SMALLINT,
    defensive_rebounds      SMALLINT,
    rebounds                SMALLINT,
    assists                 SMALLINT,
    turnovers               SMALLINT,
    steals                  SMALLINT,
    blocks                  SMALLINT,
    blocked_att             SMALLINT,
    personal_fouls          SMALLINT,
    fouls_drawn             SMALLINT,
    points                  SMALLINT,
    plus_minus              SMALLINT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, team_id),
    CONSTRAINT team_game_logs_win_loss CHECK (win_loss IN ('W', 'L'))
);

CREATE TABLE play_by_play (
    game_id                 VARCHAR(10) NOT NULL REFERENCES games (game_id),
    action_number           INTEGER     NOT NULL,
    action_id               BIGINT,
    period                  SMALLINT    NOT NULL,
    game_clock              VARCHAR(16) NOT NULL,
    team_id                 INTEGER     REFERENCES teams (team_id),
    team_tricode            VARCHAR(3),
  -- person_id intentionally has no FK: PBP includes 0, coaches, and players not yet in players
    person_id               INTEGER,
    player_name             VARCHAR(80),
    x_legacy                SMALLINT,
    y_legacy                SMALLINT,
    shot_distance           SMALLINT,
    shot_result             VARCHAR(20),
    is_field_goal           BOOLEAN,
    score_home              SMALLINT,
    score_away              SMALLINT,
    points_total            SMALLINT,
    court_location          VARCHAR(10),
    description             TEXT,
    action_type             VARCHAR(40),
    sub_type                VARCHAR(80),
    video_available         BOOLEAN,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, action_number)
);

COMMENT ON COLUMN play_by_play.person_id IS
    'NBA personId. No foreign key: may be 0, non-player, or absent from players until backfilled.';

-- ---------------------------------------------------------------------------
-- 4. Future ML tables
-- ---------------------------------------------------------------------------

CREATE TABLE shot_features (
    shot_id                 BIGINT      NOT NULL REFERENCES shots (shot_id),
    season                  VARCHAR(7)  NOT NULL,
    player_id               INTEGER     NOT NULL REFERENCES players (player_id),
    team_id                 INTEGER     NOT NULL REFERENCES teams (team_id),
    feature_set_version     VARCHAR(40) NOT NULL,
    shot_distance_ft        NUMERIC(6, 2),
    loc_x_norm              NUMERIC(8, 4),
    loc_y_norm              NUMERIC(8, 4),
    shot_zone_basic         VARCHAR(40),
    shot_zone_area          VARCHAR(40),
    shot_zone_range         VARCHAR(40),
    is_three_point          BOOLEAN,
    seconds_remaining_game  NUMERIC(8, 2),
    score_margin            SMALLINT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (shot_id, feature_set_version)
);

CREATE TABLE gold_shot_features (
    shot_id                 BIGINT      NOT NULL REFERENCES shots (shot_id),
    game_id                 VARCHAR(10) NOT NULL REFERENCES games (game_id),
    game_event_id           INTEGER     NOT NULL,
    player_id               INTEGER     NOT NULL REFERENCES players (player_id),
    team_id                 INTEGER     NOT NULL REFERENCES teams (team_id),
    opponent_team_id        INTEGER     NOT NULL REFERENCES teams (team_id),
    season                  VARCHAR(7)  NOT NULL,
    game_date               DATE        NOT NULL,
    feature_set_version     VARCHAR(40) NOT NULL,
    shot_made_flag          BOOLEAN     NOT NULL,
    shot_value              SMALLINT    NOT NULL,
    -- Base shot geometry and zones
    shot_distance           SMALLINT,
    loc_x                   SMALLINT,
    loc_y                   SMALLINT,
    abs_loc_x               SMALLINT,
    shot_angle              NUMERIC(10, 6),
    is_corner_three         BOOLEAN,
    is_home                 BOOLEAN,
    shot_zone_basic         VARCHAR(40),
    shot_zone_area          VARCHAR(40),
    shot_zone_range         VARCHAR(40),
    -- Game state
    period                  SMALLINT,
    seconds_remaining_period NUMERIC(8, 2),
    seconds_remaining_game  NUMERIC(8, 2),
    score_margin            SMALLINT,
    -- Player rolling (previous 5 games, shifted — excludes current game)
    player_recent_fg_pct_5  NUMERIC(6, 4),
    player_recent_fg3_pct_5 NUMERIC(6, 4),
    player_recent_fga_5     NUMERIC(6, 2),
    player_recent_fg3a_5    NUMERIC(6, 2),
    player_recent_minutes_5 NUMERIC(6, 2),
    player_recent_points_5  NUMERIC(6, 2),
    -- Team rolling (previous 5 games, shifted — excludes current game)
    team_recent_off_eff_proxy_5     NUMERIC(8, 4),
    team_recent_pace_proxy_5        NUMERIC(8, 4),
    team_recent_fg_pct_5            NUMERIC(6, 4),
    team_recent_three_point_rate_5  NUMERIC(6, 4),
    team_recent_fga_5               NUMERIC(6, 2),
    team_recent_points_5            NUMERIC(6, 2),
    team_recent_turnovers_5         NUMERIC(6, 2),
    -- Opponent rolling (previous 5 games allowed, shifted — excludes current game)
    opp_recent_points_allowed_5             NUMERIC(6, 2),
    opp_recent_fg_pct_allowed_5           NUMERIC(6, 4),
    opp_recent_three_point_rate_allowed_5   NUMERIC(6, 4),
    opp_recent_pace_proxy_5                 NUMERIC(8, 4),
    opp_recent_fga_allowed_5                NUMERIC(6, 2),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (shot_id, feature_set_version),
    CONSTRAINT gold_shot_features_distinct_teams CHECK (team_id <> opponent_team_id),
    CONSTRAINT gold_shot_features_shot_value CHECK (shot_value IN (2, 3))
);

COMMENT ON TABLE gold_shot_features IS
    'Model-ready shot feature table (gold layer). One row per shot per feature_set_version. '
    'Geometry, game-state, and rolling columns are nullable until the feature pipeline fills them.';

CREATE TABLE shot_predictions (
    shot_id                 BIGINT      NOT NULL REFERENCES shots (shot_id),
    model_name              VARCHAR(80) NOT NULL,
    model_version           VARCHAR(40) NOT NULL,
    predicted_make_prob     NUMERIC(6, 5) NOT NULL,
    expected_shot_value     NUMERIC(6, 4),
    predicted_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (shot_id, model_name, model_version),
    CONSTRAINT shot_predictions_prob_range CHECK (
        predicted_make_prob >= 0 AND predicted_make_prob <= 1
    )
);

CREATE TABLE player_evaluation (
    player_id               INTEGER     NOT NULL REFERENCES players (player_id),
    season                  VARCHAR(7)  NOT NULL,
    evaluation_date         DATE        NOT NULL,
    model_name              VARCHAR(80) NOT NULL,
    model_version           VARCHAR(40) NOT NULL,
    shots_taken             INTEGER,
    points_above_expected   NUMERIC(8, 3),
    avg_shot_quality        NUMERIC(6, 5),
    expected_points_per_shot NUMERIC(6, 4),
    actual_points_per_shot  NUMERIC(6, 4),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, season, evaluation_date, model_name, model_version)
);

-- ---------------------------------------------------------------------------
-- 5. Create indexes
-- ---------------------------------------------------------------------------

-- Identity
CREATE INDEX idx_teams_abbreviation ON teams (abbreviation);

CREATE INDEX idx_players_full_name ON players (full_name);

CREATE INDEX idx_games_season ON games (season);
CREATE INDEX idx_games_game_date ON games (game_date);
CREATE INDEX idx_games_home_team_id ON games (home_team_id);
CREATE INDEX idx_games_away_team_id ON games (away_team_id);

-- Shots
CREATE INDEX idx_shots_game_id ON shots (game_id);
CREATE INDEX idx_shots_player_id ON shots (player_id);
CREATE INDEX idx_shots_team_id ON shots (team_id);
CREATE INDEX idx_shots_game_date ON shots (game_date);
CREATE INDEX idx_shots_shot_made_flag ON shots (shot_made_flag);

-- Player game logs
CREATE INDEX idx_player_game_logs_player_id ON player_game_logs (player_id);
CREATE INDEX idx_player_game_logs_team_id ON player_game_logs (team_id);
CREATE INDEX idx_player_game_logs_season_year ON player_game_logs (season_year);
CREATE INDEX idx_player_game_logs_game_date ON player_game_logs (game_date);

-- Team game logs
CREATE INDEX idx_team_game_logs_team_id ON team_game_logs (team_id);
CREATE INDEX idx_team_game_logs_season_year ON team_game_logs (season_year);
CREATE INDEX idx_team_game_logs_game_date ON team_game_logs (game_date);

-- Play-by-play
CREATE INDEX idx_play_by_play_person_id ON play_by_play (person_id);
CREATE INDEX idx_play_by_play_team_id ON play_by_play (team_id);
CREATE INDEX idx_play_by_play_action_type ON play_by_play (action_type);
CREATE INDEX idx_play_by_play_is_field_goal ON play_by_play (is_field_goal);

-- ML tables
CREATE INDEX idx_gold_shot_features_game_id ON gold_shot_features (game_id);
CREATE INDEX idx_gold_shot_features_player_id ON gold_shot_features (player_id);
CREATE INDEX idx_gold_shot_features_team_id ON gold_shot_features (team_id);
CREATE INDEX idx_gold_shot_features_opponent_team_id ON gold_shot_features (opponent_team_id);
CREATE INDEX idx_gold_shot_features_season ON gold_shot_features (season);
CREATE INDEX idx_gold_shot_features_game_date ON gold_shot_features (game_date);
CREATE INDEX idx_gold_shot_features_feature_set_version ON gold_shot_features (feature_set_version);

CREATE INDEX idx_shot_features_player_id ON shot_features (player_id);
CREATE INDEX idx_shot_features_season ON shot_features (season);

CREATE INDEX idx_shot_predictions_model ON shot_predictions (model_name, model_version);

CREATE INDEX idx_player_evaluation_season ON player_evaluation (season);
CREATE INDEX idx_player_evaluation_evaluation_date ON player_evaluation (evaluation_date);
