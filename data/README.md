# CourtVision Data

CourtVision uses public NBA Stats endpoints for the 2024-25 season. Raw and processed datasets are intentionally excluded from git; only this guide and collection metadata are tracked.

## Sources

| Dataset | Endpoint | Recorded rows |
|---------|----------|--------------:|
| Shot charts | `shotchartdetail` | 219,527 |
| Player game logs | `playergamelogs` | 26,306 |
| Team game logs | `teamgamelogs` | 2,460 |
| Play-by-play | `playbyplayv3` | 606,536 |

Collection timestamps, source URLs, and generated paths are recorded in `data/metadata/data_collection_metadata.json`.

## Local layout

```text
data/
|-- metadata/
|   `-- data_collection_metadata.json
|-- raw/
|   |-- shots/
|   |-- player_game_logs/
|   |-- team_game_logs/
|   `-- play_by_play/
`-- processed/
    `-- features/
        |-- train_shot_features_2024-25.parquet
        `-- test_shot_features_2024-25.parquet
```

The collector writes both CSV and Parquet raw files. The PostgreSQL loading and feature pipeline produces the processed train/test exports.

## Rebuild

Run commands from the repository root with `PYTHONPATH=src`:

```powershell
python -m courtvision.data.collect
python -m courtvision.data.load_data --season 2024-25
python -m courtvision.data.build_features --season 2024-25 --load --inspect --export
```

The collection process makes many requests, especially for play-by-play, and should respect the source service's availability and rate limits.

## Split policy

The feature export uses games ordered by date:

- Earliest approximately 80%: 175,708 training shots from 984 games
- Latest approximately 20%: 43,819 test shots from 246 games

Games do not cross splits. Neural models split the exported training set again by game date for early stopping.

## Data handling

- Treat raw downloads as immutable source snapshots.
- Do not commit raw files, processed Parquet files, local databases, or caches.
- Preserve collection metadata when refreshing a season.
- Validate schemas and critical constraints before loading.
- Use shifted rolling features and prior-event joins to prevent target leakage.
- Confirm licensing and endpoint terms before redistributing source data.

No private or sensitive personal data is intentionally collected. Player and game identifiers are public sports records from the source endpoints.
