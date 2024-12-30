# NFL EDP Analysis Data Requirements

## Data Sources

### Play-by-Play Data
- Source: nfl_data_py package
- Seasons: 2024 (current) with historical data support
- Required frequency: Weekly updates during season
- Format: Parquet files (both raw and processed)

## Required Fields
All fields listed below are required unless explicitly marked as nullable.

### Core Fields
- `game_id`: Unique identifier for each game (format: YYYY_WW_AWAY_HOME)
- `play_id`: Sequential identifier for plays within a game
- `posteam`: Team with possession
- `defteam`: Defensive team
- `week`: Game week (1-18)
- `season`: Season year
- `drive`: Drive number within game
- `down`: Down (1-4)
- `yards_gained`: Yards gained on play
- `play_type`: Type of play
- `yardline_100`: Yards from opponent's end zone
- `ydstogo`: Yards needed for first down
- `ep`: Expected points (nullable)
- `epa`: Expected points added (nullable)

### Play Types
Valid play types include:
```python
{
    'pass', 'run', 'punt', 'field_goal', 'extra_point',
    'kickoff', 'penalty', 'no_play', 'qb_kneel', 'qb_spike'
}
```

## Data Volume Expectations
- Games per season: 544 (32 teams × 17 games)
- Plays per game: 60-200 (typical range)
- Total plays per season: ~35,000-40,000

## Performance Considerations

### Data Loading
- Use parquet format for efficient storage and loading
- Implement caching for frequently accessed data
- Consider chunked processing for large historical analyses

### Memory Usage
- Typical memory requirements:
  - Raw play-by-play data: ~100MB per season
  - Processed data with metrics: ~150MB per season
  - Peak memory during processing: ~500MB

### Processing Time
Expected processing times on standard hardware:
- Data loading: 1-2 seconds per season
- Initial processing: 2-3 seconds per season
- EDP calculations: 1-2 seconds per season
- Total pipeline: 5-10 seconds per season

### Optimization Recommendations
1. Use cached data when possible
2. Process data incrementally during season
3. Parallelize drive-level calculations
4. Pre-filter data to relevant plays early in pipeline

## Data Quality Checks

### Validation Rules
1. No missing values in core fields (except nullable fields)
2. Team names must match standard abbreviations
3. Numeric fields must be within defined ranges
4. Game and drive counts must meet volume expectations

### Error Handling
1. Log all validation failures
2. Filter invalid data rather than failing
3. Generate validation reports
4. Alert on suspicious patterns

## Storage Requirements

### File Organization
```
data/
├── raw/              # Raw play-by-play data
│   └── pbp_*.parquet
├── cache/            # Cached processed data
│   └── processed_*.parquet
└── pipeline_outputs/ # Pipeline validation reports
    ├── processed_pbp_*.parquet
    └── validation_report_*.txt
```

### Backup Considerations
- Maintain weekly backups during season
- Archive historical data by season
- Version control configuration files

## Update Frequency
- During season: Weekly updates after games
- Off-season: Monthly updates for historical corrections
- Configuration: As needed for team changes or metric adjustments 