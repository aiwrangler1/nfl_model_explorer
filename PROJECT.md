# NFL EDP Analysis Project

## Overview
A personal analysis tool for calculating Expected Drive Points (EDP) in NFL games. The project focuses on three core objectives:
1. Gathering and preparing NFL play-by-play data
2. Calculating EDP metrics with sophisticated adjustments
3. Creating clear, actionable visualizations

## IMPORTANT NOTE
KEEP IT SIMPLE! All files should live in the root directory unless absolutely necessary.
This helps avoid complexity and makes the project easier to maintain.

## Project Structure
```
nfl-2024-edp-model/
├── data_loader.py      # Load and prepare NFL data
├── edp_calculator.py   # Core EDP calculations and adjustments
├── edp_analysis.py     # Main analysis script
├── data/              # Data storage
├── outputs/           # Results and visualizations
└── PROJECT.md         # Project documentation (this file)
```

## Recent Status Update (January 2025)

### SoS Adjustment Refinements
- Tested two approaches for SoS calculations:
  1. Zero-mean normalization (caused scaling issues)
  2. Recentering to original means (working well)
- Kept recentering approach which:
  - Maintains natural football metric scales
  - Allows proper iterative adjustments
  - Prevents wild value swings
- Using 0.33 dampening factor for balanced adjustments

### Visualization Plan
1. Team Performance Chart
   - Title: Team EDP, Strength Adjusted
   - X-axis: SoS-adjusted offensive EDP
   - Y-axis: SoS-adjusted defensive EDP (inverted)
   - Simplify axis titles to Off. EDP and Def. EDP (the title indicates that it's SoS adj.) 
   - Implementation:
     ```python
     def create_team_performance_chart():
         # Simple scatter plot
         # Teams as points
         # Quadrant analysis (good O/good D, etc.)
         # Optional: team logos instead of points
     ```
2. Weekly Trend Lines

### Next Steps
1. Implement basic charts first (keep it simple!)
4. Keep all visualization code in root directory
5. Focus on readability and maintainability

## Core Components

### Data Loading (data_loader.py)
- Uses nfl_data_py to fetch play-by-play data
- Basic data cleaning and preparation
- Keeps only necessary columns
- Simple team name standardization

### EDP Calculator (edp_calculator.py)
- Calculates drive-level metrics using EPA
- Aggregates to team level
- Computes offensive and defensive EDP
- Implements iterative strength of schedule adjustments
  - Looks at actual matchups rather than season averages
  - Refines adjustments over multiple iterations
  - Maintains zero-sum normalization
  - Handles bye weeks appropriately

### Analysis Script (edp_analysis.py)
- Main script to run analysis
- Implements three types of metrics:
  1. Raw metrics (unadjusted)
  2. Weighted metrics (recency-based)
  3. SoS-adjusted metrics (iterative adjustment)
- Saves results to Excel with multiple views:
  - Season Rankings
  - Weekly Results
  - Historical Data

## Implementation Details

### Metric Calculations

1. **Raw Metrics**
   - Basic EDP calculations without adjustments
   - Both total and per-drive metrics
   - Separate offensive and defensive values
   - Smart rounding based on value magnitude:
     - Numbers ≥ 100: 1 decimal point
     - Numbers < 100: 3 decimal points

2. **Weighted Metrics**
   - Implements recency-based weighting
   - Last 4 games: Full weight (1.0)
   - Exponential decay after 4 games (factor: 0.85)
   - Accounts for actual games played (not just weeks)
   - Proper normalization:
     - Counts only non-zero weighted games
     - Preserves original metric scale
     - Handles bye weeks correctly

3. **Strength of Schedule Adjustments**
   - Improved calculation flow:
     1. Calculate SoS adjustments on raw data first
     2. Merge in recency weights after
     3. Apply weights to adjusted metrics
   - Maintains proper scaling throughout
   - More accurate opponent adjustments
   - Better handling of weighted metrics

### Recent Improvements

1. **Calculation Efficiency**:
   - Eliminated redundant calculations in SoS adjustments
   - Now using total adjusted values directly
   - Reduced potential rounding errors

2. **Weight Normalization**:
   - Fixed game counting to only include non-zero weights
   - Better handling of bye weeks and zero-weighted games
   - More accurate weighting of recent performance

3. **SoS Calculation**:
   - Separated SoS calculation from weighting
   - Applied adjustments to raw data first
   - More accurate strength of schedule assessment
   - Better interaction between weighting and SoS

4. **Rounding Logic**:
   - Implemented smart rounding based on value magnitude
   - More readable output for large numbers
   - Maintained precision where needed

## Notes
- Defensive EDP: Negative values = better defense
- total_edp = offensive_edp - defensive_edp
- All metrics available in both total and per-drive format
- SoS adjustments use actual matchups and iterative refinement
- Recency weighting uses actual games played, not calendar weeks

## Future Improvements
1. Visualization Enhancements
   - Add trend charts for team performance
   - Create matchup comparison views
   - Visualize strength of schedule impact
   - Add simple visual chart for off-edp-sos-adj (x-axis) vs def-edp-sos-adj (y-axis, inverted to account for negative edp being good.)

### Long Term
1. Consider distributed processing
2. Implement database backend
3. Add real-time processing capabilities
4. Optimize visualization generation 

2. Additional Metrics
   - Add variance/consistency metrics
   - Calculate projected matchup advantages
   - Track week-over-week changes

3. Analysis Features
   - Add predictive modeling capabilities
   - Implement "what-if" scenario analysis
   - Create detailed matchup reports

4. Technical Improvements
   - Add unit tests
   - Improve error handling
   - Add logging for better debugging
   - Optimize performance for larger datasets
   - Add memory usage monitoring
   - Add performance logging


## Performance Optimization Strategy Ideas:

### 1. Data Loading and Storage
- **Use Parquet Format**
  - Columnar storage for efficient querying
  - Built-in compression
  - Fast read/write operations
  - Schema enforcement

- **Caching Strategy**
  ```python
  # Example cache configuration
  CACHE_CONFIG = {
      'format': 'parquet',
      'compression': 'snappy',
      'partition_cols': ['season', 'week'],
      'cache_duration_days': 7
  }
  ```

### 2. Memory Management

#### Data Chunking
For historical analysis, process data in chunks:
```python
def process_historical_data(seasons: List[int], chunk_size: int = 1):
    for season_chunk in np.array_split(seasons, chunk_size):
        df = load_and_process_chunk(season_chunk)
        yield df
```

#### Memory Profiling
Monitor memory usage during processing:
```python
from memory_profiler import profile

@profile
def memory_intensive_operation(df: pd.DataFrame):
    # Your code here
    pass
```

### 3. Processing Optimization

#### Vectorized Operations
Prefer vectorized operations over loops:
```python
# Good
df['success'] = df['yards_gained'] >= df['ydstogo']

# Avoid
df['success'] = df.apply(lambda x: x['yards_gained'] >= x['ydstogo'], axis=1)
```

#### Parallel Processing
For drive-level calculations:
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_drive_processing(df: pd.DataFrame, n_workers: int = 4):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        drive_groups = [group for _, group in df.groupby('drive')]
        results = executor.map(process_drive, drive_groups)
```

### 4. Database Integration
Consider using a database for large-scale analysis:
```python
DATABASE_CONFIG = {
    'type': 'postgresql',
    'batch_size': 10000,
    'indexes': ['game_id', 'play_id', ('season', 'week')]
}
```

## Performance Benchmarks

### 1. Data Loading
| Operation | Expected Time | Memory Usage |
|-----------|---------------|--------------|
| Raw Load  | 1-2s/season   | ~100MB      |
| Cache Read| 0.5s/season   | ~50MB       |
| Full Load | 5-10s         | ~500MB      |

### 2. Processing
| Operation | Expected Time | Memory Usage |
|-----------|---------------|--------------|
| Clean     | 1-2s/season   | ~150MB      |
| Calculate | 2-3s/season   | ~200MB      |
| Export    | 1-2s/season   | ~100MB      |

### 3. Visualization
| Operation | Expected Time | Memory Usage |
|-----------|---------------|--------------|
| Plot Gen  | 1-2s         | ~50MB       |
| Export    | 1-2s         | ~100MB      |

## Optimization Checklist

### Data Loading
- [ ] Use appropriate chunk sizes
- [ ] Implement caching strategy
- [ ] Monitor memory usage
- [ ] Log performance metrics

### Processing
- [ ] Use vectorized operations
- [ ] Implement parallel processing where appropriate
- [ ] Pre-filter unnecessary data
- [ ] Optimize merge operations

### Output Generation
- [ ] Buffer large writes
- [ ] Use appropriate compression
- [ ] Clean up temporary files
- [ ] Monitor disk usage

## Monitoring and Logging

### Performance Metrics to Track
```python
PERFORMANCE_METRICS = {
    'data_loading_time': {'warning': 5, 'critical': 10},
    'processing_time': {'warning': 10, 'critical': 20},
    'memory_usage': {'warning': '1GB', 'critical': '2GB'},
    'disk_usage': {'warning': '80%', 'critical': '90%'}
}
```

### Logging Configuration
```python
LOGGING_CONFIG = {
    'performance_log': 'logs/performance.log',
    'metrics_interval': 60,  # seconds
    'retain_logs_days': 30
}
```
