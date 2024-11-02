import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess data for EDP calculation, focusing on meaningful offensive plays
    """
    # Store score_differential before preprocessing
    score_differential = df[['game_id', 'score_differential']].drop_duplicates()
    
    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=[
        'play_type', 'posteam', 'defteam', 'yards_gained', 'down', 'ydstogo', 
        'yardline_100', 'epa', 'game_id', 'drive', 'play_id'
    ])

    # Filter for meaningful offensive plays (run, pass, and field goals)
    df = df[df['play_type'].isin(['run', 'pass', 'field_goal'])].reset_index(drop=True)

    # Define success based on play type and situation
    def calculate_success(row):
        if row['play_type'] == 'field_goal':
            # Handle case where field_goal_result might not exist
            if 'field_goal_result' in row:
                return 1 if row['field_goal_result'] == 'made' else 0
            else:
                # Fallback to using score differential change
                return 1 if row['score_differential'] != 0 else 0
        elif row['down'] == 1:
            return int(row['yards_gained'] >= 0.4 * row['ydstogo'])
        elif row['down'] == 2:
            return int(row['yards_gained'] >= 0.6 * row['ydstogo'])
        elif row['down'] in [3, 4]:
            return int(row['yards_gained'] >= row['ydstogo'])
        else:
            return 0

    df['success'] = df.apply(calculate_success, axis=1)

    # Calculate drive-specific metrics
    df['drive_play_number'] = df.groupby(['game_id', 'drive']).cumcount() + 1
    df['is_scoring_opportunity'] = (df['yardline_100'] <= 20).astype(int)
    
    # Merge back the score_differential
    df = df.merge(score_differential, on='game_id', how='left', suffixes=('_drop', ''))
    if 'score_differential_drop' in df.columns:
        df = df.drop('score_differential_drop', axis=1)
    
    return df

def calculate_rolling_team_edp(df):
    """
    Calculate EDP with enhanced drive-level focus
    """
    # Calculate drive-level metrics
    df['explosive_play'] = (df['yards_gained'] >= 20).astype(int)
    df['late_down_conversion'] = ((df['down'] >= 3) & (df['success'] == 1)).astype(int)
    df['scoring_zone_success'] = (df['is_scoring_opportunity'] & df['success']).astype(int)

    drive_metrics = df.groupby(['game_id', 'posteam', 'drive']).agg(
        drive_epa=('epa', 'sum'),
        num_plays=('play_id', 'count'),
        num_successful_plays=('success', 'sum'),
        num_explosive_plays=('explosive_play', 'sum'),
        num_late_down_conversions=('late_down_conversion', 'sum'),
        scoring_zone_successes=('scoring_zone_success', 'sum'),
        max_drive_play_number=('drive_play_number', 'max')
    ).reset_index()

    # Calculate drive sustainability factor
    drive_metrics['drive_sustainability'] = (
        drive_metrics['num_successful_plays'] / drive_metrics['num_plays']
    ).fillna(0)

    # Enhanced EDP calculation incorporating drive sustainability
    drive_metrics['edp'] = (
        drive_metrics['drive_epa'] * 0.3 +  # Base EPA component
        drive_metrics['drive_sustainability'] * 10 +  # Drive sustainability
        drive_metrics['num_explosive_plays'] * 2 +  # Explosive plays bonus
        drive_metrics['num_late_down_conversions'] * 1.5 +  # Critical conversions
        drive_metrics['scoring_zone_successes'] * 2  # Red zone execution
    )

    offensive_edp = drive_metrics.groupby(['posteam', 'game_id']).agg(
        offensive_edp=('edp', 'sum')
    ).reset_index()

    offensive_edp['rolling_offensive_edp'] = offensive_edp.groupby('posteam')['offensive_edp'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    defensive_drive_metrics = df.copy()
    defensive_drive_metrics['posteam'], defensive_drive_metrics['defteam'] = defensive_drive_metrics['defteam'], defensive_drive_metrics['posteam']

    defensive_metrics = defensive_drive_metrics.groupby(['game_id', 'posteam', 'drive']).agg(
        drive_epa=('epa', 'sum')
    ).reset_index()

    defensive_metrics['edp'] = -defensive_metrics['drive_epa']

    defensive_edp = defensive_metrics.groupby(['posteam', 'game_id']).agg(
        defensive_edp=('edp', 'sum')
    ).reset_index()

    defensive_edp['rolling_defensive_edp'] = defensive_edp.groupby('posteam')['defensive_edp'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    offensive_edp = offensive_edp.rename(columns={'posteam': 'team'})
    defensive_edp = defensive_edp.rename(columns={'posteam': 'team'})

    # Merge offensive and defensive EDPs
    combined_edp = pd.merge(offensive_edp, defensive_edp, on=['team', 'game_id'])
    
    # Modify the final merge to check for the correct column name
    score_col = 'score_differential'
    if score_col not in df.columns and f'{score_col}_x' in df.columns:
        score_col = f'{score_col}_x'
    
    # Merge with original DataFrame to retain necessary columns
    result = pd.merge(
        df[['game_id', 'home_team', 'away_team', score_col]].drop_duplicates(), 
        combined_edp, 
        on='game_id'
    )
    
    # Ensure the column has the correct name
    if score_col != 'score_differential':
        result = result.rename(columns={score_col: 'score_differential'})
    
    return result

def calculate_weighted_rolling_edp(df, short_window=5, long_window=16, alpha=0.8):
    """
    Calculate a combination of short-term and long-term EDP metrics with recency weighting.
    
    Parameters:
    - short_window: Rolling window for short-term performance
    - long_window: Rolling window for season-long stability
    - alpha: Smoothing factor for exponentially weighted moving average (recency weight)
    
    Returns:
    DataFrame with short-term, long-term, and weighted EDP metrics.
    """
    # Ensure 'offensive_edp' and 'defensive_edp' columns exist
    if 'offensive_edp' not in df.columns or 'defensive_edp' not in df.columns:
        raise ValueError("'offensive_edp' or 'defensive_edp' columns not found. Make sure to run calculate_rolling_team_edp first.")

    # Calculate short-term EDP (rolling window)
    df['rolling_short_offensive_edp'] = df.groupby('team')['offensive_edp'].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df['rolling_long_offensive_edp'] = df.groupby('team')['offensive_edp'].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    
    # Exponentially weighted moving average for recency emphasis
    df['ewm_offensive_edp'] = df.groupby('team')['offensive_edp'].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )

    # Combine them to form a weighted metric
    df['weighted_offensive_edp'] = (
        0.5 * df['rolling_short_offensive_edp'] +
        0.3 * df['rolling_long_offensive_edp'] +
        0.2 * df['ewm_offensive_edp']
    )

    # Repeat for defensive EDP
    df['rolling_short_defensive_edp'] = df.groupby('team')['defensive_edp'].transform(
        lambda x: x.rolling(window=short_window, min_periods=1).mean()
    )
    df['rolling_long_defensive_edp'] = df.groupby('team')['defensive_edp'].transform(
        lambda x: x.rolling(window=long_window, min_periods=1).mean()
    )
    df['ewm_defensive_edp'] = df.groupby('team')['defensive_edp'].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean()
    )

    df['weighted_defensive_edp'] = (
        0.5 * df['rolling_short_defensive_edp'] +
        0.3 * df['rolling_long_defensive_edp'] +
        0.2 * df['ewm_defensive_edp']
    )

    return df[['game_id', 'team', 'weighted_offensive_edp', 'weighted_defensive_edp']]

def calculate_strength_adjusted_edp(df_base, df_offense, df_defense):
    """
    Calculate strength-adjusted EDP by considering opponent's defensive/offensive strength
    """
    # Merge opponent's defensive strength for offensive adjustment
    df_merged = df_base.merge(
        df_defense[['team', 'game_id', 'weighted_defensive_edp']],  # Changed column names
        left_on=['opponent_team', 'game_id'],  # Changed to use game_id instead of date
        right_on=['team', 'game_id'],
        suffixes=('', '_opp')
    )
    
    # Merge opponent's offensive strength for defensive adjustment
    df_merged = df_merged.merge(
        df_offense[['team', 'game_id', 'weighted_offensive_edp']],  # Changed column names
        left_on=['opponent_team', 'game_id'],
        right_on=['team', 'game_id'],
        suffixes=('', '_opp')
    )
    
    # Calculate adjusted values
    df_merged['adjusted_offensive_edp'] = df_merged['weighted_offensive_edp'] + df_merged['weighted_defensive_edp_opp']
    df_merged['adjusted_defensive_edp'] = df_merged['weighted_defensive_edp'] + df_merged['weighted_offensive_edp_opp']
    
    return df_merged

def apply_home_away_adjustment(df):
    """
    Add a binary column indicating home (1) or away (0) for each game.
    """
    df['home_field_advantage'] = np.where(df['team'] == df['home_team'], 1, 0)
    return df

# Add any other functions here as needed
