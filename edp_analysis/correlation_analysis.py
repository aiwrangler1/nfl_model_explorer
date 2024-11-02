import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_loader import load_and_prepare_data
from edp_calculator import preprocess_data, calculate_rolling_team_edp, calculate_weighted_rolling_edp, apply_home_away_adjustment

def calculate_strength_adjusted_edp(df_base, df_offense, df_defense):
    """
    Calculate strength-adjusted EDP by considering opponent's defensive/offensive strength
    """
    # Merge opponent's defensive strength for offensive adjustment
    df_merged = df_base.merge(
        df_defense[['team', 'game_id', 'weighted_defensive_edp']],
        left_on=['opponent_team', 'game_id'],
        right_on=['team', 'game_id'],
        suffixes=('', '_opp')
    )
    
    # Merge opponent's offensive strength for defensive adjustment
    df_merged = df_merged.merge(
        df_offense[['team', 'game_id', 'weighted_offensive_edp']],
        left_on=['opponent_team', 'game_id'],
        right_on=['team', 'game_id'],
        suffixes=('', '_opp')
    )
    
    # Calculate adjusted values
    df_merged['adjusted_offensive_edp'] = df_merged['weighted_offensive_edp'] + df_merged['weighted_defensive_edp_opp']
    df_merged['adjusted_defensive_edp'] = df_merged['weighted_defensive_edp'] + df_merged['weighted_offensive_edp_opp']
    
    return df_merged

def run_grid_search(df, short_windows=[3, 5, 7], long_windows=[10, 16, 32], alphas=[0.6, 0.8, 0.9]):
    print("Initial columns:", df.columns.tolist())
    
    # Initialize best_mse and best_params
    best_mse = float('inf')
    best_params = None
    
    # Preprocess the data first
    df = preprocess_data(df)
    print("After preprocess columns:", df.columns.tolist())
    
    # Calculate rolling team EDP
    df_rolling = calculate_rolling_team_edp(df)
    print("After rolling EDP columns:", df_rolling.columns.tolist())
    
    # Add opponent information
    df_rolling['opponent_team'] = df_rolling.apply(
        lambda row: row['away_team'] if row['team'] == row['home_team'] else row['home_team'], 
        axis=1
    )
    
    total_iterations = len(short_windows) * len(long_windows) * len(alphas)
    current_iteration = 0
    
    print(f"\nStarting grid search with {total_iterations} combinations...")
    
    for short_window in short_windows:
        for long_window in long_windows:
            for alpha in alphas:
                current_iteration += 1
                print(f"\nProcessing combination {current_iteration}/{total_iterations}")
                print(f"Parameters: short_window={short_window}, long_window={long_window}, alpha={alpha}")
                
                try:
                    # Calculate weighted rolling EDP
                    df_weighted = calculate_weighted_rolling_edp(df_rolling, short_window, long_window, alpha)
                    print("Weighted EDP calculated")
                    
                    # Add opponent strength and home/away adjustment
                    df_adjusted = calculate_strength_adjusted_edp(df_rolling, df_weighted, df_weighted)
                    print("Strength adjustment applied")
                    
                    df_final = apply_home_away_adjustment(df_adjusted)
                    print("Home/away adjustment applied")
                    
                    # Define features and target
                    X = df_final[['adjusted_offensive_edp', 'adjusted_defensive_edp', 'home_field_advantage']]
                    y = df_final['score_differential']
                    
                    # Train and evaluate the model
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    
                    print(f"MSE: {mse:.4f}")
                    
                    # Update best params if current configuration is better
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'short_window': short_window, 'long_window': long_window, 'alpha': alpha}
                        print(f"New best MSE found: {best_mse:.4f}")
                
                except Exception as e:
                    print(f"Error processing combination: {e}")
                    continue
    
    if best_params is None:
        print("\nNo valid results found during grid search.")
    else:
        print(f"\nBest Mean Squared Error: {best_mse:.4f}")
        print(f"Best Parameters: Short Window={best_params['short_window']}, Long Window={best_params['long_window']}, Alpha={best_params['alpha']}")
    
    return best_params

def run_baseline_comparison(df, best_params):
    # Preprocess data
    df = preprocess_data(df)
    
    # Calculate rolling offensive and defensive EDP for each team using best parameters
    df_rolling = calculate_weighted_rolling_edp(df, best_params['short_window'], best_params['long_window'], best_params['alpha'])
    
    # Add opponent strength and home/away adjustment
    df_adjusted = calculate_strength_adjusted_edp(df, df_rolling, df_rolling)
    df_final = apply_home_away_adjustment(df_adjusted)

    # Calculate the mean score differential as the baseline
    mean_score_differential = df_final['score_differential'].mean()
    baseline_predictions = np.full(len(df_final), mean_score_differential)
    
    # Calculate MSE for the baseline
    baseline_mse = mean_squared_error(df_final['score_differential'], baseline_predictions)
    print(f"Baseline Mean Squared Error (using mean score differential): {baseline_mse:.4f}")
    
    # Run model prediction and MSE calculation for comparison
    X = df_final[['adjusted_offensive_edp', 'adjusted_defensive_edp', 'home_field_advantage']]
    y = df_final['score_differential']
    
    # Train the model on the entire dataset
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    model_mse = mean_squared_error(y, predictions)
    
    print(f"Model Mean Squared Error: {model_mse:.4f}")
    print("\nComparison:")
    if model_mse < baseline_mse:
        print("The model outperforms the baseline, indicating predictive value.")
    else:
        print("The model does not outperform the baseline; further feature engineering or model tuning may be needed.")

if __name__ == "__main__":
    df = load_and_prepare_data()
    best_params = run_grid_search(df)
    run_baseline_comparison(df, best_params)