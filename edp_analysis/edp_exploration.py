"""
Exploratory analysis and modeling of Expected Defensive Points (EDP) metrics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Union
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from config import WP_DECAY_FACTOR, WP_CALIBRATION_BINS
from edp_calculator import calculate_team_edp

def prepare_wp_features(
    df: pd.DataFrame,
    decay_factor: float = WP_DECAY_FACTOR
) -> Optional[pd.DataFrame]:
    """
    Prepare features for win probability model.
    
    Args:
        df (pd.DataFrame): Input dataframe with play-by-play data
        decay_factor (float): Decay factor for spread adjustment
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with prepared features, or None if error occurs
        
    Note:
        Will automatically add home_field_advantage and rest_advantage columns with
        default values if they don't exist in the input data.
    """
    logging.info("Starting feature preparation")
    
    try:
        # Calculate base EDP metrics
        edp_metrics = calculate_team_edp(df)
        
        # Add missing columns with default values if they don't exist
        if 'home_field_advantage' not in edp_metrics.columns:
            logging.warning("home_field_advantage column not found, adding with default value")
            edp_metrics['home_field_advantage'] = edp_metrics['team'].map(
                lambda x: 1 if x == edp_metrics['home_team'].iloc[0] else -1
            )
            
        if 'rest_advantage' not in edp_metrics.columns:
            logging.warning("rest_advantage column not found, adding with default value of 0")
            edp_metrics['rest_advantage'] = 0
        
        # Calculate decayed spread
        edp_metrics['decayed_spread'] = edp_metrics.apply(
            lambda row: row['spread_line'] * np.exp(-decay_factor * (3600 - row['time_remaining'])), 
            axis=1
        )
        
        logging.info(f"Features prepared successfully. Shape: {edp_metrics.shape}")
        return edp_metrics
        
    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")
        return None

def train_wp_model(
    features_df: pd.DataFrame,
    decay_factor: float = WP_DECAY_FACTOR,
    n_bins: Optional[int] = None,
    output_dir: str = 'model_outputs',
    calibration_metrics: Optional[Dict] = None,
    binned_results: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Train a win probability model and evaluate its performance.
    
    Args:
        features_df (pd.DataFrame): Dataframe with prepared features
        decay_factor (float): Decay factor for spread adjustment
        n_bins (Optional[int]): Number of bins for calibration plots (calculated dynamically if None)
        output_dir (str): Directory to save model outputs
        calibration_metrics (Optional[Dict]): Dictionary to store calibration metrics
        binned_results (Optional[pd.DataFrame]): Dataframe to store binned prediction results
        
    Returns:
        Dict: Dictionary with model results and metadata including:
            - model: Trained XGBoost model
            - train_loss/test_loss: Log loss metrics
            - train_brier/test_brier: Brier score metrics
            - importance_dict: Feature importance scores
            - predictions: Game-level predictions
            - calibration_metrics: Model calibration metrics
            - binned_results: Binned prediction results
            - timestamp: Training timestamp
    """
    logging.info("Starting win probability model training")
    
    # Prepare features
    features_df = prepare_wp_features(features_df, decay_factor)
    if features_df is None:
        raise ValueError("Feature preparation failed")
    
    # Filter to only games with EDP data
    features_df = features_df[features_df['offensive_edp'].notnull() & features_df['defensive_edp'].notnull()]
    
    # Create target variable
    features_df['target'] = (features_df['score_differential'] > 0).astype(int)
    
    # Create feature matrix
    feature_cols = [
        'offensive_edp', 'defensive_edp', 'decayed_spread', 
        'home_field_advantage', 'rest_advantage'
    ]
    X = features_df[feature_cols]
    y = features_df['target']
    
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Evaluate performance
    train_loss = log_loss(y_train, train_preds)
    test_loss = log_loss(y_test, test_preds)
    train_brier = brier_score_loss(y_train, train_preds)
    test_brier = brier_score_loss(y_test, test_preds)
    
    logging.info(f"Train log loss: {train_loss:.4f}, Test log loss: {test_loss:.4f}")
    logging.info(f"Train Brier score: {train_brier:.4f}, Test Brier score: {test_brier:.4f}")
    
    # Calculate feature importances
    importance_dict = dict(zip(feature_cols, model.feature_importances_))
    logging.info(f"Feature importances: {importance_dict}")
    
    # Save feature importance plot
    plot_feature_importance(importance_dict, f"{output_dir}/feature_importance.png")
    
    # Predict on full dataset for analysis
    features_df['predicted'] = model.predict_proba(X)[:, 1]
    
    # Evaluate calibration
    if n_bins is None:
        n_bins = int(np.ceil(len(features_df) / 1000))  # About 1000 plays per bin
        
    calibration_metrics, binned_results = evaluate_calibration(
        features_df, 
        n_bins=n_bins,
        output_dir=output_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'model': model,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_brier': train_brier,
        'test_brier': test_brier,
        'importance_dict': importance_dict,
        'predictions': features_df[['game_id', 'target', 'predicted']],
        'calibration_metrics': calibration_metrics,
        'binned_results': binned_results,
        'timestamp': timestamp
    }
    
    return results

def evaluate_calibration(
    df: pd.DataFrame,
    n_bins: int = 10,
    output_dir: str = 'model_outputs'
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate model calibration and save results.
    
    Args:
        df (pd.DataFrame): Dataframe with actual and predicted values
        n_bins (int): Number of bins for calibration plot
        output_dir (str): Directory to save calibration results
        
    Returns:
        Tuple[Dict[str, float], pd.DataFrame]: 
            - metrics_dict contains calibration metrics
            - binned_df contains binned actual and predicted values
    """
    # Calculate binned actuals and predictions
    df['bin_pred_prob'] = pd.cut(
        df['predicted'], 
        bins=np.linspace(0, 1, n_bins+1), 
        labels=np.linspace(0, 1, n_bins)
    )
    
    binned_df = df.groupby('bin_pred_prob').agg(
        bin_actual_prob=('target', 'mean'),
        n_plays=('target', 'count')
    ).reset_index()
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(binned_df['bin_actual_prob'] - binned_df['bin_pred_prob']))
    
    # Calculate Brier score
    brier_score = brier_score_loss(df['target'], df['predicted'])
    
    # Plot calibration curve
    plot_calibration_curve(df['target'], df['predicted'], n_bins=n_bins)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curve.png")
    plt.close()
    
    # Save binned results
    binned_df.to_csv(f"{output_dir}/binned_results.csv", index=False)
    
    metrics_dict = {
        'calibration_error': calibration_error,
        'brier_score': brier_score
    }
    
    return metrics_dict, binned_df

def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = WP_CALIBRATION_BINS,
    title: Optional[str] = None
) -> None:
    """
    Plot calibration curve for a binary classifier.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        n_bins (int): Number of bins for the calibration curve
        title (Optional[str]): Plot title
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=n_bins)
    
    plt.figure(figsize=(10, 10))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title or "Calibration plot")
    plt.legend()
    plt.show()
    
def plot_calibration_by_quarter(
    df: pd.DataFrame,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Plot calibration curves by quarter.
    
    Args:
        df (pd.DataFrame): Dataframe with 'quarter', 'actual', and 'predicted' columns
        output_dir (str): Directory to save the plots
        timestamp (str): Timestamp to include in the plot filenames
    """
    for i, qtr in enumerate(['1', '2', '3', '4'], 1):
        qtr_df = df[df['quarter'] == i]
        plot_calibration_curve(
            qtr_df['actual'], 
            qtr_df['predicted'], 
            title=f"Quarter {qtr} Calibration"
        )
        plt.savefig(f"{output_dir}/calibration_q{qtr}_{timestamp}.png")
        plt.close()
        
def plot_feature_importance(
    importance_dict: Dict[str, float],
    output_path: str
) -> None:
    """
    Plot feature importance from a dictionary.
    
    Args:
        importance_dict (Dict[str, float]): Dictionary of feature names and their importances
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        importance_dict.items(), 
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
