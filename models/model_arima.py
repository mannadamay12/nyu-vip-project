"""
ARIMA Model Module

This module implements an ARIMA (AutoRegressive Integrated Moving Average) model
following the standardized model interface.

ARIMA(p, d, q):
    - p: Order of the autoregressive (AR) component
    - d: Degree of differencing (I)
    - q: Order of the moving average (MA) component

ARIMA is a classical statistical model for univariate time series forecasting.
Unlike neural network models, it works with a single time series (price/returns)
rather than multiple features.

The model uses statsmodels library for implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import warnings
from itertools import product

# Suppress convergence warnings from statsmodels
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA model will not work.")

from config import TASK_TYPE, ARIMA_CONFIG, RANDOM_SEED


def check_stationarity(series: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.

    Args:
        series: 1D array of time series values
        significance: Significance level for the test

    Returns:
        Tuple of (is_stationary, p_value)
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < significance
    return is_stationary, p_value


def auto_select_arima_order(
    train_series: np.ndarray,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
) -> Tuple[int, int, int]:
    """
    Automatically select best ARIMA order (p, d, q) using AIC criterion.

    Args:
        train_series: Training time series
        max_p: Maximum AR order to try
        max_d: Maximum differencing order to try
        max_q: Maximum MA order to try

    Returns:
        Tuple of (best_p, best_d, best_q)
    """
    best_aic = np.inf
    best_order = (1, 0, 0)

    # Determine d based on stationarity
    is_stationary, _ = check_stationarity(train_series)
    d_values = [0] if is_stationary else [0, 1]
    d_values = [d for d in d_values if d <= max_d]

    # Grid search over (p, d, q)
    p_values = range(1, max_p + 1)
    q_values = range(0, max_q + 1)

    print(f"  Auto-selecting ARIMA order (max_p={max_p}, max_d={max_d}, max_q={max_q})...")

    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(train_series, order=(p, d, q))
            fitted = model.fit()

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue

    print(f"  Best order: ARIMA{best_order} (AIC: {best_aic:.2f})")
    return best_order


def fit_arima(
    train_series: np.ndarray,
    order: Tuple[int, int, int] = None,
    config: dict = None
) -> Tuple[Any, Tuple[int, int, int]]:
    """
    Fit ARIMA model on training data.

    Args:
        train_series: 1D array of training values
        order: (p, d, q) tuple. If None, auto-select.
        config: Configuration dict

    Returns:
        Tuple of (fitted_model, order_used)
    """
    if config is None:
        config = ARIMA_CONFIG

    # Auto-select order if not specified
    if order is None and config.get("auto_select", True):
        order = auto_select_arima_order(
            train_series,
            max_p=config.get("max_p", 5),
            max_d=config.get("max_d", 2),
            max_q=config.get("max_q", 5),
        )
    elif order is None:
        order = (2, 1, 2)  # Default order

    # Fit the model
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    return fitted, order


def rolling_forecast(
    train_series: np.ndarray,
    actual_values: np.ndarray,
    order: Tuple[int, int, int],
    max_history: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate rolling one-step-ahead forecasts.

    For each test point, fit the model on available history,
    predict one step ahead, then add the actual observed value to history.

    Args:
        train_series: Initial training series
        actual_values: Actual values for the forecast period (used to update history)
        order: ARIMA order (p, d, q)
        max_history: Maximum history length to use (sliding window).
                     If None, use all available history (slower but potentially better).
        verbose: Whether to print progress

    Returns:
        Array of one-step-ahead predictions
    """
    predictions = []
    history = list(train_series)
    n_predictions = len(actual_values)

    for i in range(n_predictions):
        # Use sliding window if specified
        if max_history is not None and len(history) > max_history:
            fit_history = history[-max_history:]
        else:
            fit_history = history

        # Fit model on current history and predict
        try:
            model = ARIMA(fit_history, order=order)
            model_fit = model.fit()
            pred = model_fit.forecast(steps=1)[0]
            predictions.append(pred)
        except Exception:
            # If fitting fails, use last prediction or 0
            predictions.append(predictions[-1] if predictions else 0)

        # Add actual observed value to history for next iteration
        history.append(actual_values[i])

        if verbose and (i + 1) % 200 == 0:
            print(f"    Progress: {i + 1}/{n_predictions} predictions")

    return np.array(predictions)


def simple_forecast(
    fitted_model,
    steps: int,
) -> np.ndarray:
    """
    Generate simple multi-step ahead forecast (faster but less accurate).

    Args:
        fitted_model: Fitted ARIMA model
        steps: Number of steps to forecast

    Returns:
        Array of predictions
    """
    return fitted_model.forecast(steps=steps)


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict = None
) -> Dict[str, Any]:
    """
    Standard model interface for ARIMA using rolling one-step-ahead forecasts.

    ARIMA is a univariate model, so it only uses the price/return series,
    not the full feature set. The model fits on training returns and
    predicts future returns using rolling forecasts.

    For classification: predictions are the sign of predicted return (>0 = Up)
    For regression: predictions are the raw return values

    Args:
        datasets: dict from make_dataset_for_task() containing:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting

        config: Optional ARIMA configuration dict

    Returns:
        dict containing:
            - y_pred_val: Predictions on validation set
            - y_pred_test: Predictions on test set
            - model: Fitted ARIMA model info
            - order: (p, d, q) order used
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")

    if config is None:
        config = ARIMA_CONFIG

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Extract data
    y_train = datasets["y_train"]
    y_val = datasets["y_val"]
    y_test = datasets["y_test"]

    # Check if we have sequential data (3D) or tabular (2D)
    X_train = datasets["X_train"]
    if len(X_train.shape) == 3:
        # Sequential data - extract price feature (first column of last timestep)
        train_series = X_train[:, -1, 0]
        val_series = datasets["X_val"][:, -1, 0]
        test_series = datasets["X_test"][:, -1, 0]
    else:
        # Tabular data - first column is typically price
        train_series = X_train[:, 0]
        val_series = datasets["X_val"][:, 0]
        test_series = datasets["X_test"][:, 0]

    # Calculate returns from price series for ARIMA
    train_returns = np.diff(train_series) / train_series[:-1]
    train_returns = np.nan_to_num(train_returns, nan=0, posinf=0, neginf=0)

    val_returns = np.diff(np.concatenate([[train_series[-1]], val_series])) / \
                  np.concatenate([[train_series[-1]], val_series[:-1]])
    val_returns = np.nan_to_num(val_returns, nan=0, posinf=0, neginf=0)

    test_returns = np.diff(np.concatenate([[val_series[-1]], test_series])) / \
                   np.concatenate([[val_series[-1]], test_series[:-1]])
    test_returns = np.nan_to_num(test_returns, nan=0, posinf=0, neginf=0)

    # Get max_history from config for sliding window optimization
    max_history = config.get("max_history", 1000)

    print("\nARIMA Model (Rolling Forecast):")
    print("-" * 80)
    print(f"  Task type: {TASK_TYPE}")
    print(f"  Training samples: {len(train_returns)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Max history window: {max_history}")

    # Check stationarity
    is_stationary, p_value = check_stationarity(train_returns)
    print(f"  Series stationary: {is_stationary} (p-value: {p_value:.4f})")

    # Fit initial ARIMA on training returns to select order
    print("\nFitting ARIMA model to select order...")
    fitted_model, order = fit_arima(train_returns, order=None, config=config)

    print(f"\nModel Summary:")
    print(f"  Order: ARIMA{order}")
    print(f"  AIC: {fitted_model.aic:.2f}")
    print(f"  BIC: {fitted_model.bic:.2f}")

    # Generate rolling predictions
    print("\nGenerating rolling one-step-ahead predictions...")

    # Validation set predictions
    print("  Validation set:")
    y_pred_val_raw = rolling_forecast(
        train_series=train_returns,
        actual_values=val_returns,
        order=order,
        max_history=max_history,
        verbose=True,
    )

    # Test set predictions - continue from where validation ended
    print("  Test set:")
    full_train_returns = np.concatenate([train_returns, val_returns])
    y_pred_test_raw = rolling_forecast(
        train_series=full_train_returns,
        actual_values=test_returns,
        order=order,
        max_history=max_history,
        verbose=True,
    )

    # Convert predictions based on task type
    if TASK_TYPE == "classification":
        # For classification: use sign of predicted return
        # Positive predicted return -> 1.0 (Up), Negative -> 0.0 (Down)
        y_pred_val = (y_pred_val_raw > 0).astype(float)
        y_pred_test = (y_pred_test_raw > 0).astype(float)

        # Also store raw predictions for probability-based metrics
        val_up_pct = y_pred_val.mean() * 100
        test_up_pct = y_pred_test.mean() * 100
        print(f"\nPrediction distribution:")
        print(f"  Val: {val_up_pct:.1f}% Up, {100-val_up_pct:.1f}% Down")
        print(f"  Test: {test_up_pct:.1f}% Up, {100-test_up_pct:.1f}% Down")
    else:
        # Regression - use raw return predictions
        y_pred_val = y_pred_val_raw
        y_pred_test = y_pred_test_raw

    print(f"\nRaw prediction stats:")
    print(f"  Val: min={y_pred_val_raw.min():.6f}, max={y_pred_val_raw.max():.6f}, mean={y_pred_val_raw.mean():.6f}")
    print(f"  Test: min={y_pred_test_raw.min():.6f}, max={y_pred_test_raw.max():.6f}, mean={y_pred_test_raw.mean():.6f}")
    print("-" * 80)

    return {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "y_pred_val_raw": y_pred_val_raw,
        "y_pred_test_raw": y_pred_test_raw,
        "model": {
            "order": order,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
        },
        "order": order,
    }


if __name__ == "__main__":
    """
    Test ARIMA model with data pipeline
    """
    print("=" * 80)
    print("Testing ARIMA Model Module")
    print("=" * 80)

    if not STATSMODELS_AVAILABLE:
        print("statsmodels not installed. Skipping test.")
        exit(1)

    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    # Get tabular data for ARIMA (no sequences needed)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,  # ARIMA doesn't need sequences
        scaler_type="standard"
    )

    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")

    # Train and predict
    print("\n" + "=" * 80)
    print("Training ARIMA Model")
    print("=" * 80)

    # Use smaller search space for quick testing
    test_config = {
        "auto_select": True,
        "max_p": 3,
        "max_d": 1,
        "max_q": 2,
    }

    results = train_and_predict(datasets, config=test_config)

    # Evaluate results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    y_pred_test = results["y_pred_test"]
    y_test = datasets["y_test"]

    if TASK_TYPE == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        y_pred_binary = (y_pred_test > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)

        print(f"\nClassification Metrics (Test Set):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        print(f"\nRegression Metrics (Test Set):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R2:  {r2:.4f}")

    print(f"\nARIMA Order: {results['order']}")

    print("\n" + "=" * 80)
    print("ARIMA Model Test Complete!")
    print("=" * 80)
