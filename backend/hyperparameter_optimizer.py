#!/usr/bin/env python3
"""
Advanced hyperparameter optimization for time series forecasting
with time-series-aware cross-validation and comprehensive parameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
from scipy.optimize import minimize
import time

warnings.filterwarnings('ignore')

class TimeSeriesOptimizer:
    """Advanced hyperparameter optimizer for time series forecasting"""
    
    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        """
        Initialize the optimizer
        
        Args:
            n_splits: Number of splits for TimeSeriesSplit
            test_size: Size of test set for each split
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
    def time_series_cross_validate(self, 
                                 model_func: Callable,
                                 X: np.ndarray, 
                                 y: np.ndarray, 
                                 params: Dict[str, Any],
                                 scoring: str = 'rmse') -> float:
        """
        Perform time series cross-validation
        
        Args:
            model_func: Function that creates and fits the model
            X: Feature matrix
            y: Target values
            params: Model parameters
            scoring: Scoring metric ('rmse', 'mae', 'mape')
            
        Returns:
            Average cross-validation score
        """
        scores = []
        
        for train_idx, test_idx in self.tscv.split(X):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Fit model with parameters
                model = model_func(**params)
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate score
                if scoring == 'rmse':
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                elif scoring == 'mae':
                    score = mean_absolute_error(y_test, y_pred)
                elif scoring == 'mape':
                    score = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
                else:
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                
                scores.append(score)
                
            except Exception as e:
                # If model fails, assign a high penalty score
                scores.append(1e6)
                
        return np.mean(scores)
    
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize Random Forest hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            score = self.time_series_cross_validate(
                lambda **p: RandomForestRegressor(**p), X, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize XGBoost (GradientBoosting) hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            
            score = self.time_series_cross_validate(
                lambda **p: GradientBoostingRegressor(**p), X, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_svr(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize SVR hyperparameters using Optuna"""
        
        # Scale features for SVR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'rbf' else 'scale',
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'poly' else 3
            }
            
            # Clean up params based on kernel
            if params['kernel'] != 'poly':
                params.pop('degree', None)
            if params['kernel'] not in ['rbf', 'poly']:
                params.pop('gamma', None)
            
            score = self.time_series_cross_validate(
                lambda **p: SVR(**p), X_scaled, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=40, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_neural_network(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize Neural Network hyperparameters using Optuna"""
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            # Suggest network architecture
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []
            
            for i in range(n_layers):
                layer_size = trial.suggest_int(f'layer_{i}_size', 10, 200)
                hidden_layer_sizes.append(layer_size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 1000),
                'early_stopping': True,
                'validation_fraction': 0.1,
                'random_state': 42
            }
            
            score = self.time_series_cross_validate(
                lambda **p: MLPRegressor(**p), X_scaled, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_gaussian_process(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize Gaussian Process hyperparameters using Optuna"""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            # Suggest kernel type and parameters
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'rbf_white'])
            
            if kernel_type == 'rbf':
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
                constant_value = trial.suggest_float('constant_value', 0.1, 10.0)
                kernel = C(constant_value) * RBF(length_scale)
            elif kernel_type == 'matern':
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
                constant_value = trial.suggest_float('constant_value', 0.1, 10.0)
                nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
                kernel = C(constant_value) * Matern(length_scale=length_scale, nu=nu)
            else:  # rbf_white
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
                constant_value = trial.suggest_float('constant_value', 0.1, 10.0)
                noise_level = trial.suggest_float('noise_level', 1e-5, 1e-1, log=True)
                kernel = C(constant_value) * RBF(length_scale) + WhiteKernel(noise_level)
            
            params = {
                'kernel': kernel,
                'alpha': trial.suggest_float('alpha', 1e-10, 1e-1, log=True),
                'normalize_y': trial.suggest_categorical('normalize_y', [True, False]),
                'random_state': 42
            }
            
            score = self.time_series_cross_validate(
                lambda **p: GaussianProcessRegressor(**p), X_scaled, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def optimize_exponential_smoothing(self, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize Exponential Smoothing hyperparameters"""
        
        best_params = None
        best_score = float('inf')
        
        # Comprehensive parameter grid for Exponential Smoothing
        param_combinations = [
            {'trend': None, 'seasonal': None, 'seasonal_periods': None},
            {'trend': 'add', 'seasonal': None, 'seasonal_periods': None},
            {'trend': 'mul', 'seasonal': None, 'seasonal_periods': None},
            {'trend': None, 'seasonal': 'add', 'seasonal_periods': 4},
            {'trend': None, 'seasonal': 'add', 'seasonal_periods': 12},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 4},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12},
            {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 4},
            {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 12},
        ]
        
        # Additional parameters to optimize
        smoothing_params = [
            {'smoothing_level': 0.1, 'smoothing_trend': 0.1, 'smoothing_seasonal': 0.1},
            {'smoothing_level': 0.3, 'smoothing_trend': 0.1, 'smoothing_seasonal': 0.1},
            {'smoothing_level': 0.5, 'smoothing_trend': 0.2, 'smoothing_seasonal': 0.2},
            {'smoothing_level': 0.7, 'smoothing_trend': 0.3, 'smoothing_seasonal': 0.3},
            {'smoothing_level': None, 'smoothing_trend': None, 'smoothing_seasonal': None}  # Auto-optimize
        ]
        
        for base_params in param_combinations:
            for smooth_params in smoothing_params:
                try:
                    # Skip invalid combinations
                    if base_params['seasonal_periods'] and len(y) < 2 * base_params['seasonal_periods']:
                        continue
                    
                    params = {**base_params, **smooth_params}
                    
                    # Time series cross-validation for exponential smoothing
                    scores = []
                    for train_idx, test_idx in self.tscv.split(y):
                        try:
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            if len(y_train) < 3:
                                continue
                                
                            model = ExponentialSmoothing(
                                y_train,
                                trend=params['trend'],
                                seasonal=params['seasonal'],
                                seasonal_periods=params['seasonal_periods'],
                                initialization_method="estimated"
                            )
                            
                            fit = model.fit(
                                smoothing_level=params['smoothing_level'],
                                smoothing_trend=params['smoothing_trend'],
                                smoothing_seasonal=params['smoothing_seasonal'],
                                optimized=params['smoothing_level'] is None
                            )
                            
                            forecast = fit.forecast(len(y_test))
                            forecast = np.maximum(forecast, 0)
                            
                            score = np.sqrt(mean_squared_error(y_test, forecast))
                            scores.append(score)
                            
                        except:
                            scores.append(1e6)
                    
                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score < best_score:
                            best_score = avg_score
                            best_params = params
                            
                except:
                    continue
        
        return best_params or {}, best_score
    
    def optimize_holt_winters(self, y: np.ndarray, seasonal_periods: int = 12) -> Tuple[Dict[str, Any], float]:
        """Optimize Holt-Winters hyperparameters using grid search with time series CV"""
        
        if len(y) < 2 * seasonal_periods:
            seasonal_periods = max(4, len(y) // 3)
        
        best_params = None
        best_score = float('inf')
        
        # Comprehensive parameter grid
        param_grid = {
            'trend': [None, 'add', 'mul'],
            'seasonal': [None, 'add', 'mul'],
            'damped_trend': [True, False],
            'seasonal_periods': [seasonal_periods, seasonal_periods // 2, seasonal_periods * 2] if seasonal_periods > 4 else [seasonal_periods]
        }
        
        # Generate all combinations
        for params in ParameterGrid(param_grid):
            try:
                # Skip invalid combinations
                if params['seasonal_periods'] and len(y) < 2 * params['seasonal_periods']:
                    continue
                if params['seasonal'] is None and params['damped_trend']:
                    continue
                
                # Time series cross-validation
                scores = []
                for train_idx, test_idx in self.tscv.split(y):
                    try:
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        if len(y_train) < params['seasonal_periods'] * 2:
                            continue
                        
                        model = ExponentialSmoothing(
                            y_train,
                            trend=params['trend'],
                            seasonal=params['seasonal'],
                            seasonal_periods=params['seasonal_periods'] if params['seasonal'] else None,
                            damped_trend=params['damped_trend'],
                            initialization_method="estimated"
                        )
                        
                        fit = model.fit(optimized=True)
                        forecast = fit.forecast(len(y_test))
                        forecast = np.maximum(forecast, 0)
                        
                        score = np.sqrt(mean_squared_error(y_test, forecast))
                        scores.append(score)
                        
                    except:
                        scores.append(1e6)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = params
                        
            except:
                continue
        
        return best_params or {}, best_score
    
    def optimize_arima_with_external(self, y: np.ndarray, exog: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float]:
        """Optimize ARIMA/ARIMAX parameters using time series cross-validation"""
        
        best_params = None
        best_score = float('inf')
        
        # ARIMA parameter ranges
        p_range = range(0, 4)
        d_range = range(0, 3)
        q_range = range(0, 4)
        
        # For computational efficiency, limit combinations
        param_combinations = []
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    if p + d + q <= 6:  # Limit complexity
                        param_combinations.append((p, d, q))
        
        for order in param_combinations:
            try:
                scores = []
                for train_idx, test_idx in self.tscv.split(y):
                    try:
                        y_train, y_test = y[train_idx], y[test_idx]
                        exog_train = exog[train_idx] if exog is not None else None
                        exog_test = exog[test_idx] if exog is not None else None
                        
                        if len(y_train) < 10:
                            continue
                        
                        model = ARIMA(y_train, exog=exog_train, order=order)
                        fit = model.fit()
                        
                        forecast = fit.forecast(steps=len(y_test), exog=exog_test)
                        forecast = np.maximum(forecast, 0)
                        
                        score = np.sqrt(mean_squared_error(y_test, forecast))
                        scores.append(score)
                        
                    except:
                        scores.append(1e6)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = {'order': order}
                        
            except:
                continue
        
        return best_params or {'order': (1, 1, 1)}, best_score
    
    def optimize_knn(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Optimize KNN hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, min(50, len(X) // 2)),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                'p': trial.suggest_int('p', 1, 3)  # 1=manhattan, 2=euclidean, 3=minkowski
            }
            
            score = self.time_series_cross_validate(
                lambda **p: KNeighborsRegressor(**p), X, y, params
            )
            return score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        return study.best_params, study.best_value

class EnhancedForecastingEngine:
    """Enhanced forecasting engine with advanced hyperparameter optimization"""
    
    @staticmethod
    def prepare_features_for_ml(data: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning models with comprehensive feature engineering"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        n = len(y)
        
        # Get external factor columns
        external_factor_cols = [col for col in data.columns 
                             if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        
        # Adaptive window size
        window_size = min(window_size, n - 1, 10)
        
        if window_size < 1:
            # Fallback for very small datasets
            X = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                X = np.hstack([X, data[external_factor_cols].values])
            return X, y
        
        X, y_target = [], []
        
        for i in range(window_size, n):
            # Lag features
            lags = list(y[i-window_size:i])
            
            # Time-based features
            date = dates.iloc[i]
            time_features = [
                i,  # Linear trend
                date.month,
                date.quarter,
                date.dayofyear,
                date.weekday(),
                i % 12,  # Monthly cycle
                i % 4,   # Quarterly cycle
            ]
            
            # Statistical features from recent history
            recent_window = min(6, i)
            recent_data = y[max(0, i-recent_window):i]
            
            statistical_features = [
                np.mean(recent_data),
                np.std(recent_data) if len(recent_data) > 1 else 0,
                np.min(recent_data),
                np.max(recent_data),
                np.median(recent_data),
            ]
            
            # Trend features
            if i >= 3:
                short_trend = np.polyfit(range(3), y[i-3:i], 1)[0]
                long_trend = np.polyfit(range(min(6, i)), y[max(0, i-6):i], 1)[0] if i >= 6 else short_trend
                trend_features = [short_trend, long_trend]
            else:
                trend_features = [0, 0]
            
            # Combine all features
            feature_vector = lags + time_features + statistical_features + trend_features
            
            # Add external factors
            if external_factor_cols:
                feature_vector.extend(data[external_factor_cols].iloc[i].values)
            
            X.append(feature_vector)
            y_target.append(y[i])
        
        return np.array(X), np.array(y_target)
    
    @staticmethod
    def enhanced_random_forest_forecast(data: pd.DataFrame, periods: int, 
                                      optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced Random Forest with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 10:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 5:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing Random Forest hyperparameters...")
        best_params, best_score = optimizer.optimize_random_forest(X, y_target)
        print(f"Best Random Forest params: {best_params}, Score: {best_score:.4f}")
        
        # Train final model with best parameters
        model = RandomForestRegressor(**best_params)
        model.fit(X, y_target)
        
        # Generate forecasts
        forecast = EnhancedForecastingEngine.generate_ml_forecast(
            model, data, periods, len(X[0])
        )
        
        # Calculate metrics
        predicted = model.predict(X)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_xgboost_forecast(data: pd.DataFrame, periods: int, 
                                optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced XGBoost with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 10:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 5:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing XGBoost hyperparameters...")
        best_params, best_score = optimizer.optimize_xgboost(X, y_target)
        print(f"Best XGBoost params: {best_params}, Score: {best_score:.4f}")
        
        # Train final model with best parameters
        model = GradientBoostingRegressor(**best_params)
        model.fit(X, y_target)
        
        # Generate forecasts
        forecast = EnhancedForecastingEngine.generate_ml_forecast(
            model, data, periods, len(X[0])
        )
        
        # Calculate metrics
        predicted = model.predict(X)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_svr_forecast(data: pd.DataFrame, periods: int, 
                            optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced SVR with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 8:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 5:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing SVR hyperparameters...")
        best_params, best_score = optimizer.optimize_svr(X, y_target)
        print(f"Best SVR params: {best_params}, Score: {best_score:.4f}")
        
        # Scale features for SVR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train final model with best parameters
        model = SVR(**best_params)
        model.fit(X_scaled, y_target)
        
        # Generate forecasts with scaling
        forecast = EnhancedForecastingEngine.generate_ml_forecast_scaled(
            model, data, periods, len(X[0]), scaler
        )
        
        # Calculate metrics
        predicted = model.predict(X_scaled)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_neural_network_forecast(data: pd.DataFrame, periods: int, 
                                       optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced Neural Network with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 15:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 10:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing Neural Network hyperparameters...")
        best_params, best_score = optimizer.optimize_neural_network(X, y_target)
        print(f"Best Neural Network params: {best_params}, Score: {best_score:.4f}")
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train final model with best parameters
        model = MLPRegressor(**best_params)
        model.fit(X_scaled, y_target)
        
        # Generate forecasts with scaling
        forecast = EnhancedForecastingEngine.generate_ml_forecast_scaled(
            model, data, periods, len(X[0]), scaler
        )
        
        # Calculate metrics
        predicted = model.predict(X_scaled)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_gaussian_process_forecast(data: pd.DataFrame, periods: int, 
                                         optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced Gaussian Process with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 8:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 5:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing Gaussian Process hyperparameters...")
        best_params, best_score = optimizer.optimize_gaussian_process(X, y_target)
        print(f"Best Gaussian Process params: {best_params}, Score: {best_score:.4f}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train final model with best parameters
        model = GaussianProcessRegressor(**best_params)
        model.fit(X_scaled, y_target)
        
        # Generate forecasts with scaling
        forecast = EnhancedForecastingEngine.generate_ml_forecast_scaled(
            model, data, periods, len(X[0]), scaler
        )
        
        # Calculate metrics
        predicted, _ = model.predict(X_scaled, return_std=True)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_knn_forecast(data: pd.DataFrame, periods: int, 
                            optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced KNN with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 8:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        # Prepare features
        X, y_target = EnhancedForecastingEngine.prepare_features_for_ml(data)
        
        if len(X) < 5:
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
        
        print("Optimizing KNN hyperparameters...")
        best_params, best_score = optimizer.optimize_knn(X, y_target)
        print(f"Best KNN params: {best_params}, Score: {best_score:.4f}")
        
        # Train final model with best parameters
        model = KNeighborsRegressor(**best_params)
        model.fit(X, y_target)
        
        # Generate forecasts
        forecast = EnhancedForecastingEngine.generate_ml_forecast(
            model, data, periods, len(X[0])
        )
        
        # Calculate metrics
        predicted = model.predict(X)
        metrics = EnhancedForecastingEngine.calculate_metrics(y_target, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def enhanced_exponential_smoothing_forecast(data: pd.DataFrame, periods: int, 
                                              optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced Exponential Smoothing with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        
        if len(y) < 3:
            return np.full(periods, y[-1] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}
        
        print("Optimizing Exponential Smoothing hyperparameters...")
        best_params, best_score = optimizer.optimize_exponential_smoothing(y)
        print(f"Best Exponential Smoothing params: {best_params}, Score: {best_score:.4f}")
        
        try:
            # Train final model with best parameters
            model = ExponentialSmoothing(
                y,
                trend=best_params.get('trend'),
                seasonal=best_params.get('seasonal'),
                seasonal_periods=best_params.get('seasonal_periods'),
                initialization_method="estimated"
            )
            
            fit = model.fit(
                smoothing_level=best_params.get('smoothing_level'),
                smoothing_trend=best_params.get('smoothing_trend'),
                smoothing_seasonal=best_params.get('smoothing_seasonal'),
                optimized=best_params.get('smoothing_level') is None
            )
            
            forecast = fit.forecast(periods)
            forecast = np.maximum(forecast, 0)
            
            # Calculate metrics
            fitted = fit.fittedvalues
            metrics = EnhancedForecastingEngine.calculate_metrics(y, fitted)
            
            return forecast, metrics
            
        except Exception as e:
            print(f"Enhanced Exponential Smoothing failed: {e}")
            return EnhancedForecastingEngine.fallback_linear_forecast(data, periods)
    
    @staticmethod
    def enhanced_holt_winters_forecast(data: pd.DataFrame, periods: int, 
                                     optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced Holt-Winters with comprehensive hyperparameter optimization"""
        y = data['quantity'].values
        
        if len(y) < 8:
            return EnhancedForecastingEngine.enhanced_exponential_smoothing_forecast(data, periods, optimizer)
        
        print("Optimizing Holt-Winters hyperparameters...")
        best_params, best_score = optimizer.optimize_holt_winters(y)
        print(f"Best Holt-Winters params: {best_params}, Score: {best_score:.4f}")
        
        try:
            # Train final model with best parameters
            model = ExponentialSmoothing(
                y,
                trend=best_params.get('trend'),
                seasonal=best_params.get('seasonal'),
                seasonal_periods=best_params.get('seasonal_periods'),
                damped_trend=best_params.get('damped_trend', False),
                initialization_method="estimated"
            )
            
            fit = model.fit(optimized=True)
            forecast = fit.forecast(periods)
            forecast = np.maximum(forecast, 0)
            
            # Calculate metrics
            fitted = fit.fittedvalues
            metrics = EnhancedForecastingEngine.calculate_metrics(y, fitted)
            
            return forecast, metrics
            
        except Exception as e:
            print(f"Enhanced Holt-Winters failed: {e}")
            return EnhancedForecastingEngine.enhanced_exponential_smoothing_forecast(data, periods, optimizer)
    
    @staticmethod
    def enhanced_arima_forecast(data: pd.DataFrame, periods: int, 
                              optimizer: TimeSeriesOptimizer) -> Tuple[np.ndarray, Dict[str, float]]:
        """Enhanced ARIMA with time-series-aware hyperparameter optimization"""
        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns 
                              if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        
        if len(y) < 10:
            return EnhancedForecastingEngine.enhanced_exponential_smoothing_forecast(data, periods, optimizer)
        
        # Prepare external regressors if available
        exog = data[external_factor_cols].values if external_factor_cols else None
        
        print("Optimizing ARIMA hyperparameters...")
        best_params, best_score = optimizer.optimize_arima_with_external(y, exog)
        print(f"Best ARIMA params: {best_params}, Score: {best_score:.4f}")
        
        try:
            # Train final model with best parameters
            model = ARIMA(y, exog=exog, order=best_params['order'])
            fit = model.fit()
            
            # Forecast external factors if needed
            if external_factor_cols:
                future_factors = EnhancedForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
                future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                forecast = fit.forecast(steps=periods, exog=future_exog)
            else:
                forecast = fit.forecast(steps=periods)
            
            forecast = np.maximum(forecast, 0)
            
            # Calculate metrics
            fitted = fit.fittedvalues
            if len(fitted) == len(y):
                metrics = EnhancedForecastingEngine.calculate_metrics(y, fitted)
            else:
                start_idx = len(y) - len(fitted)
                metrics = EnhancedForecastingEngine.calculate_metrics(y[start_idx:], fitted)
            
            return forecast, metrics
            
        except Exception as e:
            print(f"Enhanced ARIMA failed: {e}")
            return EnhancedForecastingEngine.enhanced_exponential_smoothing_forecast(data, periods, optimizer)
    
    @staticmethod
    def generate_ml_forecast(model, data: pd.DataFrame, periods: int, feature_count: int) -> np.ndarray:
        """Generate forecasts for ML models"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        external_factor_cols = [col for col in data.columns 
                              if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        
        forecast = []
        window_size = min(5, len(y) - 1)
        recent_values = list(y[-window_size:])
        last_date = dates.iloc[-1]
        
        # Forecast external factors if needed
        future_factors = {}
        if external_factor_cols:
            future_factors = EnhancedForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
        
        for i in range(periods):
            # Prepare features for this forecast step
            next_date = last_date + pd.DateOffset(months=i+1)
            
            # Lag features
            lags = list(recent_values)
            
            # Time-based features
            time_features = [
                len(y) + i,
                next_date.month,
                next_date.quarter,
                next_date.dayofyear,
                next_date.weekday(),
                (len(y) + i) % 12,
                (len(y) + i) % 4,
            ]
            
            # Statistical features
            statistical_features = [
                np.mean(recent_values),
                np.std(recent_values) if len(recent_values) > 1 else 0,
                np.min(recent_values),
                np.max(recent_values),
                np.median(recent_values),
            ]
            
            # Trend features
            if len(recent_values) >= 3:
                short_trend = np.polyfit(range(3), recent_values[-3:], 1)[0]
                long_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                trend_features = [short_trend, long_trend]
            else:
                trend_features = [0, 0]
            
            # Combine features
            feature_vector = lags + time_features + statistical_features + trend_features
            
            # Add external factors
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                feature_vector.extend(forecasted_factors)
            
            # Ensure feature vector has correct length
            while len(feature_vector) < feature_count:
                feature_vector.append(0)
            feature_vector = feature_vector[:feature_count]
            
            # Predict
            next_pred = model.predict([feature_vector])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            
            # Update recent values
            recent_values = recent_values[1:] + [next_pred]
        
        return np.array(forecast)
    
    @staticmethod
    def generate_ml_forecast_scaled(model, data: pd.DataFrame, periods: int, 
                                  feature_count: int, scaler) -> np.ndarray:
        """Generate forecasts for ML models that require feature scaling"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        external_factor_cols = [col for col in data.columns 
                              if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        
        forecast = []
        window_size = min(5, len(y) - 1)
        recent_values = list(y[-window_size:])
        last_date = dates.iloc[-1]
        
        # Forecast external factors if needed
        future_factors = {}
        if external_factor_cols:
            future_factors = EnhancedForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
        
        for i in range(periods):
            # Prepare features (same as generate_ml_forecast)
            next_date = last_date + pd.DateOffset(months=i+1)
            
            lags = list(recent_values)
            time_features = [
                len(y) + i, next_date.month, next_date.quarter,
                next_date.dayofyear, next_date.weekday(),
                (len(y) + i) % 12, (len(y) + i) % 4,
            ]
            
            statistical_features = [
                np.mean(recent_values), np.std(recent_values) if len(recent_values) > 1 else 0,
                np.min(recent_values), np.max(recent_values), np.median(recent_values),
            ]
            
            if len(recent_values) >= 3:
                short_trend = np.polyfit(range(3), recent_values[-3:], 1)[0]
                long_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                trend_features = [short_trend, long_trend]
            else:
                trend_features = [0, 0]
            
            feature_vector = lags + time_features + statistical_features + trend_features
            
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                feature_vector.extend(forecasted_factors)
            
            # Ensure correct length and scale
            while len(feature_vector) < feature_count:
                feature_vector.append(0)
            feature_vector = feature_vector[:feature_count]
            
            # Scale and predict
            feature_vector_scaled = scaler.transform([feature_vector])
            next_pred = model.predict(feature_vector_scaled)[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            
            recent_values = recent_values[1:] + [next_pred]
        
        return np.array(forecast)
    
    @staticmethod
    def forecast_external_factors(data: pd.DataFrame, external_factor_cols: List[str], periods: int) -> Dict[str, np.ndarray]:
        """Enhanced external factor forecasting with trend analysis"""
        future_factors = {}
        
        for col in external_factor_cols:
            if col not in data.columns:
                continue
            
            values = data[col].dropna().values
            if len(values) < 2:
                future_factors[col] = np.full(periods, values[-1] if len(values) > 0 else 0)
                continue
            
            try:
                # Use more sophisticated trend analysis
                x = np.arange(len(values))
                
                # Try polynomial fits of different degrees
                best_forecast = None
                best_score = float('inf')
                
                for degree in [1, 2, 3]:
                    try:
                        coeffs = np.polyfit(x, values, degree)
                        poly_func = np.poly1d(coeffs)
                        
                        # Evaluate on last 20% of data
                        split_idx = int(len(values) * 0.8)
                        if split_idx < len(values) - 1:
                            train_values = values[:split_idx]
                            test_values = values[split_idx:]
                            
                            train_coeffs = np.polyfit(np.arange(len(train_values)), train_values, degree)
                            train_poly = np.poly1d(train_coeffs)
                            
                            test_x = np.arange(len(train_values), len(values))
                            test_pred = train_poly(test_x)
                            
                            score = np.sqrt(mean_squared_error(test_values, test_pred))
                            
                            if score < best_score:
                                best_score = score
                                future_x = np.arange(len(values), len(values) + periods)
                                best_forecast = poly_func(future_x)
                    except:
                        continue
                
                if best_forecast is not None:
                    future_factors[col] = best_forecast
                else:
                    # Fallback to last value
                    future_factors[col] = np.full(periods, values[-1])
                    
            except Exception as e:
                future_factors[col] = np.full(periods, values[-1])
        
        return future_factors
    
    @staticmethod
    def fallback_linear_forecast(data: pd.DataFrame, periods: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fallback linear regression forecast"""
        y = data['quantity'].values
        n = len(y)
        
        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}
        
        X = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(n, n + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        forecast = np.maximum(forecast, 0)
        
        predicted = model.predict(X)
        metrics = EnhancedForecastingEngine.calculate_metrics(y, predicted)
        
        return forecast, metrics
    
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Calculate accuracy as percentage
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        accuracy = max(0, 100 - mape)
        
        return {
            'accuracy': min(accuracy, 99.9),
            'mae': mae,
            'rmse': rmse
        }

class AdaptiveParameterSelector:
    """Adaptive parameter selection based on data characteristics"""
    
    @staticmethod
    def analyze_data_characteristics(y: np.ndarray) -> Dict[str, Any]:
        """Analyze time series characteristics to guide parameter selection"""
        n = len(y)
        
        # Basic statistics
        mean_val = np.mean(y)
        std_val = np.std(y)
        cv = std_val / mean_val if mean_val != 0 else 0
        
        # Trend analysis
        x = np.arange(n)
        slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)[:5] if n > 1 else (0, 0, 0, 0, 0)
        trend_strength = abs(slope) / (std_val + 1e-8)
        
        # Seasonality detection (simple)
        seasonality_strength = 0
        if n >= 24:  # Need at least 2 years for monthly seasonality
            try:
                # Check for 12-month seasonality
                seasonal_diff = np.abs(np.diff(y, 12))
                seasonality_strength = 1 - (np.mean(seasonal_diff) / (std_val + 1e-8))
                seasonality_strength = max(0, min(1, seasonality_strength))
            except:
                seasonality_strength = 0
        
        # Volatility analysis
        if n > 1:
            returns = np.diff(y) / (y[:-1] + 1e-8)
            volatility = np.std(returns)
        else:
            volatility = 0
        
        # Data quality indicators
        zero_ratio = np.sum(y == 0) / n
        outlier_ratio = np.sum(np.abs(y - mean_val) > 3 * std_val) / n
        
        return {
            'length': n,
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'trend_strength': trend_strength,
            'seasonality_strength': seasonality_strength,
            'volatility': volatility,
            'zero_ratio': zero_ratio,
            'outlier_ratio': outlier_ratio,
            'r_squared': r_value**2 if n > 1 else 0
        }
    
    @staticmethod
    def select_algorithm_priorities(characteristics: Dict[str, Any]) -> List[str]:
        """Select algorithm priorities based on data characteristics"""
        algorithms = []
        
        # High seasonality -> prioritize seasonal methods
        if characteristics['seasonality_strength'] > 0.3:
            algorithms.extend(['holt_winters', 'sarima', 'prophet_like', 'seasonal_decomposition'])
        
        # Strong trend -> prioritize trend-aware methods
        if characteristics['trend_strength'] > 0.2:
            algorithms.extend(['arima', 'linear_regression', 'polynomial_regression', 'drift_method'])
        
        # High volatility -> prioritize robust ML methods
        if characteristics['volatility'] > 0.1:
            algorithms.extend(['random_forest', 'xgboost', 'svr', 'gaussian_process'])
        
        # Intermittent demand (high zero ratio) -> prioritize specialized methods
        if characteristics['zero_ratio'] > 0.3:
            algorithms.extend(['croston', 'ses', 'theta_method'])
        
        # Stable data -> prioritize smoothing methods
        if characteristics['cv'] < 0.2 and characteristics['trend_strength'] < 0.1:
            algorithms.extend(['exponential_smoothing', 'moving_average', 'ses'])
        
        # Complex patterns -> prioritize advanced ML
        if characteristics['length'] > 50 and characteristics['cv'] > 0.3:
            algorithms.extend(['lstm_like', 'neural_network', 'xgboost', 'random_forest'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_algorithms = []
        for alg in algorithms:
            if alg not in seen:
                seen.add(alg)
                unique_algorithms.append(alg)
        
        # Add remaining algorithms
        all_algorithms = [
            'linear_regression', 'polynomial_regression', 'exponential_smoothing',
            'holt_winters', 'arima', 'random_forest', 'seasonal_decomposition',
            'moving_average', 'sarima', 'prophet_like', 'lstm_like', 'xgboost',
            'svr', 'knn', 'gaussian_process', 'neural_network', 'theta_method',
            'croston', 'ses', 'damped_trend', 'naive_seasonal', 'drift_method'
        ]
        
        for alg in all_algorithms:
            if alg not in unique_algorithms:
                unique_algorithms.append(alg)
        
        return unique_algorithms
    
    @staticmethod
    def get_adaptive_cv_splits(data_length: int) -> int:
        """Get adaptive number of CV splits based on data length"""
        if data_length < 20:
            return 3
        elif data_length < 50:
            return 4
        elif data_length < 100:
            return 5
        else:
            return 6
    
    @staticmethod
    def get_adaptive_test_size(data_length: int, forecast_periods: int) -> int:
        """Get adaptive test size for time series CV"""
        # Test size should be similar to forecast horizon but not too large
        min_test_size = max(1, forecast_periods)
        max_test_size = max(min_test_size, data_length // 4)
        
        return min(max_test_size, max(min_test_size, data_length // 6))

class BayesianOptimizer:
    """Bayesian optimization for complex hyperparameter spaces"""
    
    @staticmethod
    def optimize_prophet_parameters(y: np.ndarray, exog: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize Prophet parameters using Bayesian optimization"""
        try:
            from prophet import Prophet
            
            def objective(trial):
                params = {
                    'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                    'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                    'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
                    'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                    'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
                    'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False, 'auto']),
                    'weekly_seasonality': False,  # Assuming monthly data
                    'daily_seasonality': False,
                    'interval_width': trial.suggest_float('interval_width', 0.8, 0.95)
                }
                
                # Time series cross-validation for Prophet
                scores = []
                tscv = TimeSeriesSplit(n_splits=3, test_size=max(1, len(y) // 6))
                
                for train_idx, test_idx in tscv.split(y):
                    try:
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Create Prophet dataframe
                        train_df = pd.DataFrame({
                            'ds': pd.date_range(start='2020-01-01', periods=len(y_train), freq='MS'),
                            'y': y_train
                        })
                        
                        model = Prophet(**params)
                        
                        # Add external regressors if available
                        if exog is not None:
                            for i in range(exog.shape[1]):
                                col_name = f'regressor_{i}'
                                train_df[col_name] = exog[train_idx, i]
                                model.add_regressor(col_name)
                        
                        model.fit(train_df)
                        
                        # Create future dataframe
                        future = model.make_future_dataframe(periods=len(y_test), freq='MS')
                        
                        if exog is not None:
                            for i in range(exog.shape[1]):
                                col_name = f'regressor_{i}'
                                future[col_name] = np.concatenate([exog[train_idx, i], exog[test_idx, i]])
                        
                        forecast = model.predict(future)
                        y_pred = forecast['yhat'].iloc[-len(y_test):].values
                        y_pred = np.maximum(y_pred, 0)
                        
                        score = np.sqrt(mean_squared_error(y_test, y_pred))
                        scores.append(score)
                        
                    except:
                        scores.append(1e6)
                
                return np.mean(scores) if scores else 1e6
            
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            
            return study.best_params
            
        except Exception as e:
            print(f"Prophet optimization failed: {e}")
            return {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False
            }