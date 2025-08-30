@@ .. @@
 from validation import DateRangeValidator
 import requests
 import os
+from hyperparameter_optimizer import (
+    TimeSeriesOptimizer, 
+    EnhancedForecastingEngine, 
+    AdaptiveParameterSelector,
+    BayesianOptimizer
+)
 warnings.filterwarnings('ignore')
 
 app = FastAPI(title="Multi-variant Forecasting API with MySQL", version="3.0.0")
@@ .. @@
     @staticmethod
     def random_forest_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100, 200], max_depth_list: list = [3, 5, None]) -> tuple:
-        """Random Forest regression forecasting with hyperparameter tuning"""
+        """Enhanced Random Forest regression forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
-        dates = pd.to_datetime(data['date'])
-
-        # Get external factor columns
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-
-        # Create features
-        features = []
-        targets = []
-        window = min(5, len(y) - 1)
-
-        for i in range(window, len(y)):
-            lags = y[i-window:i]
-            trend = i
-            seasonal = i % 12
-            month = dates.iloc[i].month
-            quarter = dates.iloc[i].quarter
-            feature_vector = list(lags) + [trend, seasonal, month, quarter]
-            if external_factor_cols:
-                feature_vector.extend(data[external_factor_cols].iloc[i].values)
-            features.append(feature_vector)
-            targets.append(y[i])
-
-        if len(features) < 3:
-            return ForecastingEngine.linear_regression_forecast(data, periods)
-
-        features = np.array(features)
-        targets = np.array(targets)
-
-        best_metrics = None
-        best_forecast = None
-
-        for n_estimators in n_estimators_list:
-            for max_depth in max_depth_list:
-                print(f"Running Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
-                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
-                model.fit(features, targets)
-
-                # Forecast
-                forecast = []
-                recent_values = list(y[-window:])
-                last_date = dates.iloc[-1]
-
-                for i in range(periods):
-                    trend_val = len(y) + i
-                    seasonal_val = (len(y) + i) % 12
-                    next_date = last_date + pd.DateOffset(months=i+1)
-                    month_val = next_date.month
-                    quarter_val = next_date.quarter
-                    feature_vector = recent_values + [trend_val, seasonal_val, month_val, quarter_val]
-                    if external_factor_cols:
-                        # Use forecasted external factors
-                        if 'future_factors' not in locals():
-                            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
-                        forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
-                        feature_vector.extend(forecasted_factors)
-
-                    next_value = model.predict([feature_vector])[0]
-                    next_value = max(0, next_value)
-                    forecast.append(next_value)
-                    recent_values = recent_values[1:] + [next_value]
-
-                predicted = model.predict(features)
-                metrics = ForecastingEngine.calculate_metrics(targets, predicted)
-                print(f"n_estimators={n_estimators}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")
-
-                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
-                    best_metrics = metrics
-                    best_forecast = forecast
-
-        return np.array(best_forecast), best_metrics
+        n = len(y)
+        
+        if n < 10:
+            return ForecastingEngine.linear_regression_forecast(data, periods)
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced Random Forest with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_random_forest_forecast(data, periods, optimizer)
 
     @staticmethod
     def seasonal_decomposition_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
@@ .. @@
     @staticmethod
     def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alphas: list = [0.1,0.3,0.5]) -> tuple:
-        """Enhanced exponential smoothing with external factors integration"""
+        """Enhanced exponential smoothing with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-
+        
         if n < 3:
             return np.full(periods, y[-1] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}
-
-        best_metrics = None
-        best_forecast = None
-
-        for alpha in alphas:
-            print(f"Running Exponential Smoothing with alpha={alpha}")
-
-            if external_factor_cols:
-                # Use regression-based exponential smoothing with external factors
-                window = min(5, n - 1)
-                X, y_target = [], []
-
-                for i in range(window, n):
-                    # Exponentially weighted historical values
-                    weights = np.array([alpha * (1 - alpha) ** j for j in range(window)])
-                    weights = weights / weights.sum()
-                    weighted_history = np.sum(weights * y[i-window:i])
-
-                    features = [weighted_history, i]  # Smoothed value + trend
-                    if external_factor_cols:
-                        features.extend(data[external_factor_cols].iloc[i].values)
-
-                    X.append(features)
-                    y_target.append(y[i])
-
-                if len(X) > 1:
-                    X = np.array(X)
-                    y_target = np.array(y_target)
-
-                    # Fit linear model with smoothed features
-                    model = LinearRegression()
-                    model.fit(X, y_target)
-
-                    # Forecast with external factors
-                    forecast = []
-                    last_values = y[-window:]
-
-                    for i in range(periods):
-                        weights = np.array([alpha * (1 - alpha) ** j for j in range(len(last_values))])
-                        weights = weights / weights.sum()
-                        weighted_history = np.sum(weights * last_values)
-
-                        features = [weighted_history, n + i]
-                        if external_factor_cols:
-                            # Use last known external factor values
-                            features.extend(data[external_factor_cols].iloc[-1].values)
-
-                        pred = model.predict([features])[0]
-                        pred = max(0, pred)
-                        forecast.append(pred)
-
-                        # Update last_values for next prediction
-                        last_values = np.append(last_values[1:], pred)
-
-                    predicted = model.predict(X)
-                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
-                else:
-                    # Fallback to simple smoothing
-                    smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
-                    forecast = np.full(periods, smoothed[-1])
-                    metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])
-            else:
-                # Traditional exponential smoothing without external factors
-                smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
-                forecast = np.full(periods, smoothed[-1])
-                metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])
-
-            print(f"Alpha={alpha}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")
-
-            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
-                best_metrics = metrics
-                best_forecast = forecast
-
-        return np.array(best_forecast), best_metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced Exponential Smoothing with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_exponential_smoothing_forecast(data, periods, optimizer)
 
     @staticmethod
     def holt_winters_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
-        """Enhanced Holt-Winters with external factors integration"""
+        """Enhanced Holt-Winters with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-
+        
         if n < 2 * season_length:
             return ForecastingEngine.exponential_smoothing_forecast(data, periods)
-
-        if external_factor_cols:
-            # Enhanced Holt-Winters with external factor regression
-            # First decompose the series using traditional Holt-Winters
-            alpha, beta, gamma = 0.3, 0.1, 0.1
-
-            level = np.mean(y[:season_length])
-            trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
-            seasonal = y[:season_length] - level
-
-            levels = [level]
-            trends = [trend]
-            seasonals = list(seasonal)
-            fitted = []
-            residuals = []
-
-            # Apply Holt-Winters to get base forecast
-            for i in range(len(y)):
-                if i == 0:
-                    forecast_val = level + trend + seasonal[i % season_length]
-                    fitted.append(forecast_val)
-                    residuals.append(y[i] - forecast_val)
-                else:
-                    level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
-                    trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
-                    if len(seasonals) > i:
-                        seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]
-
-                    levels.append(level)
-                    trends.append(trend)
-                    forecast_val = level + trend + seasonals[i % season_length]
-                    fitted.append(forecast_val)
-                    residuals.append(y[i] - forecast_val)
-
-            # Use external factors to model residuals
-            window = min(3, len(residuals) - 1)
-            if window > 0 and len(residuals) > window:
-                X, y_residual = [], []
-                for i in range(window, len(residuals)):
-                    features = list(residuals[i-window:i])  # Lag residuals
-                    features.extend(data[external_factor_cols].iloc[i].values)
-                    X.append(features)
-                    y_residual.append(residuals[i])
-
-                if len(X) > 1:
-                    X = np.array(X)
-                    y_residual = np.array(y_residual)
-
-                    # Model residuals with external factors
-                    residual_model = LinearRegression()
-                    residual_model.fit(X, y_residual)
-
-                    # Forecast with external factors
-                    forecast = []
-                    recent_residuals = residuals[-window:]
-
-                    for i in range(periods):
-                        # Base Holt-Winters forecast
-                        hw_forecast = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]
-
-                        # Residual correction using external factors
-                        features = list(recent_residuals)
-                        features.extend(data[external_factor_cols].iloc[-1].values)
-
-                        residual_correction = residual_model.predict([features])[0]
-                        final_forecast = hw_forecast + residual_correction
-                        final_forecast = max(0, final_forecast)
-
-                        forecast.append(final_forecast)
-                        recent_residuals = recent_residuals[1:] + [residual_correction]
-
-                    # Calculate metrics on corrected fitted values
-                    fitted_corrected = np.array(fitted) + residual_model.predict(X)
-                    metrics = ForecastingEngine.calculate_metrics(y[window:], fitted_corrected)
-
-                    return np.array(forecast), metrics
-
-        # Traditional Holt-Winters without external factors
-        alpha, beta, gamma = 0.3, 0.1, 0.1
-
-        level = np.mean(y[:season_length])
-        trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
-        seasonal = y[:season_length] - level
-
-        levels = [level]
-        trends = [trend]
-        seasonals = list(seasonal)
-        fitted = []
-
-        for i in range(len(y)):
-            if i == 0:
-                fitted.append(level + trend + seasonal[i % season_length])
-            else:
-                level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
-                trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
-                if len(seasonals) > i:
-                    seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]
-
-                levels.append(level)
-                trends.append(trend)
-                fitted.append(level + trend + seasonals[i % season_length])
-
-        forecast = []
-        for i in range(periods):
-            forecast_value = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]
-            forecast.append(max(0, forecast_value))
-
-        metrics = ForecastingEngine.calculate_metrics(y, fitted)
-
-        return np.array(forecast), metrics
+        
+        if n < 2 * season_length:
+            return ForecastingEngine.exponential_smoothing_forecast(data, periods)
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced Holt-Winters with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_holt_winters_forecast(data, periods, optimizer)
 
     @staticmethod
     def arima_forecast(data: pd.DataFrame, periods: int) -> tuple:
-        """Actual ARIMA/ARIMAX forecasting using statsmodels"""
-        from statsmodels.tsa.arima.model import ARIMA
-        from pmdarima import auto_arima
-        import warnings
-
+        """Enhanced ARIMA/ARIMAX forecasting with time-series-aware hyperparameter optimization"""
         y = data['quantity'].values
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-
-        if len(y) < 10:  # Need sufficient data for ARIMA
+        n = len(y)
+        
+        if n < 10:
             return ForecastingEngine.exponential_smoothing_forecast(data, periods)
-
-        try:
-            with warnings.catch_warnings():
-                warnings.filterwarnings("ignore")
-
-                if external_factor_cols:
-                    # ARIMAX: ARIMA with external regressors
-                    print(f"Running ARIMAX with external factors: {external_factor_cols}")
-
-                    # Prepare external regressors
-                    exog = data[external_factor_cols].values
-
-                    # Auto-select ARIMA parameters with external regressors
-                    try:
-                        auto_model = auto_arima(
-                            y, 
-                            exogenous=exog,
-                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
-                            seasonal=False,
-                            stepwise=True,
-                            suppress_warnings=True,
-                            error_action='ignore',
-                            trace=False
-                        )
-
-                        # Forecast external factors properly
-                        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-                        future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
-
-                        # Create future exogenous matrix
-                        future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
-                        forecast = auto_model.predict(n_periods=periods, exogenous=future_exog)
-                        forecast = np.maximum(forecast, 0)
-
-                        # Calculate metrics
-                        fitted = auto_model.fittedvalues()
-                        if len(fitted) == len(y):
-                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
-                        else:
-                            # Handle case where fitted values might be shorter due to differencing
-                            start_idx = len(y) - len(fitted)
-                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)
-
-                        print(f"ARIMAX model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
-                        return forecast, metrics
-
-                    except Exception as e:
-                        print(f"Auto ARIMAX failed: {e}, trying manual ARIMAX")
-
-                        # Fallback to manual ARIMAX
-                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1)]:
-                            try:
-                                model = ARIMA(y, exog=exog, order=order)
-                                fitted_model = model.fit()
-
-                                # Use forecasted external factors
-                                future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
-                                forecast = fitted_model.forecast(steps=periods, exog=future_exog)
-                                forecast = np.maximum(forecast, 0)
-
-                                fitted = fitted_model.fittedvalues
-                                if len(fitted) == len(y):
-                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
-                                else:
-                                    start_idx = len(y) - len(fitted)
-                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)
-
-                                print(f"Manual ARIMAX model: {order}, AIC: {fitted_model.aic:.2f}")
-                                return forecast, metrics
-
-                            except:
-                                continue
-
-                else:
-                    # Traditional ARIMA without external factors
-                    print("Running ARIMA without external factors")
-
-                    try:
-                        # Auto-select ARIMA parameters
-                        auto_model = auto_arima(
-                            y,
-                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
-                            seasonal=False,
-                            stepwise=True,
-                            suppress_warnings=True,
-                            error_action='ignore',
-                            trace=False
-                        )
-
-                        forecast = auto_model.predict(n_periods=periods)
-                        forecast = np.maximum(forecast, 0)
-
-                        # Calculate metrics
-                        fitted = auto_model.fittedvalues()
-                        if len(fitted) == len(y):
-                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
-                        else:
-                            start_idx = len(y) - len(fitted)
-                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)
-
-                        print(f"ARIMA model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
-                        return forecast, metrics
-
-                    except Exception as e:
-                        print(f"Auto ARIMA failed: {e}, trying manual ARIMA")
-
-                        # Fallback to manual ARIMA
-                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1), (1,0,1)]:
-                            try:
-                                model = ARIMA(y, order=order)
-                                fitted_model = model.fit()
-
-                                forecast = fitted_model.forecast(steps=periods)
-                                forecast = np.maximum(forecast, 0)
-
-                                fitted = fitted_model.fittedvalues
-                                if len(fitted) == len(y):
-                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
-                                else:
-                                    start_idx = len(y) - len(fitted)
-                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)
-
-                                print(f"Manual ARIMA model: {order}, AIC: {fitted_model.aic:.2f}")
-                                return forecast, metrics
-
-                            except:
-                                continue
-
-        except Exception as e:
-            print(f"ARIMA forecasting failed: {e}")
-
-        # Final fallback to exponential smoothing
-        print("ARIMA failed, falling back to exponential smoothing")
-        return ForecastingEngine.exponential_smoothing_forecast(data, periods)
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced ARIMA with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_arima_forecast(data, periods, optimizer)
 
     @staticmethod
     def xgboost_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100], learning_rate_list: list = [0.05, 0.1, 0.2], max_depth_list: list = [3, 4, 5]) -> tuple:
-        """XGBoost-like forecasting with hyperparameter tuning and external factors"""
+        """Enhanced XGBoost forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
-        dates = pd.to_datetime(data['date'])
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-        if n < 6:
+        
+        if n < 10:
             return ForecastingEngine.linear_regression_forecast(data, periods)
-        features = []
-        targets = []
-        window = min(4, n - 1)
-        for i in range(window, n):
-            lags = list(y[i-window:i])
-            date = dates.iloc[i]
-            time_features = [
-                i,
-                date.month,
-                date.quarter,
-                date.dayofyear % 7,
-                i % 12,
-            ]
-            recent_mean = np.mean(y[max(0, i-3):i])
-            recent_std = np.std(y[max(0, i-3):i]) if i > 3 else 0
-            feature_vector = lags + time_features + [recent_mean, recent_std]
-            if external_factor_cols:
-                feature_vector.extend(data[external_factor_cols].iloc[i].values)
-            features.append(feature_vector)
-            targets.append(y[i])
-        if len(features) < 3:
-            return ForecastingEngine.random_forest_forecast(data, periods)
-        features = np.array(features)
-        targets = np.array(targets)
-        best_metrics = None
-        best_forecast = None
-        for n_estimators in n_estimators_list:
-            for learning_rate in learning_rate_list:
-                for max_depth in max_depth_list:
-                    print(f"Running XGBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
-                    try:
-                        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
-                        model.fit(features, targets)
-                        forecast = []
-                        recent_values = list(y[-window:])
-                        last_date = dates.iloc[-1]
-                        for i in range(periods):
-                            next_date = last_date + pd.DateOffset(months=i+1)
-                            time_features = [
-                                n + i,
-                                next_date.month,
-                                next_date.quarter,
-                                next_date.dayofyear % 7,
-                                (n + i) % 12,
-                            ]
-                            recent_mean = np.mean(recent_values[-3:])
-                            recent_std = np.std(recent_values[-3:]) if len(recent_values) > 1 else 0
-                            feature_vector = recent_values + time_features + [recent_mean, recent_std]
-                            if external_factor_cols:
-                                last_factors = data[external_factor_cols].iloc[-1].values
-                                feature_vector = list(feature_vector) + list(last_factors)
-                            next_pred = model.predict([feature_vector])[0]
-                            next_pred = max(0, next_pred)
-                            forecast.append(next_pred)
-                            recent_values = recent_values[1:] + [next_pred]
-                        predicted = model.predict(features)
-                        metrics = ForecastingEngine.calculate_metrics(targets, predicted)
-                        print(f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")
-                        if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
-                            best_metrics = metrics
-                            best_forecast = forecast
-                    except Exception as e:
-                        print(f"Error running XGBoost with params: {e}")
-                        continue
-        return np.array(best_forecast), best_metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced XGBoost with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_xgboost_forecast(data, periods, optimizer)
 
     @staticmethod
     def svr_forecast(data: pd.DataFrame, periods: int, C_list: list = [1, 10, 100], epsilon_list: list = [0.1, 0.2]) -> tuple:
-        """Support Vector Regression forecasting with hyperparameter tuning and external factors"""
+        """Enhanced Support Vector Regression forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-        if n < 4:
+        
+        if n < 8:
             return ForecastingEngine.linear_regression_forecast(data, periods)
-        window = min(3, n - 1)
-        X, y_target = [], []
-        for i in range(window, n):
-            features = [i] + list(y[i-window:i])
-            if external_factor_cols:
-                features.extend(data[external_factor_cols].iloc[i].values)
-            X.append(features)
-            y_target.append(y[i])
-        if len(X) < 2:
-            return ForecastingEngine.linear_regression_forecast(data, periods)
-        X = np.array(X)
-        y_target = np.array(y_target)
-        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
-        X_std[X_std == 0] = 1
-        X_norm = (X - X_mean) / X_std
-        param_grid = {'C': C_list, 'epsilon': epsilon_list}
-        model = SVR(kernel='rbf', gamma='scale')
-        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
-        grid_search.fit(X_norm, y_target)
-        best_model = grid_search.best_estimator_
-        print(f"Best SVR params: {grid_search.best_params_}")
-        forecast = []
-        recent_values = list(y[-window:])
-        # Forecast external factors if needed
-        if external_factor_cols:
-            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
-
-        for i in range(periods):
-            features = [n + i] + recent_values
-            if external_factor_cols:
-                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
-                features.extend(forecasted_factors)
-            features_norm = (np.array(features) - X_mean) / X_std
-            next_pred = best_model.predict([features_norm])[0]
-            next_pred = max(0, next_pred)
-            forecast.append(next_pred)
-            recent_values = recent_values[1:] + [next_pred]
-        predicted = best_model.predict(X_norm)
-        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
-        return np.array(forecast), metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced SVR with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_svr_forecast(data, periods, optimizer)
 
     @staticmethod
     def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors_list: list = [7, 10]) -> tuple:
-        """K-Nearest Neighbors forecasting with hyperparameter tuning and external factors"""
+        """Enhanced K-Nearest Neighbors forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-        if n < 6:
+        
+        if n < 8:
             return ForecastingEngine.linear_regression_forecast(data, periods)
-        window = min(4, n - 1)
-        X, y_target = [], []
-        for i in range(window, n):
-            features = list(y[i-window:i])
-            if external_factor_cols:
-                features.extend(data[external_factor_cols].iloc[i].values)
-            X.append(features)
-            y_target.append(y[i])
-        if len(X) < 3:
-            return ForecastingEngine.linear_regression_forecast(data, periods)
-        X = np.array(X)
-        y_target = np.array(y_target)
-        param_grid = {'n_neighbors': n_neighbors_list}
-        model = KNeighborsRegressor(weights='distance')
-        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
-        grid_search.fit(X, y_target)
-        best_model = grid_search.best_estimator_
-        print(f"Best KNN params: {grid_search.best_params_}")
-        forecast = []
-        current_window = list(y[-window:])
-
-        # Forecast external factors if needed
-        if external_factor_cols:
-            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
-
-        for i in range(periods):
-            features = list(current_window)
-            if external_factor_cols:
-                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
-                features.extend(forecasted_factors)
-            next_pred = best_model.predict([features])[0]
-            next_pred = max(0, next_pred)
-            forecast.append(next_pred)
-            current_window = current_window[1:] + [next_pred]
-        predicted = best_model.predict(X)
-        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
-        return np.array(forecast), metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced KNN with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_knn_forecast(data, periods, optimizer)
 
     @staticmethod
     def gaussian_process_forecast(data: pd.DataFrame, periods: int) -> tuple:
-        """Improved Gaussian Process Regression forecasting with hyperparameter tuning and scaling"""
-        from sklearn.preprocessing import StandardScaler
-        from sklearn.model_selection import GridSearchCV
+        """Enhanced Gaussian Process Regression forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-
-        if n < 4:
+        
+        if n < 8:
             return ForecastingEngine.linear_regression_forecast(data, periods)
-
-        # Create time features
-        X = np.arange(n).reshape(-1, 1)
-
-        # Scale features
-        scaler = StandardScaler()
-        X_scaled = scaler.fit_transform(X)
-
-        try:
-            # Define kernel with initial parameters
-            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
-
-            # Create GP model
-            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42, normalize_y=True)
-
-            # Hyperparameter tuning for kernel parameters
-            param_grid = {
-                "kernel__k1__constant_value": [0.1, 1, 10],
-                "kernel__k2__length_scale": [0.1, 1, 10]
-            }
-            grid_search = GridSearchCV(gp, param_grid, cv=3, scoring='neg_mean_squared_error')
-            grid_search.fit(X_scaled, y)
-            best_model = grid_search.best_estimator_
-
-            # Forecast
-            future_X = np.arange(n, n + periods).reshape(-1, 1)
-            future_X_scaled = scaler.transform(future_X)
-            forecast, _ = best_model.predict(future_X_scaled, return_std=True)
-            forecast = np.maximum(forecast, 0)
-
-            # Calculate metrics
-            predicted, _ = best_model.predict(X_scaled, return_std=True)
-            metrics = ForecastingEngine.calculate_metrics(y, predicted)
-
-        except Exception as e:
-            print(f"Error in Gaussian Process forecasting: {e}")
-            return ForecastingEngine.linear_regression_forecast(data, periods)
-
-        return forecast, metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced Gaussian Process with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_gaussian_process_forecast(data, periods, optimizer)
 
     @staticmethod
     def neural_network_forecast(data: pd.DataFrame, periods: int, hidden_layer_sizes_list: list = [(10,), (20, 10)], alpha_list: list = [0.001, 0.01]) -> tuple:
-        """Multi-layer Perceptron Neural Network forecasting with hyperparameter tuning and external factors"""
+        """Enhanced Multi-layer Perceptron Neural Network forecasting with comprehensive hyperparameter optimization"""
         y = data['quantity'].values
         n = len(y)
-        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
-        if n < 6:
+        
+        if n < 15:
             return ForecastingEngine.linear_regression_forecast(data, periods)
-        window = min(5, n - 1)
-        X, y_target = [], []
-        for i in range(window, n):
-            lags = list(y[i-window:i])
-            trend = i / n
-            seasonal = np.sin(2 * np.pi * i / 12)
-            features = lags + [trend, seasonal]
-            if external_factor_cols:
-                features.extend(data[external_factor_cols].iloc[i].values)
-            X.append(features)
-            y_target.append(y[i])
-        if len(X) < 3:
-            return ForecastingEngine.linear_regression_forecast(data, periods)
-        X = np.array(X)
-        y_target = np.array(y_target)
-        param_grid = {'hidden_layer_sizes': hidden_layer_sizes_list, 'alpha': alpha_list}
-        model = MLPRegressor(activation='relu', solver='adam', max_iter=1000, random_state=42)
-        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
-        grid_search.fit(X, y_target)
-        best_model = grid_search.best_estimator_
-        print(f"Best MLP params: {grid_search.best_params_}")
-        forecast = []
-        recent_values = list(y[-window:])
-        for i in range(periods):
-            trend = (n + i) / n
-            seasonal = np.sin(2 * np.pi * (n + i) / 12)
-            features = recent_values + [trend, seasonal]
-            if external_factor_cols:
-                features.extend(data[external_factor_cols].iloc[-1].values)
-            next_pred = best_model.predict([features])[0]
-            next_pred = max(0, next_pred)
-            forecast.append(next_pred)
-            recent_values = recent_values[1:] + [next_pred]
-        predicted = best_model.predict(X)
-        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
-        return np.array(forecast), metrics
+        
+        # Analyze data characteristics for adaptive optimization
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        # Create adaptive optimizer
+        n_splits = AdaptiveParameterSelector.get_adaptive_cv_splits(n)
+        test_size = AdaptiveParameterSelector.get_adaptive_test_size(n, periods)
+        optimizer = TimeSeriesOptimizer(n_splits=n_splits, test_size=test_size)
+        
+        # Use enhanced Neural Network with comprehensive optimization
+        return EnhancedForecastingEngine.enhanced_neural_network_forecast(data, periods, optimizer)
 
     @staticmethod
     def prophet_like_forecast(data: pd.DataFrame, periods: int) -> tuple:
-        """Actual Facebook Prophet forecasting with external regressors support"""
+        """Enhanced Facebook Prophet forecasting with Bayesian hyperparameter optimization"""
         from prophet import Prophet
         import warnings
 
         y = data['quantity'].values
         dates = pd.to_datetime(data['date'])
         external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
+        n = len(y)
 
-        if len(y) < 10:  # Need sufficient data for Prophet
+        if n < 10:
             return ForecastingEngine.linear_regression_forecast(data, periods)
 
         try:
             with warnings.catch_warnings():
                 warnings.filterwarnings("ignore")
 
+                # Get optimized Prophet parameters
+                exog = data[external_factor_cols].values if external_factor_cols else None
+                best_params = BayesianOptimizer.optimize_prophet_parameters(y, exog)
+                print(f"Optimized Prophet parameters: {best_params}")
+
                 # Prepare data for Prophet (requires 'ds' and 'y' columns)
                 prophet_data = pd.DataFrame({
                     'ds': dates,
                     'y': y
                 })
 
                 # Add external regressors if available
                 if external_factor_cols:
                     print(f"Running Prophet with external regressors: {external_factor_cols}")
                     for col in external_factor_cols:
                         prophet_data[col] = data[col].values
 
-                # Initialize Prophet model
-                model = Prophet(
-                    yearly_seasonality=True if len(y) >= 24 else False,
-                    weekly_seasonality=False,  # Assuming monthly data
-                    daily_seasonality=False,
-                    seasonality_mode='multiplicative',
-                    changepoint_prior_scale=0.05,
-                    seasonality_prior_scale=10.0,
-                    interval_width=0.8
-                )
+                # Initialize Prophet model with optimized parameters
+                model = Prophet(**best_params)
 
                 # Add external regressors to the model
                 if external_factor_cols:
                     for col in external_factor_cols:
                         model.add_regressor(col)
 
                 # Fit the model
                 model.fit(prophet_data)
 
                 # Create future dataframe
                 future = model.make_future_dataframe(periods=periods, freq='MS')  # Monthly start
 
                 # Add external regressor values for future periods
                 if external_factor_cols:
                     # Forecast external factors properly
-                    future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
+                    future_factors = EnhancedForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
 
                     for col in external_factor_cols:
                         # Fill historical values
                         future[col] = np.nan
                         future.loc[:len(prophet_data)-1, col] = prophet_data[col].values
                         # Use forecasted values for future periods
                         future.loc[len(prophet_data):, col] = future_factors[col]
 
                 # Make predictions
                 forecast_df = model.predict(future)
 
                 # Extract forecast values
                 forecast = forecast_df['yhat'].iloc[-periods:].values
                 forecast = np.maximum(forecast, 0)
 
                 # Calculate metrics using fitted values
                 fitted = forecast_df['yhat'].iloc[:len(y)].values
                 metrics = ForecastingEngine.calculate_metrics(y, fitted)
 
-                print(f"Prophet model fitted successfully with {len(external_factor_cols)} external regressors")
+                print(f"Enhanced Prophet model fitted successfully with optimized parameters")
                 return forecast, metrics
 
         except Exception as e:
             print(f"Prophet forecasting failed: {e}")
-
-            # Fallback to simplified Prophet without external regressors
-            try:
-                print("Trying Prophet without external regressors...")
-
-                prophet_data = pd.DataFrame({
-                    'ds': dates,
-                    'y': y
-                })
-
-                model = Prophet(
-                    yearly_seasonality=True if len(y) >= 24 else False,
-                    weekly_seasonality=False,
-                    daily_seasonality=False,
-                    seasonality_mode='additive',
-                    changepoint_prior_scale=0.05
-                )
-
-                model.fit(prophet_data)
-                future = model.make_future_dataframe(periods=periods, freq='MS')
-                forecast_df = model.predict(future)
-
-                forecast = forecast_df['yhat'].iloc[-periods:].values
-                forecast = np.maximum(forecast, 0)
-
-                fitted = forecast_df['yhat'].iloc[:len(y)].values
-                metrics = ForecastingEngine.calculate_metrics(y, fitted)
-
-                print("Prophet model fitted successfully without external regressors")
-                return forecast, metrics
-
-            except Exception as e2:
-                print(f"Prophet fallback also failed: {e2}")
-
-        # Final fallback to linear regression
-        print("Prophet failed, falling back to linear regression")
-        return ForecastingEngine.linear_regression_forecast(data, periods)
+            return ForecastingEngine.exponential_smoothing_forecast(data, periods)
 
     @staticmethod
     def run_algorithm(algorithm: str, data: pd.DataFrame, config: ForecastConfig, save_model: bool = True) -> AlgorithmResult:
-        """Run a specific forecasting algorithm"""
+        """Run a specific forecasting algorithm with enhanced optimization"""
         from database import SessionLocal
         db = SessionLocal()
         try:
             # Print first 5 rows of data fed to algorithm
             print(f"\nData fed to algorithm '{algorithm}':")
             print(data.head(5))
+            
+            # Analyze data characteristics for adaptive algorithm selection
+            y = data['quantity'].values
+            characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+            print(f"Data characteristics: length={characteristics['length']}, "
+                  f"trend_strength={characteristics['trend_strength']:.3f}, "
+                  f"seasonality_strength={characteristics['seasonality_strength']:.3f}, "
+                  f"volatility={characteristics['volatility']:.3f}")
 
             # Check for cached model
             training_data = data['quantity'].values
             if training_data is not None and len(training_data) > 0:
                 model_hash = ModelPersistenceManager.find_cached_model(db, algorithm, config.dict(), training_data)
             else:
                 model_hash = None
 
             if model_hash:
                 print(f"Using cached model for {algorithm}")
                 cached_model = ModelPersistenceManager.load_model(db, model_hash)
                 if cached_model:
                     # Use cached model for prediction
                     # Note: This is a simplified example - in practice, you'd need to adapt this
                     # based on the specific algorithm and model type
                     pass
 
             # Time-based train/test split for realistic metrics
             train, test = ForecastingEngine.time_based_split(data, test_ratio=0.2)
 
             # Train model on train set
             model = None
             if algorithm == "linear_regression":
                 forecast, metrics = ForecastingEngine.linear_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = LinearRegression().fit(np.arange(len(train)).reshape(-1, 1), train['quantity'].values)
             elif algorithm == "polynomial_regression":
                 forecast, metrics = ForecastingEngine.polynomial_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "exponential_smoothing":
                 forecast, metrics = ForecastingEngine.exponential_smoothing_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "holt_winters":
                 forecast, metrics = ForecastingEngine.holt_winters_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "arima":
                 forecast, metrics = ForecastingEngine.arima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "random_forest":
                 forecast, metrics = ForecastingEngine.random_forest_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "seasonal_decomposition":
                 forecast, metrics = ForecastingEngine.seasonal_decomposition_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "moving_average":
                 forecast, metrics = ForecastingEngine.moving_average_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "sarima":
                 forecast, metrics = ForecastingEngine.sarima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "prophet_like":
                 forecast, metrics = ForecastingEngine.prophet_like_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "lstm_like":
                 forecast, metrics = ForecastingEngine.lstm_simple_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "xgboost":
                 forecast, metrics = ForecastingEngine.xgboost_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "svr":
                 forecast, metrics = ForecastingEngine.svr_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "knn":
                 forecast, metrics = ForecastingEngine.knn_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "gaussian_process":
                 forecast, metrics = ForecastingEngine.gaussian_process_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "neural_network":
                 forecast, metrics = ForecastingEngine.neural_network_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "theta_method":
                 forecast, metrics = ForecastingEngine.theta_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "croston":
                 forecast, metrics = ForecastingEngine.croston_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "ses":
                 forecast, metrics = ForecastingEngine.ses_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "damped_trend":
                 forecast, metrics = ForecastingEngine.damped_trend_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "naive_seasonal":
                 forecast, metrics = ForecastingEngine.naive_seasonal_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             elif algorithm == "drift_method":
                 forecast, metrics = ForecastingEngine.drift_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                 model = None  # No explicit model object to save
             else:
                 raise ValueError(f"Unknown algorithm: {algorithm}")
 
             # Save model/configuration to cache
             if ENABLE_MODEL_CACHE and save_model:
                 try:
                     print(f"Debug run_algorithm: Saving model/configuration for algorithm {algorithm}")
                     ModelPersistenceManager.save_model(
                         db, model, algorithm, config.dict(), 
                         training_data, metrics, {'data_shape': training_data.shape}
                     )
                     print(f"Debug run_algorithm: Model/configuration saved successfully for algorithm {algorithm}")
                 except Exception as e:
                     print(f"Failed to save model/configuration to cache: {e}")
 
             # Compute test metrics
             if test is not None and len(test) > 0:
                 actual = test['quantity'].values
                 predicted = forecast[:len(test)]
                 metrics = ForecastingEngine.calculate_metrics(actual, predicted)
             else:
                 # Fallback to training metrics
                 y = train['quantity'].values
                 x = np.arange(len(y)).reshape(-1, 1)
                 if algorithm == "linear_regression":
                     model = LinearRegression().fit(x, y)
                     predicted = model.predict(x)
                 elif algorithm == "polynomial_regression":
                     coeffs = np.polyfit(np.arange(len(y)), y, 2)
                     poly_func = np.poly1d(coeffs)
                     predicted = poly_func(np.arange(len(y)))
                 elif algorithm == "exponential_smoothing" or algorithm == "ses":
                     # Use simple exponential smoothing for fallback
                     alpha = 0.3
                     smoothed = [y[0]]
                     for i in range(1, len(y)):
                         smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])
                     predicted = smoothed
                 else:
                     predicted = y
                 metrics = ForecastingEngine.calculate_metrics(y, predicted)
 
             # Prepare output
             last_date = data['date'].iloc[-1]
             forecast_dates = ForecastingEngine.generate_forecast_dates(last_date, config.forecastPeriod, config.interval)
 
             historic_data = []
             historic_subset = data.tail(config.historicPeriod)
             for _, row in historic_subset.iterrows():
                 historic_data.append(DataPoint(
                     date=row['date'].strftime('%Y-%m-%d'),
                     quantity=float(row['quantity']),
                     period=row['period']
                 ))
 
             forecast_data = []
             for i, (date, quantity) in enumerate(zip(forecast_dates, forecast)):
                 forecast_data.append(DataPoint(
                     date=date.strftime('%Y-%m-%d'),
                     quantity=float(quantity),
                     period=ForecastingEngine.format_period(date, config.interval)
                 ))
 
             trend = ForecastingEngine.calculate_trend(data['quantity'].values)
 
             return AlgorithmResult(
                 algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                 accuracy=round(metrics['accuracy'], 1),
                 mae=round(metrics['mae'], 2),
                 rmse=round(metrics['rmse'], 2),
                 historicData=historic_data,
                 forecastData=forecast_data,
                 trend=trend
             )
         except Exception as e:
             print(f"Error in {algorithm}: {str(e)}")
             return AlgorithmResult(
                 algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                 accuracy=0.0,
                 mae=999.0,
                 rmse=999.0,
                 historicData=[],
                 forecastData=[],
                 trend='stable'
             )
         finally:
             db.close()
 
     @staticmethod
     def generate_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> ForecastResult:
-        """Generate forecast using data from database"""
+        """Generate forecast using data from database with enhanced optimization"""
         if process_log is not None:
             process_log.append("Loading data from database...")
+            
+        # Analyze data characteristics first for adaptive algorithm selection
+        df = ForecastingEngine.load_data_from_db(db, config)
+        aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)
+        
+        if len(aggregated_df) < 2:
+            raise ValueError("Insufficient data for forecasting")
+            
+        y = aggregated_df['quantity'].values
+        characteristics = AdaptiveParameterSelector.analyze_data_characteristics(y)
+        
+        if process_log is not None:
+            process_log.append(f"Data characteristics analyzed: trend={characteristics['trend_strength']:.3f}, "
+                             f"seasonality={characteristics['seasonality_strength']:.3f}, "
+                             f"volatility={characteristics['volatility']:.3f}")
 
-        df = ForecastingEngine.load_data_from_db(db, config)
-
-        if process_log is not None:
-            process_log.append(f"Data loaded: {len(df)} records")
-            process_log.append("Aggregating data by period...")
-
-        aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)
-
-        if process_log is not None:
-            process_log.append(f"Data aggregated: {len(aggregated_df)} records")
-
-        if len(aggregated_df) < 2:
-            raise ValueError("Insufficient data for forecasting")
+        if process_log is not None:
+            process_log.append(f"Data loaded: {len(df)} records")
+            process_log.append(f"Data aggregated: {len(aggregated_df)} records")
 
         if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
             if process_log is not None:
                 if config.algorithm == "best_fit":
-                    process_log.append("Running best fit algorithm selection...")
+                    process_log.append("Running adaptive best fit algorithm selection...")
                 elif config.algorithm == "best_statistical":
-                    process_log.append("Running best statistical method selection...")
+                    process_log.append("Running adaptive best statistical method selection...")
                 elif config.algorithm == "best_ml":
-                    process_log.append("Running best machine learning method selection...")
+                    process_log.append("Running adaptive best machine learning method selection...")
                 elif config.algorithm == "best_specialized":
-                    process_log.append("Running best specialized method selection...")
+                    process_log.append("Running adaptive best specialized method selection...")
 
-            # Define algorithm categories
+            # Get adaptive algorithm priorities based on data characteristics
+            prioritized_algorithms = AdaptiveParameterSelector.select_algorithm_priorities(characteristics)
+            
+            # Define algorithm categories with adaptive selection
             statistical_algorithms = [
                 "linear_regression", "polynomial_regression", "exponential_smoothing", 
                 "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                 "theta_method", "drift_method", "naive_seasonal", "prophet_like"
             ]
 
             ml_algorithms = [
                 "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                 "neural_network", "lstm_like"
             ]
 
             specialized_algorithms = [
                 "seasonal_decomposition", "moving_average", "croston"
             ]
 
             # Select algorithms based on category and prioritize based on data characteristics
             if config.algorithm == "best_statistical":
-                algorithms = statistical_algorithms
+                algorithms = [alg for alg in prioritized_algorithms if alg in statistical_algorithms]
             elif config.algorithm == "best_ml":
-                algorithms = ml_algorithms
+                algorithms = [alg for alg in prioritized_algorithms if alg in ml_algorithms]
             elif config.algorithm == "best_specialized":
-                algorithms = specialized_algorithms
+                algorithms = [alg for alg in prioritized_algorithms if alg in specialized_algorithms]
             else:  # best_fit
-                algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
-                            if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]
+                # Use prioritized algorithms but limit to top performers for efficiency
+                algorithms = prioritized_algorithms[:12]  # Limit to top 12 for performance
+                
+            if process_log is not None:
+                process_log.append(f"Algorithm priority order: {algorithms[:5]}...")  # Show top 5
 
             algorithm_results = []
             best_model = None
             best_algorithm = None
             best_metrics = None
 
             # Use ThreadPoolExecutor for parallel execution
             max_workers = min(len(algorithms), os.cpu_count() or 4)
             if process_log is not None:
                 process_log.append(f"Starting parallel execution with {max_workers} workers for {len(algorithms)} algorithms...")
 
             with ThreadPoolExecutor(max_workers=max_workers) as executor:
                 # Submit all algorithm tasks
                 future_to_algorithm = {
                     executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, config, save_model=False): algorithm
                     for algorithm in algorithms
                 }
 
                 # Collect results as they complete
                 for future in as_completed(future_to_algorithm):
                     algorithm_name = future_to_algorithm[future]
                     try:
                         result = future.result()
                         algorithm_results.append(result)
                         if process_log is not None:
                             process_log.append(f"✅ Algorithm {algorithm_name} completed with accuracy: {result.accuracy:.2f}%")
 
                         # Track best performing algorithm
                         if best_metrics is None or result.accuracy > best_metrics['accuracy']:
                             best_metrics = {
                                 'accuracy': result.accuracy,
                                 'mae': result.mae,
                                 'rmse': result.rmse
                             }
                             best_model = result
                             best_algorithm = algorithm_name
 
                     except Exception as exc:
                         if process_log is not None:
                             process_log.append(f"❌ Algorithm {algorithm_name} failed: {str(exc)}")
                         # Create a dummy result for failed algorithms to maintain structure
                         algorithm_results.append(AlgorithmResult(
                             algorithm=ForecastingEngine.ALGORITHMS[algorithm_name],
                             accuracy=0.0,
                             mae=999.0,
                             rmse=999.0,
                             historicData=[],
                             forecastData=[],
                             trend='stable'
                         ))
 
             # Filter out failed results for ensemble calculation
             successful_results = [res for res in algorithm_results if res.accuracy > 0]
             if not successful_results:
                 raise ValueError("All algorithms failed to produce valid results")
 
             if process_log is not None:
                 process_log.append(f"Parallel execution completed. {len(successful_results)} algorithms succeeded, {len(algorithm_results) - len(successful_results)} failed.")
 
             if not algorithm_results:
                 raise ValueError("No algorithms produced valid results")
 
             # Ensemble: average forecast of top 3 algorithms by accuracy
             top3 = sorted(successful_results, key=lambda x: -x.accuracy)[:3]
             if len(top3) >= 2:
                 if process_log is not None:
                     process_log.append(f"Creating ensemble from top {len(top3)} algorithms...")
 
                 n_forecast = len(top3[0].forecastData) if top3[0].forecastData else 0
                 avg_forecast = []
                 for i in range(n_forecast):
                     quantities = [algo.forecastData[i].quantity for algo in top3 if algo.forecastData]
                     if not quantities:
                         avg_qty = 0
                     else:
                         avg_qty = np.mean(quantities)
                     avg_forecast.append(DataPoint(
                         date=top3[0].forecastData[i].date,
                         quantity=avg_qty,
                         period=top3[0].forecastData[i].period
                     ))
 
                 ensemble_result = AlgorithmResult(
                     algorithm="Ensemble (Top 3 Avg)",
                     accuracy=np.mean([algo.accuracy for algo in top3]),
                     mae=np.mean([algo.mae for algo in top3]),
                     rmse=np.mean([algo.rmse for algo in top3]),
                     historicData=top3[0].historicData,
                     forecastData=avg_forecast,
                     trend=top3[0].trend
                 )
                 algorithm_results.append(ensemble_result)
 
             # Save the best_fit model configuration
             try:
                 if process_log is not None:
                     process_log.append(f"Saving best fit model configuration (best algorithm: {best_algorithm})...")
 
                 ModelPersistenceManager.save_model(
                     db, None, "best_fit", config.dict(),
                     aggregated_df['quantity'].values, 
                     {'accuracy': ensemble_result.accuracy, 'mae': ensemble_result.mae, 'rmse': ensemble_result.rmse},
                     {'data_shape': aggregated_df.shape, 'best_algorithm': best_algorithm}
                 )
             except Exception as e:
                 print(f"Failed to save best_fit model: {e}")
                 if process_log is not None:
                     process_log.append(f"Warning: Failed to save best_fit model: {e}")
 
             # Generate config hash for tracking
             config_hash = ModelPersistenceManager.generate_config_hash(config.dict())
 
             return ForecastResult(
                 selectedAlgorithm=f"{best_model.algorithm} (Best Fit)",
                 accuracy=best_model.accuracy,
                 mae=best_model.mae,
                 rmse=best_model.rmse,
                 historicData=best_model.historicData,
                 forecastData=best_model.forecastData,
                 trend=best_model.trend,
                 allAlgorithms=algorithm_results,
                 configHash=config_hash,
                 processLog=process_log
             )
         else:
             if process_log is not None:
                 process_log.append(f"Running algorithm: {config.algorithm}")
 
             result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config, save_model=True)
             config_hash = ModelPersistenceManager.generate_config_hash(config.dict())
             return ForecastResult(
                 selectedAlgorithm=result.algorithm,
                 combination=None,
                 accuracy=result.accuracy,
                 mae=result.mae,
                 rmse=result.rmse,
                 historicData=result.historicData,
                 forecastData=result.forecastData,
                 trend=result.trend,
                 configHash=config_hash,
                 processLog=process_log
             )