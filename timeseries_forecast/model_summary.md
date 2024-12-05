
# Time Series Models Summary

## 1. Simple Forecasting Methods

### Mean Method
- **Description**: All forecasts equal to mean of historical data
- **Pros**:
  - Simple baseline
  - Works for stationary data without trend/seasonality
- **Cons**:
  - Ignores trends and seasonality
  - Poor for non-stationary data

### Naïve Method 
- **Description**: All forecasts equal to last observed value
- **Pros**:
  - Simple baseline
  - Good for random walk processes
- **Cons**:
  - Ignores patterns and seasonality
  - Sensitive to outliers

### Seasonal Naïve
- **Description**: Forecast equals last observed value from same season
- **Pros**:
  - Captures seasonal patterns
  - Simple to implement
- **Cons**:
  - Ignores trends
  - Requires full season of history

### Drift Method
- **Description**: Linear extrapolation between first and last observation
- **Pros**:
  - Captures linear trends
  - Simple to implement
- **Cons**:
  - Only handles linear trends
  - Sensitive to outliers at endpoints

## 2. Exponential Smoothing Models

### Simple Exponential Smoothing (ETS(A,N,N))
- **Description**: Weighted average with exponentially decreasing weights
- **Pros**:
  - Handles level changes
  - Single parameter (α)
- **Cons**:
  - No trend or seasonality
  - Requires stationary data

### Holt's Linear Method (ETS(A,A,N))
- **Description**: Adds trend component to SES
- **Parameters**: α (level), β (trend)
- **Pros**:
  - Handles both level and trend
- **Cons**:
  - Can overforecast long-term
  - No seasonality

### Holt-Winters (ETS(A,A,A))
- **Description**: Adds seasonal component to Holt's
- **Parameters**: α (level), β (trend), γ (seasonal)
- **Pros**:
  - Handles level, trend and seasonality
  - Multiplicative or additive seasonality
- **Cons**:
  - Many parameters to estimate
  - Can produce negative forecasts with additive model

### Damped Trend (ETS(A,Ad,N))
- **Description**: Dampens trend to flatten over time
- **Parameters**: α (level), β (trend), φ (damping)
- **Pros**:
  - More conservative long-term forecasts
  - Often more accurate than Holt's
- **Cons**:
  - Additional parameter to estimate
  - No seasonality

## 3. ARIMA Models
- **Description**: Combines autoregression, differencing and moving average
- **Parameters**: p (AR order), d (differencing), q (MA order)
- **Pros**:
  - Flexible for many time series patterns
  - Handles non-stationarity through differencing
- **Cons**:
  - Model selection can be complex
  - Requires longer time series
  - Less interpretable than ETS

## 4. Dynamic Regression
- **Description**: Regression with ARIMA errors
- **Pros**:
  - Incorporates external predictors
  - Handles autocorrelated errors
- **Cons**:
  - Requires predictor forecasts
  - More complex estimation

## 5. Complex Seasonality Models

### Multiple Seasonal Decomposition (STL)
- **Description**: Decomposition with multiple seasonal periods
- **Pros**:
  - Handles multiple seasonal patterns
  - Robust to outliers
  - Flexible seasonal periods
- **Cons**:
  - Purely additive decomposition
  - No probabilistic forecasts

### Dynamic Harmonic Regression
- **Description**: Fourier terms with ARIMA errors
- **Pros**:
  - Handles multiple seasonality
  - Flexible seasonal patterns
- **Cons**:
  - Many parameters with high frequency data
  - Requires selection of Fourier terms

## Selection Guidelines

1. For simple, stationary data:
   - Start with simple methods
   - Use SES if no clear patterns

2. For data with trend:
   - Use Holt's or damped trend
   - Prefer damped trend for longer horizons

3. For seasonal data:
   - Use Holt-Winters or seasonal ETS
   - STL for complex seasonality

4. With external predictors:
   - Use dynamic regression
   - Consider leading indicators

5. For high-frequency data:
   - STL decomposition
   - Dynamic harmonic regression

6. For hierarchical data:
   - Consider reconciliation approaches
   - Bottom-up or optimal combination

## Key Considerations
1. Forecast horizon length
2. Data frequency and seasonality
3. Available history length
4. Required interpretability
5. Computational requirements
6. Availability of external predictors