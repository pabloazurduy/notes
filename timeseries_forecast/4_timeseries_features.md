
*based on  [Forecast Principles and Practices][1]*
*created on: 2024-11-23 18:06:55*
## Time Sereis Features

here most of the features comes from the package `feats` that automatically estimates all this features for time series. Nevertheless, we will present some of the features presented in the package so it can be calculated without the package. or to explain them if needed to be use on other model types (or to identify the TS in a better way).

### ACF features 

1. `ACF(k)` : Autocorrelation at lag k
2. `SUM_ACF10` : sum of the first squared 10 autocorrelations. this is sometimes a good indicator of the total autocorrelation of a time series.
2. `diff_ACF(k)` : Autocorrelation of the differenced series ($h_t =  y_t-y_{t-1}$) of lag k.

### STL features

1. `seasonal_strength` : 
    The strength of the seasonal component is    defined as the proportion of the variance of the seasonal component to the total variance of the time series.. So we define the strength of trend as:
    $$
    F_T = \max\left(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t+R_t)}\right).
    $$
    This will give a measure of the strength of the trend between 0 and 1, the closer to 1 the stronger the trend (or explains more of the variance of the time series).
1. `seasonal_streght`:
    The strength of seasonality is defined similarly, but with respect to the detrended data rather than the seasonally adjusted data:
    $$
    F_S = \max\left(0, 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_{t}+R_t)}\right).
    $$

1. `seasonal_peak_year`: indicates the timing of the peaks — which month or quarter contains the largest seasonal component. This tells us something about the nature of the seasonality. 
1. `seasonal_trough_year` indicates the timing of the troughs — which month or quarter contains the smallest seasonal component.
1. `spikiness` measures the prevalence of spikes in the remainder component $R_t$ of the STL decomposition. It is the variance of the leave-one-out variances of $R_t$.
1. `linearity` measures the linearity of the trend component of the STL decomposition. It is based on the coefficient of a linear regression applied to the trend component.
1. `curvature` measures the curvature of the trend component of the STL decomposition. It is based on the coefficient from an orthogonal quadratic regression applied to the trend component.
1. `stl_e_acf1` is the first autocorrelation coefficient of the remainder series.
1. `stl_e_acf10` is the sum of squares of the first ten autocorrelation coefficients of the remainder series.


### Other Features:

1. `box_pierce` gives the Box-Pierce statistic for testing if a time series is white noise, and the corresponding p-value. This test is discussed in Section
1. `guerrero` computes the optimal $\lambda$ value for a Box-Cox transformation using the Guerrero method




[//]: <> (References)
[1]: <https://otexts.com/fpp3/features.html>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)