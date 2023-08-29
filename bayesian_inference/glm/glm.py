import numpy as np
import pymc as pm
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr


np.random.seed(0)
n_samples = 1000
education_years = np.random.normal(10, 2, (n_samples,2))
b1_real = np.array([-1,5])
b0_real = np.array([5_000, 5])
y =  b0_real  + b1_real*education_years + stats.halfnorm.rvs(0, 20,(n_samples,2)) #income 


with pm.Model() as model_city:
    b0 = pm.Normal("b0", mu=0, sigma=10_000, shape=2)
    b1 = pm.Normal("b1", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=10, shape=2)
    
    mu = pm.Deterministic('mu', b0 + b1 * education_years)
    inc_mu = pm.Normal("inc_mu", mu=mu, sigma=sigma, observed=y, shape=y.shape)

    trace = pm.sample(draws=500, cores=4, tune=400, return_inferencedata=True)
    post_idata = pm.sample_posterior_predictive(trace) # extend_inferencedata=True predictions=True, return_inferencedata=True
    map = pm.find_MAP()

az.plot_forest(trace, combined=True, hdi_prob=0.95, var_names=['sigma','b1','b0'])
posterior_pred = az.extract(post_idata, group='posterior_predictive', num_samples=100) 
posterior      = az.extract(trace, group='posterior', num_samples=100) 
y_pred = posterior_pred['inc_mu']
y_trend = posterior['mu']

# plot and add linear trend 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, ax in enumerate(axs):
    ax.scatter(education_years[:, i], y[:, i], s=2)
    ax.set_xlabel('Education Years')
    ax.set_ylabel('income')
    ax.set_title (f'City {i}')

    # add linear regression trend line
    x = education_years[:, i]
    b1, b0, r_value, p_value, std_err = stats.linregress(x, y[:, i])
    ax.plot(x, b0 + b1*x, 'r', label=f'fitted line {b0=:.2} {b1=:.2}')
    ax.plot(x, map['b0'][i] + map['b1'][i]*x, label=f'map_value {map["b0"][i]=:.2} {map["b1"][i]=:.2}', color='green')
    ax.legend()
    
    ax.scatter(x=x,y=y_trend.mean(axis=2)[:,i], color='lightblue', label='prediction_mean', s=1) 
    p05 = np.percentile(y_pred[:, i, :], 5, axis=1)
    p95 = np.percentile(y_pred[:, i, :], 95, axis=1)
    f_p05 = interp1d(x, p05, kind='linear')
    f_p95 = interp1d(x, p95, kind='linear')
    xnew = np.linspace(min(x), max(x), 500)
    ax.fill_between(xnew, f_p05(xnew), f_p95(xnew), alpha=0.2, color='gray', interpolate=True)
plt.show()

