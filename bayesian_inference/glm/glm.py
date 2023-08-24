import numpy as np
import pymc as pm
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az


np.random.seed(0)
n_samples = 1000
education_years = np.random.normal(10, 2, (n_samples,2))
b1_real = np.array([-1,5])
b0_real = np.array([5_000, 5])
income =  b0_real  + b1_real*education_years + stats.halfnorm.rvs(0, 20,(n_samples,2))


with pm.Model() as model_city:
    b0 = pm.Normal("b0", mu=0, sigma=10, shape=2)
    b1 = pm.Normal("b1", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=10, shape=2)
    
    mu = b0 + b1 * education_years
    income = pm.Normal("income", mu=mu, sigma=sigma, observed=income, shape=income.shape)

    trace = pm.sample(draws=1000, cores=4, target_accept=0.99)
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['b0', 'b1', 'income', 'sigma'])
    map = pm.find_MAP()

az.plot_forest(trace, combined=True, hdi_prob=0.95)



# plot and add linear trend 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, ax in enumerate(axs):
    ax.scatter(education_years[:, i], income[:, i], s=2)
    ax.set_xlabel('Education Years')
    ax.set_ylabel('Income')
    ax.set_title (f'City {i}')

    # add linear regression trend line
    x = education_years[:, i]
    y = income[:, i]
    b1, b0, r_value, p_value, std_err = stats.linregress(x, y)
    ax.plot(x, b0 + b1*x, 'r', label=f'fitted line {b0=:.2} {b1=:.2}')
    ax.legend()
plt.show()
