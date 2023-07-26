import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

plt.pause(1) # fix plots 

# Set up the pymc3 model. Again assume Uniform priors for p_A and p_B.
#these two quantities are unknown to us.
true_p_A = 0.05
true_p_B = 0.04

#notice the unequal sample sizes -- no problem in Bayesian analysis.
for i in range(1,10):
    N_A = 15 *i**3
    N_B = 7 * i**3 #500

    #generate some observations
    observations_A = stats.bernoulli.rvs(true_p_A, size=N_A)
    observations_B = stats.bernoulli.rvs(true_p_B, size=N_B)

    with pm.Model() as model:
        p_A = pm.Uniform("p_A", 0, 1)
        p_B = pm.Uniform("p_B", 0, 1)
        
        # Define the deterministic delta function. This is our unknown of interest.
        delta = pm.Deterministic("delta", p_A - p_B)
        
        # Set of observations, in this case we have two observation datasets.
        obs_A = pm.Bernoulli("obs_A", p_A, observed=observations_A)
        obs_B = pm.Bernoulli("obs_B", p_B, observed=observations_B)

        # To be explained in chapter 3.
        # step = pm.Metropolis()
        step = pm.NUTS()
        burned_trace = pm.sample(10000, step=step, cores=1, chains=1, progressbar=False)
    az.plot_trace(burned_trace, combined=True)
    delta_samples = burned_trace.posterior["delta"].values[0]
    
    print(f'{N_A = } {N_B = }')
    print("Probability site A is WORSE than site B: %.3f" % \
        np.mean(delta_samples < 0))
    print("Probability site A is BETTER than site B: %.3f" % \
        np.mean(delta_samples > 0))
