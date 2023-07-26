import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

plt.pause(1)  # fix plots

# ======================= #
# Simple model declaration
# ======================= #
with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0, initval=0.5)  # initval = testval
    data_generator = pm.Poisson("data_generator", parameter)
    data_plus_one = data_generator + 1
    # idata = pm.sample(1000, tune=1000, cores=1)

# plot results
# az.plot_trace(idata, combined=True)
# az.plot_posterior(idata, rope=(0,0.5))
# az.summary(idata, round_to=2)

with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)
    new_deterministic_variable = lambda_1 + lambda_2
    # idata = pm.sample(1000, tune=1000, cores=1)

# =============================== #
# Deterministic tracking activate
# =============================== #
def subtract(x, y):
    return x - y


with pm.Model() as model:
    stochastic_1 = pm.Uniform("U_1", 0, 1)
    stochastic_2 = pm.Uniform("U_2", 0, 1)
    det_1 = pm.Deterministic("Delta", subtract(stochastic_1, stochastic_2))
    idata = pm.sample(1000, tune=1000, cores=1)
az.plot_trace(idata, combined=True)

# ===================== #
# use of aesara tensor
# ===================== #
import aesara.tensor as at

with pm.Model() as theano_test:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = at.stack([p1, p2])
    assignment = pm.Categorical("assignment", p)  # pair of probabilities for our two categories p=(p1,p2)
    # idata = pm.sample(1000, tune=1000, cores=1)
# az.plot_trace(idata, combined=True)

# We're using some fake data here
import numpy as np

data = np.array([10, 25, 15, 20, 35])
with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_3", 1.0)
    lambda_2 = pm.Exponential("lambda_4", 1.0)
    idx = np.arange(len(data))  # Index
    tau = pm.DiscreteUniform("tau2", lower=0, upper=len(data) - 1)
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
    obs = pm.Poisson("obs", lambda_, observed=data)
    # idata = pm.sample(1000, tune=1000, cores=1)
# az.plot_trace(idata, combined=True)

# ============================== #
# ======= AB TESTING =========== #
# ============================== #
# The parameters are the bounds of the Uniform.
from scipy import stats

# set constants
p_true = 0.05  # remember, this is unknown.
N = 1500
# sample N Bernoulli random variables from Ber(0.05).
# each random variable has a 0.05 chance of being a 1.
# this is the data-generation step
occurrences = stats.bernoulli.rvs(p_true, size=N)

print(occurrences)  # Remember: Python treats True == 1, and False == 0
print(np.sum(occurrences))
# Occurrences.mean is equal to n/N.
print("What is the observed frequency in Group A? %.4f" % np.mean(occurrences))
print("Does this equal the true frequency? %s" % (np.mean(occurrences) == p_true))

# include the observations, which are Bernoulli
with pm.Model() as model:
    p = pm.Uniform("p", lower=0, upper=1)
    obs = pm.Bernoulli("obs", p, observed=occurrences)
    # To be explained in chapter 3
    # step = pm.Metropolis()
    # trace = pm.sample(18000, step=step, cores=1, chains=1)
    # burned_trace = trace.posterior.p.values[0][1000:] # select first chain the first 1000 values
# az.plot_trace(trace, combined=True)
from matplotlib import pyplot as plt

plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
# plt.hist(burned_trace, bins=25, histtype="stepfilled", density=True)
plt.legend()

# Set up the pymc3 model. Again assume Uniform priors for p_A and p_B.
# these two quantities are unknown to us.
true_p_A = 0.05
true_p_B = 0.04

# notice the unequal sample sizes -- no problem in Bayesian analysis.
for i in range(1, 10):
    N_A = 15 * i**3
    N_B = 7 * i**3  # 500

    # generate some observations
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
        step = pm.Metropolis()
        # burned_trace = pm.sample(10000, step=step, cores=1, chains=1, progressbar=False)
    # delta_samples = burned_trace.posterior["delta"].values[0]
    # print(f'{N_A = } {N_B = }')
    # print("Probability site A is WORSE than site B: %.3f" % \
    #    np.mean(delta_samples < 0))
    # print("Probability site A is BETTER than site B: %.3f" % \
    #    np.mean(delta_samples > 0))

# =================================== #
# ===== Cheating among students ===== #
# =================================== #
# anonymized data

N = 100
with pm.Model() as model_ch:
    p = pm.Uniform("freq_cheating", 0, 1)  # probability of cheating
    true_answers = pm.Bernoulli("truths", p, shape=N, initval=np.random.binomial(1, 0.5, N))
    first_coin_flips = pm.Bernoulli("first_flips", 0.5, shape=N, initval=np.random.binomial(1, 0.5, N))
    second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, initval=np.random.binomial(1, 0.5, N))
    val = first_coin_flips * true_answers + (1 - first_coin_flips) * second_coin_flips
    observed_proportion = pm.Deterministic("observed_proportion", at.sum(val) / 100.0)
    observations = pm.Binomial("obs", N, observed_proportion, observed=35)
    step = pm.NUTS()
    burned_trace = pm.sample(1000, step=step, cores=1, chains=1, progressbar=True)

# alternative model
with pm.Model() as model_alt:
    p = pm.Uniform("freq_cheating", 0, 1)
    p_skewed = pm.Deterministic("p_skewed", 0.5 * p + 0.25)
    yes_responses = pm.Binomial("number_cheaters", N, p_skewed, observed=35)
    step = pm.NUTS()
    burned_trace = pm.sample(1000, step=step, cores=1, chains=1, progressbar=True)
# this model runs way faster than the first one, it might be the observed proportion formula (?) or just the amount of dist
# the plot works for both models
from matplotlib import pyplot as plt

p_trace = burned_trace.posterior["freq_cheating"].values[0]
plt.hist(p_trace, histtype="stepfilled", density=True, alpha=0.85, bins=30, label="posterior distribution", color="#348ABD")
plt.vlines([0.05, 0.35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()
