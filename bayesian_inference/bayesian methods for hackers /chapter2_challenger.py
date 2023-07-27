# ========================================== #
# Example: Challenger Space Shuttle Disaster #
# ========================================== #
import numpy as np
import aesara.tensor as at
from matplotlib import pyplot as plt
import pymc as pm 
import arviz as az

challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
#drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

#plot it, as a function of tempature (the first column)
print("Temp (F), O-Ring failure?")
print(challenger_data)

plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature")

temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?
with pm.Model() as model_ch:
    beta = pm.Normal("beta", mu=0, tau=0.001, initval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, initval=0)
    p = pm.Deterministic("p", 1.0/(1. + at.exp(beta*temperature + alpha)))
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    
    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.NUTS()
    idata = pm.sample(120000, step=step, start=start, cores=1, chains=1, progressbar=True)
    
az.plot_trace(idata, combined=True)
# plot confidence interval 
from scipy.stats.mstats import mquantiles
from scipy.special import expit as logistic

alpha_samples = idata.posterior["alpha"].values[0] 
beta_samples = idata.posterior["beta"].values[0] 
t = np.linspace(temperature.min() - 5, temperature.max()+5, 50)
p_t = logistic(-1*(np.outer(beta_samples,t)+ alpha_samples[:, None])) # logistic is 1/(1+e(-X)) in this example they invert the X sign 
mean_prob_t = p_t.mean(axis=0)

# vectorized bottom and top 2.5% quantiles for "confidence interval"
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t, *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t, qs[0], label="95% CI", color="#7A68A6", alpha=0.7)
plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of defect")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.xlabel("temp, $t$")

plt.ylabel("probability estimate")
plt.title("Posterior probability estimates given temp. $t$")

# day of the challenger disaster 
prob_31 = logistic(-(beta_samples * 31 + alpha_samples))
plt.xlim(0.995, 1)
plt.hist(prob_31, bins=1000, density=True, histtype='stepfilled')
plt.title("Posterior distribution of probability of defect, given $t = 31$")
plt.xlabel("probability of defect occurring in O-ring")


# evaluation of the posterior 
import aesara.tensor as at
N = 10000
with pm.Model() as model_ch_ev:
    beta = pm.Normal("beta", mu=0, tau=0.001, initval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, initval=0)
    p = pm.Deterministic("p", 1.0/(1. + at.exp(beta*temperature + alpha)))
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    
    simulated = pm.Bernoulli("bernoulli_sim", p)
    step = pm.NUTS()
    trace = pm.sample(N, step=step, cores=1, chains=1, progressbar=True)
simulations = trace.posterior["bernoulli_sim"].values[0]
posterior_probability = simulations.mean(axis=0)

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay, 
                              roc_curve, RocCurveDisplay, 
                              det_curve, DetCurveDisplay,
                              roc_auc_score, average_precision_score)
auc = roc_auc_score(D, posterior_probability)
auc_pr = average_precision_score(D, posterior_probability)
print(f'{auc = } {auc_pr = } ')

fpr, fnr, thresholds = det_curve(D, posterior_probability)
det_display = DetCurveDisplay(fpr=fpr, fnr=fnr)

fpr, tpr, _ = roc_curve(D, posterior_probability)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

prec, recall, thresholds = precision_recall_curve(D, posterior_probability)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(12, 8))
roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
det_display.plot(ax=ax3)
ax1.set_title(f'ROC curve AUC={auc:.4f}')
ax2.set_title(f'PR curve AUCPR={auc_pr:.4f}')
ax3.set_title(f'Detection Error Tradeoff (DET) curves')
plt.show()

# this plot is from https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):    
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("probability threshold")
    plt.axis([1, 0, 0, 1.1])
    plt.title('Precision Recall Threshold curves')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_precision_recall_vs_threshold(prec, recall, thresholds)