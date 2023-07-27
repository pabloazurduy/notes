import pymc as pm 
import arviz as az

obs = [1,2,2,1,3,3,4,1,1,3,4, 1,2,3,2] # some realizations of the process 
with pm.Model() as model:
    theta = pm.Exponential("poisson_param", 1.0, initval=0.5) #prior 
    y = pm.Poisson("y", theta, observed=obs) #likelihood function 
    y_sim = pm.Poisson("y_sim", theta ) # predictive posterior distribution samples
    
    step = pm.NUTS()
    trace = pm.sample(1000, step=step, cores=1, chains=1, progressbar=True)
    ppc = pm.sample_posterior_predictive(trace, var_names=["y"])
az.summary(trace)
az.summary(ppc)
az.plot_ppc(ppc)


with pm.Model() as model_2:
    theta = pm.Exponential("poisson_param", 1.0, initval=0.5) #prior 
    y = pm.Poisson("y", theta, observed=obs) #likelihood function 
    y_sim = pm.Poisson("y_sim", theta ) # predictive posterior distribution samples
    
    step = pm.NUTS()
    trace = pm.sample(1000, step=step, cores=1, chains=1, progressbar=True)

az.summary(trace)
