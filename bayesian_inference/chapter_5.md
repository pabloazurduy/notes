extracted from  [Probabilistic Programming and Bayesian Methods for Hackers Chapter 5][1]

### Loss Functions 
A loss function is a function of the true parameter, and an estimate of that parameter:

$$L( \theta, \hat{\theta} ) = f( \theta, \hat{\theta} )$$

The important point of loss functions is that it measures how bad our current estimate is: the larger the loss, the worse the estimate is according to the loss function. A simple, and very common, example of a loss function is the squared-error loss:

$$L( \theta, \hat{\theta} ) = ( \theta -  \hat{\theta} )^2$$

another very popular loss function is the log-loss function

$$L( \theta, \hat{\theta} ) = -\theta\log( \hat{\theta} ) - (1- \theta)\log( 1 - \hat{\theta} ), \; \; \theta \in {0,1}, \; \hat{\theta} \in [0,1]$$

### In the real world 

In Bayesian Inference we assume that every parameter is a random variable with a prior and a posterior, therefore there is nothing like a "true value" just a few "realizations". Given that we will be focused in estimating the `Expected Loss` rather than a Loss estimation based in a point estimate (or just one sample from the posterior).

Given a point estimate $\hat{\theta}$ of the parameter $\theta$ we can estimate the **risk** of that estimation estimating the loss function on that particular point estimation we define as $l(\hat{\theta})$, also know as **the expected loss of** $\hat{\theta}$. We can estimate the value of this expected value using simulations over $\theta$ posterior. **This is a fixed value for** $\hat{\theta}$ and **we sample over** $\theta_i$ using $\theta$ posterior.

$$l(\hat{\theta} ) = E_{\theta}\left[ \; L(\theta, \hat{\theta}) \; \right] = \frac{1}{N} * \sum_{i \in N} L(\theta_i, \hat{\theta})$$

> Compare this with frequentist methods, that traditionally only aim to minimize the error, and do not consider the loss associated with the result of that error. Compound this with the fact that frequentist methods are almost guaranteed to never be absolutely accurate. Bayesian point estimates fix this by planning ahead: your estimate is going to be wrong, you might as well err on the right side of wrong.




[//]: # (References)
[1]: <https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter5_LossFunctions/Ch5_LossFunctions_PyMC3.ipynb#Loss-Functions>