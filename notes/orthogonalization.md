Notes from the [Orthogonalization article][1]

## Orthogonalization Using Regression

This procedure is a method to debiasing a dataset when there is a strong autocorrelation between some of the regressors and the predicted variable, such as price and demand. Usually, in order to estimate elasticity, we will model the demand as a regression over the price and other demand regressors. 

$$Q = \beta_p * P + \beta_X*X + \beta_0$$

the problem is that usually the price is a variable that is defined based on the demand or the expectation of the demand ($P =f(E(Q))$). by example summer icecream prices and demand in summer. 

Orthogonalization is a method to somehow reduce that bias in the dataset, considering that the variable to study ($P$) is **not randomly assigned** rather than defined throw an optimization process. This method is based on the [Frisch–Waugh–Lovell theorem][2], and is also explained in the book [*The Elements of Statistical Learning : Data Mining, Inference, and Prediction*][3]. 

### The method 

The idea behind the method is similar to the one behind the [propensity score][4]. We try to use the confounders matrix $X$ to try to predict the treatment variable $P$. After that we make a few more weird steps; we isolate the unexplained effect in the dependent variable $Q$ of the confounders $X$ and try to explained by the "randomly" debiased $\hat{P}$. This steps are detailed on the following procedure:

1. we first fit a model to predict the treatment variable $P$ against the confounders (in our price elasticity example, $X$ = time/season/zone), then we create a "debiased random" variable $\hat{P}$ that will contain the *non explained by X* variance of $P$. 

$$P = \theta_X*X + \theta_0$$

```python 
m_t = smf.ols("price ~ cost + C(weekday) + temp", data=test).fit()
test['price-Mt(X)'] = m_t.resid # price - pred_price(X)
```

2. secondly we debiased the outcome variable $Q$ following a similar procedure that the first step, the idea behind this is to remove the effect of the confounders $X$ on $Q$ so we will keep only the _non explained by X_ variance of $Q$ ($\hat{Q}$.
)

$$Q = \gamma_X*X + \gamma_0$$

```python 
m_y = smf.ols("sales ~ cost + C(weekday) + temp", data=test).fit()
test['sales-My(X)'] = m_y.resid # sales - pred_sales(X)
```

3. finally we regress our "debiased" treatment variable $\hat{P}$ against our "debiased" outcome variable $\hat{Q}$

$$\hat{Q} = \beta_p *\hat{P} + \beta_0 $$
```python 
m_y_debiased = smf.ols("sales-My(X) ~ price-Mt(X)", data=test).fit()
```

We know from the theorem of [Frisch][2] that $\beta_p$ is equivalent to the original model regression

$$y_i = \beta_0 + \pmb{\beta_X X} + \pmb{\beta_p P} $$

This is incredibly powerful for causal inference. It says that I can build a model that predicts my treatment $P$ using my features $X$ , a model that predicts the outcome $Q$ using the same features, take the residuals from both models and run a model that estimates how the residual of $\hat{P}$  affects the residual of $\hat{Q}$. This last model will tell me how $P$ affects $Q$  while controlling for $X$. In other words, the first two models are controlling for the confounding variables. They are generating data which is as good as random. This is debiasing my data. That’s what we use in the final model to estimate the elasticity.

Also is worth nothing that we should keep everything in the [regression world][5] in order to use orthogonalization

[Comment]: References 
[1]: <https://matheusfacure.github.io/python-causality-handbook/Debiasing-with-Orthogonalization.html#linear-regression-reborn>
[2]: <https://en.wikipedia.org/wiki/Frisch%E2%80%93Waugh%E2%80%93Lovell_theorem>
[3]: <https://hastie.su.domains/Papers/ESLII.pdf#page=71>
[4]: <https://mixtape.scunning.com/05-matching_and_subclassification#propensity-score-methods>
[5]: <https://matheusfacure.github.io/python-causality-handbook/Debiasing-with-Orthogonalization.html#key-ideas:~:text=Finally%2C%20before%20we,run%20the%20risk%3F>
