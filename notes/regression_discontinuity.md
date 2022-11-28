*notes based on [The Effect The Book (Ch. Regr. Disc.)][1]*
*created on: 2022-11-27 11:48:47*

## Regression Discontinuity
Regression Discontinuity is the study of analyzing two very similar groups where a treatment is assigned based in an arbitrary *cutoff* that separates very similar people (and therefore where "the only difference" is the treatment). Usually the treatment is a binary variable. 

- Running Variable or Forcing Variable $X$: The variable that will be used to evaluate if you receive the treatment or not $D$. Ex: some admissions test score 
- Cutoff $X_C$: The threshold value at where the treatment and control group are separated. 
- Bandwidth $[X_C - \delta, X_C + \delta]$ Interval where it's reasonable to think that people at both sides of the cutoff are similar. Beyond that, there might be other variables that change the two groups that are not only related with the treatment. 

**Some Assumptions**

1. People close to the cutoff are "randomly assigned". 
> We need **continuity** of the relationship between the outcome and the running variable. If, for example, there’s a positive relationship between the outcome and the running variable, we’d expect a higher outcome to the right of the cutoff than to the left. That’s totally fine, even though the two groups aren’t exactly comparable. As long as there wouldn’t be a discontinuity in that positive relationship without the treatment”

2. People shouldn’t be able to manipulate the running variable to choose their own treatment 

3. People who choose what the cutoff it shouldn’t be able to make that choice in response to finding out who has which running variable values

4. The only thing changing in the cutoff is the treatment. For example if two benefits are provided for certain cutoff score, is not possible to separate the effect of both separated treatments. This is also conflicting if another variable (confounder) happens to change at the same cutoff. 

5. Consequence of 1, In the cutoff interval we should be able to zoom in until both sides of the sample are comparable, this sometimes is not possible if the "Running Variable" is binned or discretized until a point where is no longer possible to reduce that interval. 

6. The statistical power of this methodology might be seriously compromised if there is not enough samples on the neighborhood of the cutoff. Using a wider interval might introduce bias too, so there is a complex trade-off. 

Finally, the causal graph that represents this study is the following one. Notice the $Z$ backdoor that might or might not be closed 
<p align="center">
<img src="img/regressiondiscontinuity-dag-1.png" style='height:80px;align:center;'>
</p>

However, even if we have other paths such as $ RunningVariable \rightarrow Z \rightarrow Outcome$ when we condition by the "arbitrary" cutoff we actually can also isolate the effect of the other paths (such as the path via $Z$ ). This makes regression discontinuity more powerful than a simple OLS that controls for a variable. 

The procedure is simple (it only consists in 4 steps)
<p align="center">
<img src="img/regressiondiscontinuity-animation-1.png" style='height:400px;align:center;'>
</p>

The interesting steps are (b) and (c). In (b) we choose a model that predict values on both ends without using the bandwidth interval also we fit two models one per each side. 

In (c) we constrained the model and re-fit it in the bandwidth region (both sides again). We should ignore the samples that are too far from the cutoff because we might be adding additional backdoors that break our "randomly assign" assumption, but also not too close to the cutoff because we will introduce too much variance into our models (fewer samples more overfit), so, there is a trade-off when choosing the bandwidth. 

With this result we are **only able to estimate the effect of treatment on the bandwidth area**, near the cutoff, we can't estimate or extrapolate this effect to the rest of the population, unless we make a huge assumption. Some use cases, such as moving the cutoff, are not limited by this constraint, but if we want to estimate the effects far from the cutoff we might be limited when using this methodology. 

### Estimation 

Using OLS we can fit the following model ($D$ treatment binary). In this particular case $D$ is an indicator of being treated and also being over the cutoff. This is not valid when we talk about [fuzzy regression](#fuzzy-regression-discontinuity)

$$ Y = \beta_0 + \beta_1(X-X_C) + \beta_2*D+\beta_3(X-X_C)*D+ \epsilon
$$

We usually fit this model using a heteroskedasticity-robust standard errors procedure. The results are two lines:

1. $\beta_0$ as intercept and $\beta_1$ as slope (untreated side)
2. $\beta_0 + \beta_2$ intercept and $\beta_1 + \beta_3$ as slope (treated side)

Finally the treatment effect will be estimated by $\beta_2$

One advice is always use a simple model (OLS) and ignore the non-linearity of the data using a reduced bandwidth range. Even if there is a non lineal relationship, on the border of the cutoff, when constrained the bandwidth, the lineal approximation is as good as a non lineal model when comes to estimating the treatment effect. This is similar to lineally approximate a curve, when we zoom in, the approximation is not that bad, at least as close as needed to determinate the ATE. This procedure is call **Local Regression** This approach, however will require a fair amount of data.

There are many **Local Regression** models, some of them included weighted regressions (with triangular kernels) and more popularly the LOESS regression. 

### Fuzzy Regression Discontinuity 

Sometimes the cutoff does not apply directly the treatment, but it increases the probability of having it. This could also include self selection or opt-in treatments (such as admission or retirement)

<p align="center">
<img src="img/regressiondiscontinuity-treatmentshare-1.png" style='height:180px;align:center;'>
</p>




[//]: <> (References)
[1]: <https://theeffectbook.net/ch-RegressionDiscontinuity.html>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)