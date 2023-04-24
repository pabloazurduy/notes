
*based on [link][1]*
*created on: 2023-04-21 17:38:26*
# Synthetic Control

Synthetic control is a event studies methodology that is super close to [diff and diff][2]. It usually used when DnD its limited for one of two reasons:

1. We don't have a way to select the control units for our treatment group, and this can introduce selection bias, given that our criteria will be subjective.

2. The capacity of the control group to represent the treatment group might be limited not only because we might have not enough units that can accurately represent a conterfactual, but they might not have an extrapolation capacity (for example small cities vs large cities)

[Abadie and Gardeazabal 2003][3] method uses a weighted average of units in the donor pool to model the counterfactual. This is just to apply the weighting subclassification technique to solve for the lack of comparable units in both groups. 

There are some advantages of this method:

1. The method avoids extrapolation. It uses instead interpolation, and thus the counterfactual is based on where data actually is, as opposed to extrapolating beyond the support of the data, which can occur in extreme situations with regression

2. The construction of the counterfactual does not require access to the post-treatment outcomes during the design phase of the study, unlike regression. The advantage here is that it helps the researcher avoid “peeking” at the results while specifying the model

3. The weights that are chosen make explicit what each unit is contributing to the counterfactual. This is a transparency advantage over the regression approach, however it might generate some suspicion. 

### Caveats on SC

To search the weights we solve the following optimization problem.

$$
\min_{w} \sqrt{(X_1 - X_0 w)^T V (X_1 - X_0 w)}
$$

subject to

$$
w_j \geq 0 \quad \text{for} \quad j = 1, \dots, J
$$

and

$$
w_1 + \dots + w_J = 1
$$

where $X_1$ is a vector of covariates for the treated unit, $X_0$ is a matrix of covariates for the control units, $w$ is a vector of weights for the control units, and $V$ is a diagonal matrix of non-negative components that reflect the relative importance of the covariates.

The objective function is the weighted norm of the difference between the treated unit and the synthetic control in terms of the covariates. The constraints ensure that the weights are non-negative and sum to one. The solution to this problem is the set of weights that best match the treated unit and the synthetic control in the pre-treatment period.

[Ferman, Pinto, and Possebom (2020)][4] argue that the choosing of the covariates can introduce a lot of bias, similar to the bias that we were trying to avoid on the first place. They find that the probability of falsely rejecting the null in at least one specification for a 5% significance test can be as high as 14% when there are pre-treatment periods.

[Abadie, Diamond, and Hainmueller (2010)][5] test the effect of different periods and specifications on the model results, they find out that using lag variables (from the outcome variable), help to reduce the effect of unobserved variables on the weights' optimization. They find that adding these variables on covariant specification helps to reduce the over-representation of units in the control that might not be closer to the treatment units when hidden confounders, if they weren't hidden, are considered. The explanation behind this, is that using lag variables usually are a good method to get the effect of confounders on the outcome variable. 

A possible takeaway is that the creators of synthetic control may claim that this method reduces bias too much, without considering the bias that comes from choosing the model specification. A good way to avoid this problem is to try different specifications for the model (using different sets of covariates) and see if the results are consistent. This can lower the chance of making wrong conclusions from the data based on a specific specification

### Formalization 

As we specified on the previous chapter we solve an minimization problem to find the optimal weights $w_i$ on each control unit $i$. we assume there it's only one treated unit $J=1$, then the minimization problem usually is reduced to the following:

$$
\sum_{m=1}^k v_m \bigg(X_{1m} - \sum_{j=2}^{J+1}w_jX_{jm} \bigg)^2
\text{(2)}
$$
where $X_{i,m}$ is the value of the covariate $m$ for the unit $i$, $v_m$ represents **the weight that we assign to the covariate $m$** 

The choice of $V$, as should be seen by now, is important because $W^*$ depends on one’s choice of $V$. The synthetic control $W(V)$ is meant to reproduce the behavior of the outcome variable for the treated unit in the absence of the treatment. $V$ it's nonetheless than the specification of the model. 

[Abadie, Diamond, and Hainmueller (2010)][5] suggests different choices of $V$, but ultimately it appears from practice that most people choose a $V$ that minimizes the mean squared prediction error:

$$
\sum_{t=1}^{T_0} \bigg (Y_{1t} - \sum_{j=2}^{J+1}w_j^*(V)Y_{jt}\bigg )^2
\text{(3)}
$$

Therefore the function $w_j^*(V)$ will be given by the $argmin$ of the first optimization problem $(2)$. We replace that in the second optimization problem $(3)$ and we get $V$. 

### Significance of the estimation 

[Abadie, Diamond, and Hainmueller (2010)][5] expound on the method by using a cigarette tax in California called Proposition 99. They compare California with a synthetic control based on states that didn't approve any regulation.

<img src="img/synthetic_control_1.png" style='height:400px;'>


How do we determine whether the observed difference between the two series is a statistically significant difference? After all, we only have two observations per year. **Maybe the divergence between the two series is nothing more than prediction error, and any model chosen would’ve done that**, even if there was no treatment effect.

To determinate the significance we follow the “no treatment effect whatsoever” [Firpo and Possebom (2018)][6]. Which basically consist on running the same syhtetic control analysis assuming that the treatment control is every other control unit in the donors pool. Then we estimate how "big" is the effect on all the other units and we compare it to the actual treated unit, if the treated unit is still very extreme, compared to the placebo analysis, then we can use that value as the extreme value to determinate the significance of the method. 

We do need to first drop the units that have a bad fitting on the training period, to do that we follow this steps:

1. Iteratively apply the synthetic control method to each country/state in the donor pool and obtain a distribution of placebo effects.

2. Calculate the RMSPE (RMSE of the outcome prediction for the "treated" unit) for each placebo for the **pre-treatment** period:
$$
RMSPE = \bigg (\dfrac{1}{T-T_0} \sum_{t=T_0+t}^T \bigg (Y_{1t} - \sum_{j=2}^{J+1} w_j^* Y_{jt} \bigg )^2 \bigg )^{\tfrac{1}{2}}
$$ 
 
3. Calculate the RMSPE for each placebo for the **post-treatment** period (similar equation but for the post-treatment period).

4. Compute the ratio of the post- to pre-treatment RMSPE. This will provide a sense on how good are the synthetic control models for all "placebo-treated" units. 

5. Sort this ratio in descending order from greatest to highest.

[Abadie, Diamond, and Hainmueller (2010)][5] recommend iteratively dropping the states whose pre-treatment RMSPE is considerably different than California’s because as you can see, they’re kind of blowing up the scale and making it hard to see what’s going on. 

<img src="img/synthetic_control_2.png" style='height:400px;'>

They do this in several steps, but I’ll just skip to the last step (Figure 10.5). In this, they’ve dropped any state unit from the graph whose pre-treatment RMSPE is more than two times that of California’s. This therefore limits the picture to just units whose model fit, pre-treatment, was pretty good, like California’s.

<img src="img/synthetic_control_3.png" style='height:400px;'>

But, ultimately, inference is based on those exact $p$-values. So the way we do this is we simply create a histogram of the ratios, and more or less mark the treatment group in the distribution so that the reader can see the exact $p$-value associated with the model. To do that we basically build the distributions of the rations pre-post RMSPE and we estimate the probability of getting the result for the actual treated unit. In this case this is 0.026, that is less than 5%.

<img src="img/synthetic_control_4.png" style='height:400px;'>

Another valid way to check for the validity of our models is to use a "placebo" treatment date, [Abadie, Diamond, and Hainmueller (2015)][7] to check if the synthetic control model is actually a good predictor of the outcome value in the treatment city. 


[//]: <> (References)
[1]: <https://mixtape.scunning.com/10-synthetic_control>
[2]: <https://mixtape.scunning.com/09-difference_in_differences>
[3]: <https://mixtape.scunning.com/references.html#ref-Abadie2003>
[4]: <https://mixtape.scunning.com/references.html#ref-Ferman2020>
[5]: <https://mixtape.scunning.com/references.html#ref-Abadie2010>
[6]: <https://mixtape.scunning.com/references.html#ref-Firpo2018>
[7]: <https://mixtape.scunning.com/references.html#ref-Abadie2015>


[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)