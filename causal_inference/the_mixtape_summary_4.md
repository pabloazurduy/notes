*based on ["Causal Inference The Mixtape"][1]*
*created on: 2024-11-12 20:00:43*

## Chapter 5: Matching and Subclassification

### CIA: Conditional Independence Assumption

In many cases, we can't hold the Independence Assumption because we don't have a random experiment or can't satisfy SUTVA. In these scenarios, we can use a slightly more relaxed assumption that, given certain conditions, will allow us to achieve independence and, therefore, estimate causality. 

We define the Conditional Independence Assumption (CIA) as:

$$
\begin{align}
CIA: (Y^1,Y^0)  & \perp D \mid X  \nonumber \\
IA:  (Y^1,Y^0)  & \perp D \nonumber 
\end{align}
$$


This means that, conditioning on $X$, the potential outcomes are independent of the treatment assignment. This is an easier assumption than the Independence Assumption but very powerful.

### Common Support (Assumption)

CIA by itself is already a strong assumption, but it is not enough to guarantee that we can estimate the ATE. We also need to ensure that the treatment and control groups have common support. Common support means that there are no units in the treatment group that are more similar to the control group than to any other unit in the control group. More Formally:

$$
\begin{align}
\text{(Conditional Independence)}  \quad &(Y^1,Y^0) \perp D\mid X    \nonumber \\
\text{(Common Support)}  \quad & 0<Pr(D=1 \mid X) <1   \nonumber \\
\end{align}
$$

Having both CIA and Common Support will allow us to estimate the ATE using the following estimator:

$$\begin{align}
   \widehat{\delta_{ATE}}= \int \Big(E\big[Y\mid X,D=1\big] - E\big[Y\mid X,D=0\big]\Big)d\Pr(X)
\end{align}
$$

This estimator is a weighted average of the differences and has many versions, lets start with the discrete one (easier to grasp).

### Subclassification

Subclassification is a method of satisfying the backdoor criterion by weighting differences in means by strata-specific weights. These strata-specific weights will, in turn, adjust the differences in means so that their distribution by strata is the same as that of the counterfactual’s strata. Under CIA, we can assume that a subclassification estimator is a good estimator of ATE. 

**Which variable(s) should we use for adjustment?**  We need to choose a set of variables that satisfy the backdoor criterion. If the backdoor criterion is met, then all backdoor paths are closed, and if all backdoor paths are closed, then CIA is achieved. 

We call such a variable the covariate. A covariate is usually a random variable assigned to the individual units prior to treatment. This is sometimes also called exogenous. Harkening back to our DAG chapter, this variable must not be a collider as well. A variable is exogenous with respect to $D$ if the value of $X$ does not depend on the value of $D$.

Oftentimes, though not always and not necessarily, this variable will be time-invariant, such as race. Thus, when trying to adjust for a confounder using subclassification, rely on a credible DAG to help guide the selection of variables. Remember—your goal is to meet the backdoor criterion.

Where the estimator will be defined:

$$\widehat{\delta}_{ATE} = \sum_{k=1}^K\Big(\overline{Y}^{1,k} - \overline{Y}^{0,k}\Big)\times \bigg( \dfrac{N^k_T}{N_T} \bigg )$$

where $K$ are the stratums, and $N_T$ is the total number of units in the sample. Therefore $N_T^k$ is the number of units on the stratum. 

The main limitation is that some stratums might not have enough units to estimate the difference $\overline{Y}^{1,k} - \overline{Y}^{0,k}$, this problem increases when the dimension of $X$ increases, this problem is known as the _"curse of dimensionality"_.

### Exact Matching

Matching is a methodology that, instead of grouping units via stratification, matches units that have the same value of $X$ or a "close enough" $X$ value. Is very straight forward because, with each matched unit (or units), we have a unit-wise conterfactual estimator.

The main caveat here is what we mean by "close enough". Relaxing this "closeness" requirement increase the matching units but we are one step away of matching units that are "too far apart". To solve that problem we have another methodology known as "approximated matching", but in "exact matching" we will assume that all matched units are "close enough" so we can easily fulfill the CIA. 

To estimate ATE using exact matching, we first define what will be the matched $M$-units (closer) for each unit $i$. We do this for units on the control and units on the treatment. The estimator will be given by the following expression:

$$\widehat{\delta}_{ATE} = \dfrac{1}{N} \sum_{i=1}^N (2D_i - 1) \bigg [ Y_i - \bigg ( \dfrac{1}{M} \sum_{m=1}^M Y_{j_m(i)} \bigg ) \bigg ]$$

Where $2D_i-1$ is just a trick to invert the sign of the estimator when the $i$ unit belongs to the control group (then is -1) and when $i$ is part of the treatment then is 1. This will switch the $y_i-y_j$ difference. 

As you can see in the estimator, we never accounted for "distance" between the matched units, this is because we are assuming that the matched units are "close enough" to each other. But what if that assumption is too much of a stretch in our dataset ?, lets introduce "approximated matching".

### Approximated Matching

Approximated matching is a methodology that relaxes the "close enough" assumption in exact matching. In approximated matching, we will understand that further units might be introducing some bias to the estimator that we should correct for. 

We will use two common distances for matching: normalized Euclidean distance and Mahalanobis distance. we will avoid plain euclidian distance because is too sensitive to covariate range, The normalized Euclidean distance is given by:

$$d_{ij} = \sqrt{(X_i-X_j)'\widehat{V}^{-1}(X_i-X_j)}=\sqrt{\sum_{k=1}^K \dfrac{(X_{ik} - X_{jk})^2}{\sigma_k^2}}$$

Once we have the distance metric we can corrrect from how "far" is a sample from his "counterfactual", this will be translated on the following estimator. We will use ATT given that we are calculating the difference on one way. ATE should be calculated using the oposit sign too. 

$$
\begin{align}
   \widehat{\delta}_{ATT}^{BC} = \dfrac{1}{N_T} \sum_{D_i=1} \bigg [ (Y_i - Y_{j(i)}) - \Big(\widehat{\mu}^0(X_i) - \widehat{\mu}^0(X_{j(i)})\Big) \bigg ]
\end{align}
$$

Where $\widehat{\mu}^0(X)$ is an estimate of $E[Y\mid X=x,D=0]$ using, for example, OLS. In a nutshell $\widehat{\mu}^0(X)$ is the regression  prediction of $Y$ using $X$, in this case we are predicting $Y$ given that $D=0$ in other words, predicting the control conterfactual of the treated unit. 

[//]:the_mixtape_summary_32.md> (References)
[1]: <https://mixtape.scunning.com/>


[//]:the_mixtape_summary_32.md> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)