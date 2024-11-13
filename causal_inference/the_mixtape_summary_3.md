*based on ["Causal Inference The Mixtape"][1]*
*created on: 2024-11-12 20:00:43*

## Causal DAGs:

Backdoor criterion: we say that $A \rightarrow B$ satisfies the backdoor criterion if there is no other path from $A$ to $B$ that we are omitting on the model. If we satisfy the backdoor criterion, we can estimate the causal effect of $A$ on $B$. We can fix "ommited variable bias" we have two alternatives:
1. Conditioning for $C$. If $C$ is a confounder.
2. Closing the path using a collider. 


## Chapter 3 Potential Outcomes Model
### Potential Outcomes 

A unit’s observable outcome is a function of its potential outcomes determined according to the **switching equation**:

$$Y_i = D_iY_i^1 + (1-D_i)Y_i^0$$

where $D_i$ is the treatment assignment for unit $i$, $Y_i(1)$ is the potential outcome under treatment, and $Y_i(0)$ is the potential outcome under control. Using this notation, we define the unit-specific treatment effect, or causal effect, as the difference between the two states of the world:

$$\delta_i = Y^1_i-Y^0_i$$
Immediately we are confronted with a problem. If a treatment effect requires knowing two states of the world, $Y^1_i$ and $Y^0_i$, but by the switching equation we observe only one, then we cannot calculate the treatment effect.

From this simple definition of a treatment effect come three different parameters that are often of interest to researchers. They are all population means. The first is called the average treatment effect:

$$
\begin{align}
   ATE & = E[\delta_i] \nonumber      \\
       & = E[Y^1_i - Y^0_i] \nonumber \\
       & = E[Y^1_i] - E[Y^0_i]        
\end{align}
$$

Neither the ATE nor the individual treatment effect $\delta_i$ can be directly observed in practice. However, we can estimate these parameters, with ATE being the more commonly used of the two. (we can estimate the individual treatment effect using CATE estimators).

The second parameter of interest is the average treatment effect for the treatment group. That’s a mouthful, but let me explain. There exist two groups of people in this discussion we’ve been having: a treatment group and a control group. The average treatment effect for the treatment group, or ATT for short, is simply that population mean treatment effect for the group of units that had been assigned the treatment in the first place according to the switching equation. Insofar as $\delta_i$ differs across the population, the ATT will likely differ from the ATE.

$$
\begin{align}
   ATT & = E\big[\delta_i\mid D_i=1\big] \nonumber                 
   \\
       & = E\big[Y^1_i - Y^0_i \mid D_i = 1\big] \nonumber          
   \\
       & = E\big[Y^1_i\mid D_i=1\big] - E\big[Y^0_i\mid D_i=1\big]
\end{align}
$$

The final parameter of interest is called the average treatment effect for the control group, or untreated group. It’s shorthand is ATU, which stands for average treatment effect for the untreated. And like ATT, the ATU is simply the population mean treatment effect for those units who sorted into the control group. 

$$
\begin{align}
   ATU & = E\big[\delta_i\mid D_i = 0\big] \nonumber                          
   \\
       & = E\big[Y^1_i - Y^0_i\mid D_i = 0\big] \nonumber                     
   \\
       & =E\big[Y^1_i\mid D_i=0\big]-E\big[Y^0_i\mid D_i=0\big]
\end{align}
$$

The only alternative that we have when we have data from a randomized experiment is to estimate the simple difference in means between the treatment and control groups (also known as SDO). 

$$
\begin{align}
SDO &=E\big[Y^1\mid D=1\big] - E\big[Y^0\mid D=0\big] \nonumber
\\
&= \dfrac{1}{N_T} \sum_{i=1}^n \big(y_i\mid d_i=1\big) - \dfrac{1}{N_C} \sum_{i=1}^n \big(y_i\mid d_i=0\big)
\end{align}
$$

As we can probably guess SDO looks a lot like the ATE, so our main question is: can we use SDO as an estimator for the ATE?. to answer that we can decompose SDO into their components:

$$
\begin{align}
\underbrace{\dfrac{1}{N_T} \sum_{i=1}^n \big(y_i\mid d_i=1\big)-\dfrac{1}{N_C}
   \sum_{i=1}^n \big(y_i\mid d_i=0\big)}_{ \text{Simple Difference in Outcomes}}
&= \underbrace{E[Y^1] - E[Y^0]}_{ \text{Average Treatment Effect}}
\\
&+ \underbrace{E\big[Y^0\mid D=1\big] - E\big[Y^0\mid D=0\big]}_{ \text{Selection bias}}
\\
& + \underbrace{(1-\pi)(ATT - ATU)}_{ \text{Heterogeneous treatment effect bias}}
\end{align}
$$

As you might expect, SDO will equal the ATE only when both the "selection bias" and "heterogeneous treatment effect bias" terms are zero. Let's examine what these terms mean:

1. Selection Bias: This represents the inherent differences between the treatment and control groups. We can assume both groups are similar only if their expected outcomes under no treatment are equal:

    $$E[Y^0|D=1] - E[Y^0|D=0] = 0$$

2. Heterogeneous Treatment Effect Bias: This term captures the difference between the ATT and ATU. If the treatment effect is the same for everyone, then this term will be zero. However, if the treatment effect varies across both populations (treatment and control), then this term will be non-zero.
    
    $$ATT = ATU$$

These two assumptions "no selection bias" and "homogeneus treatment effect" rarely hold in observational studies, mainly because treatments are usually the result of an individual selection process that is usually not random. 

### Independence Assumption

Let’s start with the most credible situation for using SDO to estimate ATE: when the treatment itself (e.g., surgery) has been assigned to patients independent of their potential outcomes. But what does this word “independence” mean anyway? Well, notationally, it means:

$$(Y^1,Y^0) \perp D$$

What this means is that surgery was assigned to an individual for reasons that had nothing to do with the gains to surgery. 

Rubin argues that there are a bundle of assumptions linked to the independence assumption, and he calls these assumptions **"the stable unit treatment value assumption"**, or SUTVA for short. That’s a mouthful, but here’s what it means: our potential outcomes framework places limits on us for calculating treatment effects. We can simplify this assumption into two parts:

1. homogeneity of treatment effects: the treatment effect is the same for everyone, or everyone receive "the same sized dose". 
2. no spillovers: the treatment of one individual does not affect the outcome of another individual. (no "network effects")

### Randomized Experiments and P-Values

Under certain conditions in randomized experiments, we can easily assume that SUTVA and the independence assumption hold. These conditions, along with other factors, have made the use of randomized experiments the "gold standard" in causal inference.

#### P-value 

Let's assume that you have a randomized experiment where we can estimate some SDO. However, we don't know if the estimated SDO is different from 0 because of random chance or because there's actually a treatment effect.

To measure confidence, we build a statistic that will, under a null hypothesis, follow a known distribution. With that distribution, we can estimate how unlikely it is that the SDO is different from zero if there is actually no treatment effect.

There are two common "null hypotheses" that imply the treatment effect is zero:

1. The "Fisher's sharp null hypothesis" implies that the treatment effect is zero for all individuals.
    $$H_0: \delta_i = 0 \ \text{for all $i$}$$
2. The "Neyman null hypothesis" implies that the average treatment effect is zero.
    $$H_0: ATE = 0$$

Under both null hypotheses, we can build a test statistic that will follow a known distribution. however, for simplicity, we will use fisher's sharp null hypothesis.

Under this hypothesis we can an statistic that will allow us to estimate the probability of observing the SDO given that the individual treatment effect is zero (fisher's sharp null hypothesis). This statistic is called the p-value.

$$
\Pr\Big(t(D',Y)\geq t(D_{observed},Y) \mid \delta_i=0, \forall i)\Big)=
   \dfrac{\sum_{D'\in \Omega} I(t(D',Y) \geq t(D_{observed},Y))}{K}
$$

we have also other test that follow the same principle to calculate differences in outcomes by treatment status. We considered simple differences in averages, simple differences in log averages, differences in quantiles, and differences in ranks. 

However, Imbens and Rubin (2015) note that focusing solely on a few features of the data (e.g., skewness) can cause us to miss differences in other aspects. This can be particularly problematic if the variance in potential outcomes for the treatment group differs from that of the control group. Focusing only on simple average differences may not generate p-values that are extreme enough to reject the null hypothesis, even when it does not hold.

Therefore, we may be interested in a test statistic that can detect differences in the overall distributions between the treatment and control units. One such test statistic is the Kolmogorov-Smirnov (K-S) test statistic. The K-S test is important because it compares the entire distributions of the two groups, not just the means or specific quantiles. It is sensitive to differences in both location and shape of the empirical cumulative distribution functions (ECDFs) of the two samples. This makes the K-S test more powerful in detecting any differences between the groups, especially when the differences are not limited to the mean but involve variability, skewness, or other distributional features that mean difference tests might miss.



[//]:the_mixtape_summary_32.md> (References)
[1]: <https://mixtape.scunning.com/>
[2]: <https://stats.stackexchange.com/a/55891/274422>
[3]: <https://en.wikipedia.org/wiki/Ramsey_RESET_test>


[//]:the_mixtape_summary_32.md> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)