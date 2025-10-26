
#### **Confidence interval**
if we run an experiment many times, and you build a 95% confidence interval for each, there's a 95% chance that the true value of the parameter will be in the interval.

#### **Central Limit Theorem** (CLT)
Even if the underlaying distribution of $X$ is not normal, the distribution of the sample mean $\bar{X}$ will be approximately normal if the sample size is large enough.

You can always build a confidence inteval using the sample mean and the standard error of the mean (SEM).

$$SE = \frac{\sigma}{\sqrt{n}}$$

$$CI = \bar{X} \pm Z_{\alpha} * SE$$

Where $Z_{x=\alpha}$ is the inverse cumulative distribution function (CDF) of the standard normal distribution evaluated in $x=\alpha$. for example, for a 95% confidence interval, $Z_{\alpha} = 1.96$.

```python 
from scipy import stats

print(stats.norm.ppf(0.5))  # -> 0.0
print(stats.norm.ppf(0.975))  # -> 1.959
ci = (mean - stats.norm.ppf(0.975) * se, mean + stats.norm.ppf(0.975) * se) # 95% CI
```

#### Non Inferiority testing 
When we run a t-test to compare two means our null hypothesis is that the two means are equal. 

$$H_0: \mu_{new} - \mu_{old} = 0$$

The fact that we fail to reject the null hypothesis **does not mean that the two means are equal, it just means that we don't have enough evidence to say that they are different**. In other words **Absence of evidence is not evidence of absence.** When rejecting the null hypothesis, I can't prove that two means are equal. I can only assume that given the data I have, I can't prove that they are different.

Sometimes is useful to test if the difference, if exist, is at least $\delta$ that's called a non-inferiority margin. This might be useful when we want to shut down a new feature if it is worse than the current one by at least $\delta$.

$$H_0: \mu_{new} - \mu_{old} \geq -\delta$$
