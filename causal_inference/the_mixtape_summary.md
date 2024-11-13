*based on ["Causal Inference The Mixtape"][1] *
*created on: 2024-11-12 20:00:43*


## Chapter 2: Probability and Regression Review.

we will say that two events $A$ and $B$ are independent if and only if: 

$$\Pr(A\mid B)=\Pr(A)$$

When we have two independent events ($A \perp  B$) we can calculate the joint probability as:

$$\Pr(A \cap B) = \Pr(A , B)=\Pr(A)\Pr(B)$$

Law of total probabilities, $A$ is any event and $B_n$ is a complete set of events 

$$\Pr(A)=\sum_n Pr(A\cap B_n)$$

Conditional probability definition:

$$\Pr(A \mid B)  = \dfrac{\Pr(A \cap B)}{\Pr(B)}$$

Bayes Rule: 

$$\Pr(A\mid B) = \frac{\Pr(B\mid A) \Pr(A)}{\Pr(B)}$$

### Expected Value 

Expected Value (definition). $f(x)$ density function for variable $X $
$$
\begin{align}
  E(X) & = x_1f(x_1)+x_2f(x_2)+\dots+x_kf(x_k) \\
  E(X) & = \sum_{j=1}^k x_jf(x_j)              
\end{align}
$$
properties:

$$
\begin{align}
E(c) & =c \\
E(aX+ b) & =E(aX)+E(b)=aE(X)+b \\
E\bigg(\sum_{i=1}^na_iX_i\bigg) & =\sum_{i=1}a_iE(X_i) \\
E\bigg(\sum_{i=1}^nX_i\bigg) & =\sum_{i=1}^nE(X_i) \\
E(X - E(X)) & =0 \\
\end{align}
$$

### Variance

The variance of a random variable $X$ is the expected value of the **squared deviation** from the mean of $X$, $\mu = E(X)$
$$
\begin{align} 
& Var(X) = E[(X- E(X))^2] \\
& Var(X) = E(X^2) - E(X)^2 \\
& Var(X)=\widehat{S}^2=(n-1)^{-1}\sum_{i=1}^n(x_i - \mu)^2
\end{align}
$$

Some properties:
$$Var(aX+b)=a^2Var(X)$$

$$Var(X+Y)=Var(X)+Var(Y)+2\Big(E(XY) - E(X)E(Y)\Big)$$

when the two variables are independt $X \perp Y$ then $E(XY)=E(X)E(Y)$ and as consequence:

$$Var(X+Y)=Var(X)+Var(Y)$$

### Covariance

The **covariance** measures the amount of linear dependence between two random variables. While it’s tempting to say that a zero covariance means that two random variables are unrelated, that is incorrect. They could have a nonlinear relationship. The definition comes from the second term of the variance formula between two variables $Var(X+Y)$

$$Cov(X,Y) = E(XY) - E(X)E(Y)$$

As we said, if $X$ and $Y$ are independent, then $Cov(X,Y) = 0$ in the population. The covariance between two linear functions is:

$$C(a_1+b_1X, a_2+b_2Y)=b_1b_2C(X,Y) $$

Interpreting the magnitude of the covariance can be tricky. For that, we are better served by looking at correlation. Correlation is nothing more than a "normalized" covariance, we do this transforming the random variables via normalization

$$X \rightarrow X' = \frac{X-E(X)}{\sqrt{Var(X)}} = \frac{X- \mu}{\sigma}$$

then we have the correlation as:

$$Cor(X,Y) = Cov(X',Y') = \frac{Cov(X,Y)}{\sqrt{Var(X) Var(Y)}}$$

The correlation coefficient is bounded by -1 and 1. A positive (negative) correlation indicates that the variables move in the same (opposite) ways. The closer the coefficient is to 1 or -1, the stronger the **linear relationship** is.


### Population Model 
Assume that there are two variables, $x$ and $y$, and we want to see how $y$ varies with changes in $x$

$$y=\beta_0+\beta_1x+u$$

we will make our first strong assumption, known as "mean independence assumption". It basically says that all unobserved factors that affect $y$ are uncorrelated with $x$. For example, the ability (unobserved) and the education (observed) of a person are uncorrelated if we want to predict the salary of a person using a simple linear model.

$$E(u\mid x)=E(u) =0 \ \text{for all values $x$} $$

The condition $E(u\mid x)=0$ is known as the **zero conditional mean assumption**. If this assumption is truth we can then imply causality using what we call as the **"conditional expectation function"**. 

$$E(y\mid x)=\beta_0+\beta_1x$$

that's why this assumption is so powerful, because it allows us to interpret the coefficients as causal effects.

### OLS 
with some algebra we can find the OLS estimator for $\beta_1$ 

$$\widehat{\beta}_1 = \dfrac{\widehat{Cov}(x_i,y_i) }{\widehat{Var}(x_i)}$$

Where the $\widehat{Var}$ and $\widehat{Cov}$ are the sample variance and covariance respectively. A consequence of this estimator is that we need that the variable $X$ has some variance, meaning, in the dataset, I need to have some changes on $X$ otherwise I can't estimate the causal relationship with $Y$. 

Some OLS properties:
1. the sum of residuals is zero. Positive differences is equivalent to negative differences, this is a consequence of the minimization squared problem.

$$\sum_{i=1}^n \hat{u}_i = 0$$

2. The sample covariance (and therefore the sample correlation) between the explanatory variables and the residuals is always zero. Same with the residuals and the predicted values. That the covariance is zero means that there's no linear correlation between the covariates, the predicted values and the residuals (this is by construction)

$$\sum_{i=1}^n x_i \widehat{u_i}=0$$
$$\sum_{i=1}^n \widehat{y_i} \widehat{u_i}=0$$

### Goodness of fit 

Define the total sum of squares (SST), explained sum of squares (SSE), and residual sum of squares (SSR). Each of this terms represents variances (without the division by n-1). the SST is $Var(y_i)$ and the SSE is $Var(\widehat{y_i})$ and the SSR is $Var(\widehat{u_i})$ (multiplying by n-1)

$$
\begin{align} 
SST &= \sum_{i=1}^n (y_i - \overline{y})^2 \\

SSE &= \sum_{i=1}^n (\widehat{y_i} - \overline{y})^2 \\

SSR &= \sum_{i=1}^n \widehat{u_i}^2
\end{align}
$$

with some manipulation we can see that the SST is the sum of the SSE and SSR.

$$SST = SSE + SSR$$

Assuming $SST>0$, we can define the fraction of the total variation in $y_i$ that is explained by $x_i$ (or the OLS regression line) as


$$R^2=\dfrac{SSE}{SST}=1-\dfrac{SSR}{SST} = Corr(y_i, \widehat{y_i})^2$$

> I would encourage you not to fixate on $R$-squared in research projects where the aim is to estimate some causal effect, though. It’s a useful summary measure, but it does not tell us about causality. Remember, you aren’t trying to explain variation in $y$ if you are trying to estimate some causal effect.  **For causal inference, we need the zero conditional mean assumption**:

$$E(u\mid x)=0,\ \text{for all values $x$} $$

You can test this hypothesis using some missespecification test such as the [Ramsey RESET test][3]. another way is [to plot residuals against the predicted values][2], if there is a pattern, then the model is not correctly specified.

### Law of Iterated Expectations (LIE)

The conditional expectation function (CEF) is the mean of some outcome $y$ with some covariant value $x$ held fixed.
The LIE says that an unconditional expectation can be written as the unconditional average of the CEF. In other words, $E(y_i)=E \{E(y_i\mid x_i)\}$. This is a fairly simple idea: if you want to know the unconditional expectation of some random variable $y$, you can simply calculate the weighted sum of all conditional expectations with respect to some covariate $x$. 

We can use the CEF and the LIE to build the "CEF decomposition property":

$$y_i=E(y_i\mid x_i)+\varepsilon_i $$

This ultimately can derive with a theorem known as the "Regression CEF Theorem" 

$$\beta=\arg\min_b E\Big\{\big[E(y_i\mid x_i) - x_i'b\big]^2 \Big\}$$

This is quite powerful because it allows us to estimate the causal effect of $x$ on $y$ and also according to Angrist and Pischke (2009), this estimator is also "useful" even when the underlying CEF is not linear, linear regression is a "good" approximation to the CEF.

### Confidence Intervals 
under the previous assumptions + heteroskedasticity $V(u\mid x)=\sigma^2$  The variance of the estimator is given by the formula:

$$V(\widehat{\beta_1})=\dfrac{\sigma^2}{SST_x}$$

As the error variance increases—that is, as $\sigma^2$ increases—so does the variance in our estimator. The more “noise” in the relationship between $y$ and $x$ (i.e., the larger the variability in $u$), the harder it is to learn something about $\beta_1$. In constrat, more variation in $(x_i)$ reduces the variance of the estimator.

[//]: <> (References)
[1]: <https://mixtape.scunning.com/>
[2]: <https://stats.stackexchange.com/a/55891/274422>
[3]: <https://en.wikipedia.org/wiki/Ramsey_RESET_test>


[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)