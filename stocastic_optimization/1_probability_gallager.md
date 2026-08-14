
*based on [link][1]*
*created on: 2026-08-08 22:18:48*
## 1_probability models 

let be $\Omega$ the sample space, the set of all sample points for a given experiment. We will call A as a sequence of events, $\bigcup_{i=1}^{\infty} A_i$ as the union of events, and $\bigcap_{i=1}^{\infty} A_i$ as the intersection of events.


### Axioms for events

Given a sample space $\Omega$.  We will define events as subsets of $\Omega$. that will satisfy the following axioms:

1. The sample space $\Omega$ is an event.
2. for every sequence of events $A_1, A_2, \ldots$, the union $\bigcup_{i=1}^{\infty} A_i$ is an event.
3. for every event $A$, the complement $A^c$ is an event.

### Axioms of probability

Given any sample space $\Omega$ and any class of events $\mathcal{E}$ satisfying the axioms of events, a probability rule is a function $\Pr(\cdot)$ mapping each $A \in \mathcal{E}$ to a (finite) real number in such a way that the following three probability axioms hold:

1. $\Pr(\Omega) = 1$.
2. For every event $A$, $\Pr(A) \geq 0$.
3. The probability of the union of any sequence $A_1, A_2, \ldots$ of disjoint events is given by
$$\Pr\left(\bigcup_{n=1}^{\infty} A_n\right) = \sum_{n=1}^{\infty} \Pr[A_n],$$
where $\sum_{n=1}^{\infty} \Pr[A_n]$ is shorthand for $\lim_{m \to \infty} \sum_{n=1}^{m} \Pr[A_n]$.

#### Probability corollaries:

1. $\Pr(\emptyset) = 0.$

2. $\Pr\left(\bigcup_{n=1}^{m} A_n\right) = \sum_{n=1}^{m} \Pr[A_n] \quad \text{for } A_1, \ldots, A_m \text{ disjoint.}$

3. $\Pr[A^c] = 1 - \Pr[A] \quad \text{for all } A.$

4. $\Pr[A] \leq \Pr[B] \quad \text{for all } A \subseteq B.$

5. $\Pr[A] \leq 1 \quad \text{for all } A.$

6. $\sum_{n} \Pr[A_n] \leq 1 \quad \text{for } A_1, A_2, \ldots \text{ disjoint.}$

7. $\Pr\left(\bigcup_{n=1}^{\infty} A_n\right) = \lim_{m \to \infty} \Pr\left(\bigcup_{n=1}^{m} A_n\right).$    

8. $\Pr\left(\bigcup_{n=1}^{\infty} A_n\right) = \lim_{n \to \infty} \Pr[A_n] \quad \text{for } A_1 \subseteq A_2 \subseteq \cdots$

9. $\Pr\left(\bigcap_{n=1}^{\infty} A_n\right) = \lim_{n \to \infty} \Pr[A_n] \quad \text{for } A_1 \supseteq A_2 \supseteq \cdots$


We will define conditional probability as follows, if $\Pr(B) > 0$, then the conditional probability of $A$ given $B$ is defined as
$$\Pr(A|B) = \frac{\Pr(AB)}{\Pr(B)}.$$

As a consequence of this definition we have the bayes law, which states that if $\Pr(B) > 0$

$$\Pr(A|B) = \frac{\Pr(B|A)\Pr(A)}{\Pr(B)}.$$


definition of independence of events: Two events $A$ and $B$ are independent if 

$$\Pr(AB) = \Pr(A)\Pr(B)$$

or alternatively:

$$\Pr(A|B) = \Pr(A) \quad $$

weirdly enough $\Omega$ and $\emptyset$ are independent of any event $A$. Based on this definition.

Conditional independence. Two events $A$ and $B$ are conditionally independent given $C$ if
$$\Pr(AB|C) = \Pr(A|C)\Pr(B|C)$$

### Repeated Idealized experiments

Given an original sample space $\Omega$, we can define a new sample space $\Omega^n$ as the set of all $n$-tuples $(\omega_1, \ldots, \omega_n)$ where each $\omega_i \in \Omega$. This represents the outcomes of $n$ repeated idealized experiments.

$$ \Omega^n = \{(\omega_1, \ldots, \omega_n) : \omega_i \in \Omega, i = 1, \ldots, n\}$$

### Random variables
A random variable is a function $X: \Omega \to \mathbb{R}$ that assigns a real number to each outcome in the sample space. For any real number $x$, the event $\{X \leq x\}

we will define **the cumulative distribution function** (CDF) of $X$ as:
$$F_X(x) = \Pr(\{\omega \in \Omega : X(\omega) \leq x\}) = \Pr(X \leq x)$$

we will define **the probability density function** (PDF) of $X$ as:
$$f_X(x) = \frac{d}{dx}F_X(x)$$

This function is not always defined, will depend on the characteristics of the CDF, given that the strict conditions are that the CDF is 

in the case of a discrete random variable, the PMF (or probability mass function) is defined as:
$$p_X(x) = \Pr(X = x)$$

#### Table 1.1: The PDF, mean, variance and MGF for some common continuous rv's

| Name | PDF $f_X(x)$ | Mean | Variance |
|---|---|---|---|
| Exponential | $\lambda \exp(-\lambda x); \quad x \geq 0$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^2}$ |
| Erlang | $\dfrac{\lambda^n x^{n-1} \exp(-\lambda x)}{(n-1)!}; \quad x \geq 0$ | $\dfrac{n}{\lambda}$ | $\dfrac{n}{\lambda^2}$ |
| Gaussian | $\dfrac{1}{\sigma \sqrt{2\pi}} \exp\left(\dfrac{-(x-a)^2}{2\sigma^2}\right)$ | $a$ | $\sigma^2$ |
| Uniform | $\dfrac{1}{a}; \quad 0 \leq x \leq a$ | $\dfrac{a}{2}$ | $\dfrac{a^2}{12}$ |

#### Table 1.2: The PMF, mean, variance and MGF for some common discrete rv's

| Name | PMF $p_M(m)$ | Mean | Variance |
|---|---|---|---|
| Binary | $p_M(1) = p; \; p_M(0) = 1-p$ | $p$ | $p(1-p)$ |
| Binomial | $\binom{n}{m}p^m(1-p)^{n-m}; \quad 0 \leq m \leq n$ | $np$ | $np(1-p)$ |
| Geometric | $p(1-p)^{m-1}; \quad m \geq 1$ | $\dfrac{1}{p}$ | $\dfrac{1-p}{p^2}$ |
| Poisson | $\dfrac{\lambda^n \exp(-\lambda)}{n!}; \quad n \geq 0$ | $\lambda$ | $\lambda$ |

Two rv's, say $X$ and $Y$, are *statistically independent* (or, more briefly, *independent*) if

$$F_{XY}(x, y) = F_X(x) F_Y(y) \quad \text{for each } x \in \mathbb{R}, y \in \mathbb{R}.$$

If $X$ and $Y$ have a joint density, then statistical independence can also be expressed as
$$f_{X|Y}(x|y) = f_X(x) \quad \text{for each } x \in \mathbb{R}, y \in \mathbb{R} \text{ such that } f_Y(y) > 0.$$

If the joint density exists and the marginal density $f_Y(y)$ is positive, the conditional density can be defined as
$$f_{X|Y}(x|y) = \frac{f_{XY}(x, y)}{f_Y(y)}.$$

### Stochastic processes

A stochastic process is a collection of random variables $\{X(t): t \in T\}$ defined on a common probability model. These Rvs are usually indexed by time $t$. 

#### Bernoulli process
we will define a Bernoulli process as a sequence of independent and identically distributed (i.i.d.) Bernoulli random variables $\{Z_n\}$, where each $Z_n$ takes the value 1 with probability $p$ and 0 with probability $1-p$. that means that during the interval T we will have a sequence of independent trials, each resulting in a success (1) with probability $p$ or a failure (0) with probability $1-p$.

If we think about the bernoulli process as a sequence or arrivals, then we can also model the time between arrivals as another random variable $X_i$ where his distribution PMF will be given by the geometric distribution:

$$P(X_1 = j) = (1-p)^{j-1}p$$

Where $i=1$ is the first arrival, but subsequently arrivals $X_i$ will be given by the same distribution. If we want to model the number of arrivals up to the $\text{time n}$ we can define a random variable $S_n$ which will be given by the binomial distribution. (number of successes in n trials, and all their possible orders). 

$$ \Pr(S_n = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

if we consider the time between arrivals different than a geometric distribution, for example allowing exponential distribution or other continous non negative distributions we will define a **renewal process**. Renewal process are discrete stochastic process.

### Expected value

The expected value (or mean) of a random variable $X$ is defined as

$$\mathbb{E}[X] = \sum_x x \, \Pr(X = x) \quad \text{for discrete } X,$$
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f_X(x) \, dx \quad \text{for continuous } X.$$

we will say that the expected value **exists** only if the sum is less than infinity. $\mathbb{E}[X] < \infty$.

We can generaly define the expected value of a function $g(X)$ of a random variable $X$ as the complement of the cumulative distribution function (CCDF) of $X$. which in some scenarios can be easier to calculate than the other definition. 

$$\mathbb{E}[X] = \int_{0}^{\infty} \Pr(X > x) \, dx = \int_{0}^{\infty} {F}^c_X(x) \, dx$$

An important property of the expected value is that the expected value of a sum of random variables (**dependent** or independent) is equal to the sum of their expected values. 

$$\mathbb{E}[X_1 + X_2 + \cdots + X_n] = \mathbb{E}[X_1] + \mathbb{E}[X_2] + \cdots + \mathbb{E}[X_n]$$

for example, imagine that we have a bi-graph where each origin node $i \in [n]$ is connected to a destination node $j \in [n]$ with a equal probability among the free nodes, and we want to calculate the expected number of connections where the origin $i$ is the same as $j$. In this example the probability of the second connection p_{2,j} will depend on the first connection p_{1,j} and so on. 

however, the probability of a match is $$\Pr(X_i = j) = \frac{1}{n}$$, hence the expected value of a match in all the connections will be given by the sum of the expected values of each connection, which is equal to $1$.

$$\mathbb{E}[X] = \sum_{i=1}^{n} \mathbb{E}[X_i] = \sum_{i=1}^{n} \frac{1}{n} = 1$$

Even tho, the variables are not independent. 

We will define variance as follows:

$$\text{Var}(X) = \sigma_X^2 = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.$$

We will define the standard deviation $\sigma_X$ as the square root of the variance. We will say that the "mean" is the "typical value" and $\sigma_X$ is viewed as the typical difference between $X$ and $\bar{X}$.

**Syntaxis clarification**: we will use something that is call "Stieljes notation" to denote the expected value of a function $h(X)$ of a random variable $X$ as follows:

$$\mathbb{E}[h(X)] = \int_{-\infty}^{\infty} h(x) \, dF_X(x)$$

where $dF_X(x)$ is nothing more than $f(x)dx$.

Let's define $Z = X + Y$ as the sum of two random variables. If we assume that $X$ and $Y$ are independent the CDF of $Z$ is given by the convolution of the CDFs of $X$ and $Y$:

$$ F_Z(z) = \int_{-\infty}^{\infty} F_X(z - y) \, dF_Y(y) = \int_{-\infty}^{\infty} F_Y(z - x) \, dF_X(x)$$

if both have densities then this can be written as:

$$ f_Z(z) = \int_{-\infty}^{\infty} f_X(z - y) f_Y(y) \, dy = \int_{-\infty}^{\infty} f_Y(z - x) f_X(x) \, dx$$

if we have a set of independent random variables $X_1, X_2, \ldots, X_n$ and we define $S_n = X_1 + X_2 + \cdots + X_n$, then the CDF of $S_{n+1}$ is given by the convolution of the CDF of $S_n$ and $X_{n+1}$. 

If the rv's $X_1, X_2, \ldots, X_n$ are independent, the variance of $S_n=X_1 + X_2 + \cdots + X_n$ is given by the sum of the variances of the individual random variables:

$$\sigma_{S_n}^2 = \sigma_{X_1}^2 + \sigma_{X_2}^2 + \cdots + \sigma_{X_n}^2.$$

If they are IID then $\sigma_{S_n}^2 = n \sigma_X^2$.

It is important to remember that the mean of $S_n$ increases linearly with $n$, but the standard deviation of $S_n$ increases only as $\sqrt{n}$. 

#### Conditional Expectations. 

we will define the conditional expectation as:

$$\mathbb{E}[X|Y] = \int_{-\infty}^{\infty} x \, dF_{X|Y}(x|y)$$

or in the case of discrete random variables:
$$\mathbb{E}[X|Y] = \sum_x x \, \Pr(x|Y=y)$$


There's an interesting property of conditional expectations, which is that the expected value of the conditional expectation is equal to the expected value of the original random variable:

$$\mathbb{E}[\mathbb{E}[X|Y]] = \mathbb{E}[X]$$

The explanation behind this property is that if we consider the conditional expectation as a random variable itself conditioned on $Y$, so if you think about all the Y's that can happen, the expected value of the conditional expectation is equal to the expected value of the original random variable.

$$ \mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]] = \sum_y \mathbb{E}[X|Y=y] \Pr(y) =  \sum_x x \Pr(x|y) \sum_y \Pr(y)  = \mathbb{E}[X]$$

this will give origin to the total expectation theorem, which states that if $X$ and $Y$ are random variables, then
$$\mathbb{E}[X] = \sum_y \mathbb{E}[X|Y=y] \Pr(Y=y)$$

which is a useful tool to calculate the expected value of a random variable $X$ when calculating it directly is difficult, but calculating the conditional expectation $\mathbb{E}[X|Y=y]$ is easier.

### Basic Inequalities

1. Markov's inequality: If $Y$ is a nonnegative random variable. then

$$\Pr(Y \geq y) \leq \frac{\mathbb{E}[Y]}{y} \quad \text{for every } y > 0.$$

An example of this inequality could be that if the average population height is 1.6 meters, then the Markov inequality states that at most half of the population can be taller than 3.2 meters.

2. Chebyshev's inequality: Let be $Z$ a random variable with finite mean $\mathbb{E}[Z]$ and finite variance $\sigma_Z^2$. And lets define $Y = (Z - \mathbb{E}[Z])^2$ Thus $\mathbb{E}[Y] = \sigma_Z^2$. Then by Markov's inequality we have that

$$\Pr((Z - \mathbb{E}[Z])^2 \geq y) \leq \frac{\sigma_Z^2}{y} \quad \text{for every } y > 0.$$


### The law of large numbers

WLLN. for each integer $n \geq 1$. Let $S_n = X_1 + X_2 + \cdots + X_n$ will be the sum of iid rvs with a finite variance. then 


$$ \lim_{n \to \infty} \Pr\left(\left|\frac{S_n}{n} - \mathbb{E}[X_1]\right| > \epsilon\right) = 0 \quad \text{for every } \epsilon > 0.$$

which in other words states that the sample mean $\frac{S_n}{n}$ converges in probability to the expected value $\mathbb{E}[X_1]$ as $n \to \infty$.

### Central limit theorem
Let $X_1, X_2, \ldots$ be a sequence of iid random variables with finite mean $\mu$ and finite variance $\sigma^2$. Let $S_n = X_1 + X_2 + \cdots + X_n$. Then the central limit theorem states that the distribution of the standardized sum $\frac{S_n - n\mu}{\sigma \sqrt{n}}$ converges to the standard normal distribution as $n \to \infty$: 

$$\lim_{n \to \infty} \Pr\left(\frac{S_n - n\mu}{\sigma \sqrt{n}} \leq x\right) = \Phi(x) \quad \text{for every } x \in \mathbb{R},$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

The WLLN tells us that the sample mean converges to the population mean, but the CLT tells us that the distribution of the sample mean converges to a normal distribution as the sample size increases. So one give us a guarantee of convergence and the otherone give us an understanding of the distribution, which also by definition will give us an error estimate of the sample mean. 

$$ X_n \xrightarrow{d} \mathcal{N}(\mu, \sigma^2) \quad \text{as } n \to \infty$$
$$ \sigma_{X_n} = \frac{\sigma}{\sqrt{n}}$$



[//]:1_probability_gallager.md> (References)
[1]: <https://google.com>

[//]:1_probability_gallager.md> (Some snippets)
[//]: # (add an image <img src="" style='height:400pZ;'>)