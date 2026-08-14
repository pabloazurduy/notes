
*based on [link][1]*
*created on: 2026-08-10 13:40:01*

## Probability summary 

We will define $\Omega$ as the sample space, the set of all sample points for a given experiment. An event $A$ will be as a subset of the sample space $A \subset \Omega$. 

We will define probability as a measuring function $\Pr(\cdot)$ that maps events to real numbers making sure that the three probability axioms hold:

1. $\Pr(\Omega) = 1$.
2. For every event $A$, $\Pr(A) \geq 0$.
3. For any sequence of disjoint events $A_1, A_2, \ldots$,  $\Pr\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \Pr(A_i).$

we can also a few properties 
1. $\Pr(A) = 1 - \Pr(A^c).$
2. $\Pr(A \cup B) = \Pr(A) + \Pr(B) - \Pr(A \cap B).$

bayes formula:
$$\Pr(A|B) = \frac{\Pr(B|A)\Pr(A)}{\Pr(B)}.$$

we will define independence of events as follows: Two events $A$ and $B$ are independent if
$$\Pr(A \cap B) = \Pr(A)\Pr(B).$$

#### random variables

A random variable is a function $X: \Omega \to \mathbb{R}$ that assigns a real number to each outcome in the sample space. we will usually define a random variable in terms of its probability distribution, which describes the likelihood of different outcomes. $X$ will be determined by $P(X=x)$ for all $x \in \mathbb{R}$.

**binomial distribution** If we perform an experiment $n$ times, where each trial has two possible outcomes (success or failure) and the probability of success is $p$, the distribution of a binomial random variable is given by (where k is the number of successes in n trials):

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k},$$

**geometric distribution** If we perform an experiment until we get the first success, where each trial has two possible outcomes (success or failure) and the probability of success is $p$, the distribution of a geometric random variable is given by (where k is the number of trials until the first success):

$$P(X=k) = (1-p)^{k-1} p.$$

**poisson distribution** If we perform an experiment over a fixed interval of time or space, where expected value of successes is $\lambda$ ($\mathbb{E}[X] = \lambda$), the distribution of a Poisson random variable is given by (where k is the number of successes in the interval):
$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}.$$


**exponential distribution** we will define the exponential density function as follows:

$$f(x; \lambda) = \begin{cases}
\lambda e^{-\lambda x} & x \geq 0, \\
0 & x < 0,
\end{cases}$$

we will define the cumulative distribution function (CDF) in general as the integral of the density function from $-\infty$ to $x$:

$$F(x) = \int_{-\infty}^{x} f(t) dt.$$

That means that for example:

$$ P(a \leq X \leq b) = F(b) - F(a)$$

For the uniform distribution the CFD is given by:
$$F(x) = \begin{cases}
0 & x < a, \\
\frac{x-a}{b-a} & a \leq x \leq b, \\
1 & x > b.
\end{cases}$$

In the specific case when we have $\text{Uniform}(a=0, b=1)$ then the CDF $F(x)=x$ 

we will define independence of random variables as follows: Two random variables $X$ and $Y$ are independent based on their joint distribution if
$$f_{X,Y}(x, y) = f_X(x) f_Y(y), \quad \text{for all } x, y \in \mathbb{R}.$$

If two random variables are independent, we also have the following theorems 

1. $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$.

an interesting result is that the CDF of the sum of two independent random variables can be computed as follows:

$$P(X+Y = z) = \sum_{x} P(X=x) P(Y=z-x)$$

#### Expected value and moments 

The expected value of $h(X)$ is defined as follows:

$$\mathbb{E}[h(X)] = \sum_{x} h(x) P(X=x)$$

For continuous random variables, the expected value is defined as:

$$\mathbb{E}[h(X)] = \int_{-\infty}^{\infty} h(x) f_X(x) dx.$$

we will define variance as:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] $$
we have the following properties:

1. $\mathbb{E}[X+c] = \mathbb{E}[X] + c$.
2. $\text{Var}(X+c) = \text{Var}(X)$.
3. $\mathbb{E}[cX] = c\mathbb{E}[X]$.
4. $\text{Var}(cX) = c^2\text{Var}(X)$.
5. $\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ for any two random variables $X$ and $Y$.
6. $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ for any two **independent** random variables $X$ and $Y$.





[//]: <> (References)
[1]: <https://google.com>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)