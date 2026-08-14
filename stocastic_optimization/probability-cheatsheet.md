# Probability Cheat Sheet

#### 1. Probability foundations

**Sample space** $\Omega$: all possible outcomes. An **event** $A\subseteq\Omega$. For equally likely finite outcomes,
$$
P(A)=\frac{|A|}{|\Omega|}.
$$

#### Axioms and immediate rules

$$
P(A)\ge0,\qquad P(\Omega)=1,
$$
and for pairwise disjoint events $A_i$,
$$
P\!\left(\bigcup_i A_i\right)=\sum_iP(A_i).
$$

$$
P(\varnothing)=0,\quad 0\le P(A)\le1,\quad A\subseteq B\Rightarrow P(A)\le P(B).
$$

| Concept | Meaning | Rule |
|---|---|---|
| Complement $A^c$ | $A$ does not occur | $P(A^c)=1-P(A)$ |
| Intersection $A\cap B$ | $A$ and $B$ | see conditional rule |
| Union $A\cup B$ | $A$ or $B$, inclusive | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ |
| Difference $A\setminus B$ | $A$, not $B$ | $P(A\setminus B)=P(A)-P(A\cap B)$ |

For three events (the unlabeled sums run over all single events and all pairwise intersections),
$$
P(A\cup B\cup C)=\sum P(A)-\sum P(A\cap B)+P(A\cap B\cap C).
$$
Use the complement for “at least one”: $P(\text{at least one})=1-P(\text{none})$.

De Morgan's laws:
$$
(A\cup B)^c=A^c\cap B^c,\qquad (A\cap B)^c=A^c\cup B^c.
$$

#### 2. Conditional probability, independence, Bayes

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}\quad(P(B)>0),
\qquad P(A\cap B)=P(A\mid B)P(B).
$$

**Chain rule:**
$$
P(A_1\cap\cdots\cap A_n)=P(A_1)\prod_{i=2}^nP(A_i\mid A_1\cap\cdots\cap A_{i-1}).
$$

$A,B$ are **independent** iff
$$
P(A\cap B)=P(A)P(B),
$$
equivalently $P(A\mid B)=P(A)$ when $P(B)>0$. If $X,Y$ are independent and the expectations exist, $E[g(X)h(Y)]=E[g(X)]E[h(Y)]$. Mutually exclusive nonzero-probability events are *not* independent. Pairwise independence need not imply joint independence.

If $B_1,\ldots,B_k$ partition $\Omega$ and each relevant $P(B_i)>0$, then
$$
\boxed{P(A)=\sum_{i=1}^kP(A\mid B_i)P(B_i)}\qquad\text{(law of total probability)}.
$$

### 3. Counting

#### Core rules

- **Addition:** disjoint alternatives: $m+n$.
- **Multiplication:** sequential choices: $m\times n$. Use a decision tree if conditions change.
- **Factorial:** $n!=n(n-1)\cdots1$, $0!=1$.
- **Pigeonhole:** distributing $N$ objects into $k$ boxes guarantees one box has at least $\lceil N/k\rceil$ objects.

#### Arrangements and selections

| Object | Formula |
|---|---:|
| select $k$ ordered elements from $n$ distinct without replacement | $P(n,k)=\frac{n!}{(n-k)!}$ |
| select $k$ unordered elements from $n$ distinct without replacement | $\binom nk=\frac{n!}{k!(n-k)!}$ |
| Permutations of $n$ distinct items | $n!$ |


**Multinomial theorem/count:** allocating $n$ labeled trials into category counts $n_i$, $\sum n_i=n$:
$$
\binom{n}{n_1,\ldots,n_r}=\frac{n!}{\prod_i n_i!}.
$$

**Stars and bars:** nonnegative integer solutions to $x_1+\cdots+x_k=n$: $\binom{n+k-1}{k-1}$. If $x_i\ge1$ and $n\ge k$: $\binom{n-1}{k-1}$. Apply lower bounds $x_i\ge a_i$ by substituting $y_i=x_i-a_i$ and checking feasibility.

**Inclusion–exclusion:**
$$
\left|\bigcup_i A_i\right|=\sum|A_i|-\sum|A_i\cap A_j|+\sum|A_i\cap A_j\cap A_k|-\cdots.
$$
The same inclusion–exclusion formula holds with cardinalities replaced by probabilities. Derangements: $!n=n!\sum_{k=0}^n(-1)^k/k!$, the nearest integer to $n!/e$ for $n\ge1$.

### 4. Random variables and distributions

A random variable (RV) maps outcomes to numbers. For discrete $X$, **PMF** $p_X(x)=P(X=x)$, $\sum_xp_X(x)=1$. For an absolutely continuous $X$, **PDF** $f_X(x)\ge0$, $\int_{-\infty}^{\infty}f_X(x)dx=1$, and $P(a\le X\le b)=\int_a^bf_X(x)dx$; individual points have probability zero.

$$
F_X(x)=P(X\le x).
$$
The **CDF** is nondecreasing and right-continuous, with $\lim_{x\to-\infty}F_X(x)=0$ and $\lim_{x\to\infty}F_X(x)=1$. For an absolutely continuous $X$, $F_X'(x)=f_X(x)$ almost everywhere. The survival function is $S_X(x)=P(X>x)=1-F_X(x)$.

For joint RVs, marginalize: $p_X(x)=\sum_y p_{X,Y}(x,y)$ or $f_X(x)=\int f_{X,Y}(x,y)dy$. Condition using $p_{X\mid Y}(x\mid y)=p_{X,Y}(x,y)/p_Y(y)$ (similarly for densities). Independence means the joint PMF/PDF factors into the marginals.

## 5. Expectation and variance

$$
E[X]=\sum_xxp_X(x)\quad\text{or}\quad\int_{-\infty}^{\infty}xf_X(x)dx.
$$
**LOTUS:** $E[g(X)]=\sum_xg(x)p_X(x)$ or $\int g(x)f_X(x)dx$; no transformation distribution is needed.

If $g$ is convex and expectations exist, **Jensen's inequality** gives
$$
g(E[X])\le E[g(X)],
$$
with the inequality reversed for concave $g$.

$$
E[aX+bY+c]=aE[X]+bE[Y]+c.
$$
Linearity needs no independence. For nonnegative integer $X$, useful tail sum:
$$
E[X]=\sum_{k\ge1}P(X\ge k).
$$
For a nonnegative continuous $X$,
$$
E[X]=\int_0^\infty P(X>t)\,dt.
$$

$$
\operatorname{Var}(X)=E[(X-E[X])^2]=E[X^2]-E[X]^2.
$$
$$
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X),\quad
\operatorname{Cov}(X,Y)=E[XY]-E[X]E[Y].
$$
$$
\operatorname{Var}\!\left(\sum_iX_i\right)=\sum_i\operatorname{Var}(X_i)+2\sum_{i<j}\operatorname{Cov}(X_i,X_j).
$$
Thus variances add when the variables are independent (or merely pairwise uncorrelated).
Independent variables have covariance 0 (when second moments exist); the reverse need not hold. Also,
$$
\operatorname{Cov}(aX+b,cY+d)=ac\operatorname{Cov}(X,Y).
$$
Correlation:
$$
\rho_{X,Y}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}\in[-1,1].
$$

### Conditional expectation

$$
E[X\mid Y=y]=\sum_xxP(X=x\mid Y=y)
$$
For continuous variables, replace the sum with $\int x f_{X\mid Y}(x\mid y)\,dx$. Treat $E[X\mid Y]$ as an RV in $Y$. If $g(Y)$ is known after conditioning, it can be pulled out: $E[g(Y)X\mid Y]=g(Y)E[X\mid Y]$. If $X$ and $Y$ are independent, $E[X\mid Y]=E[X]$.
$$
\boxed{E[X]=E[E[X\mid Y]]}\quad\text{(tower property)},
$$
$$
\boxed{\operatorname{Var}(X)=E[\operatorname{Var}(X\mid Y)]+\operatorname{Var}(E[X\mid Y])}.
$$

### Indicators

$\mathbf1_A=1$ if $A$ occurs, otherwise 0. Then $E[\mathbf1_A]=P(A)$ and $\operatorname{Var}(\mathbf1_A)=P(A)(1-P(A))$. To count occurrences, let $N=\sum_i\mathbf1_{A_i}$; then $E[N]=\sum_iP(A_i)$, even with dependence.

### 6. High-yield distributions

| Distribution (key fact) | Support | Mean | Variance |
|---|---|---|---|
| Bernoulli (single success) | $X\in\{0,1\}$ | $p$ | $p(1-p)$ |
| Binomial ($n$ iid Bernoulli trials) | $P(X=k)=\binom nkp^k(1-p)^{n-k}$, $0\le k\le n$ | $np$ | $np(1-p)$ |
| Geometric (waiting time; memoryless) | $P(X=k)=(1-p)^{k-1}p$, $k\ge1$ | $1/p$ | $(1-p)/p^2$ |
| Negative binomial (trials through $r$ th success) | $P(X=k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}$, $k\ge r$ | $r/p$ | $r(1-p)/p^2$ |
| Hypergeometric (successes in $n$ draws without replacement) | $P(X=k)=\frac{\binom Kk\binom{N-K}{n-k}}{\binom Nn}$ | $np$ | $np(1-p)\frac{N-n}{N-1}$ |
| Poisson (rare-event counts; sums add rates) | $P(X=k)=e^{-\lambda}\lambda^k/k!$, $k\ge0$ | $\lambda$ | $\lambda$ |
| Uniform (continuous) | $[a,b]$, $f=1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Exponential (waiting time; memoryless) | $f(x)=\lambda e^{-\lambda x},x\ge0$ | $1/\lambda$ | $1/\lambda^2$ |

Geometric/exponential memorylessness (nonnegative integer $s,t$ for geometric; nonnegative real $s,t$ for exponential):
$$
P(X>s+t\mid X>s)=P(X>t).
$$

### 7. Transformations, sums, and order statistics

For $Y=aX+b$: $E[Y]=aE[X]+b$, $\operatorname{Var}(Y)=a^2\operatorname{Var}(X)$. If $Y=g(X)$ is one-to-one, monotone, and differentiable,
$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
$$
If several input values map to $y$, sum this expression over all inverse branches.

For independent sums, convolve:
$$
P(X+Y=s)=\sum_xP(X=x)P(Y=s-x),
$$
or, in the continuous case,
$$
f_{X+Y}(s)=\int_{-\infty}^{\infty}f_X(x)f_Y(s-x)\,dx.
$$
Independent normals sum to normal (means and variances add); independent Poissons sum to $\operatorname{Pois}(\sum_i\lambda_i)$. The moment-generating function is $M_X(t)=E[e^{tX}]$; when it exists near $0$, $M_X^{(k)}(0)=E[X^k]$. For independent RVs, $M_{X+Y}(t)=M_X(t)M_Y(t)$.

For iid continuous $X_1,\ldots,X_n$ with CDF $F$, maximum $M$:
$$
P(M\le x)=F(x)^n,\qquad f_M(x)=nF(x)^{n-1}f(x).
$$
Minimum $m$: $P(m>x)=(1-F(x))^n$ and $f_m(x)=n(1-F(x))^{n-1}f(x)$. More generally, the $k$th order statistic has density
$$
f_{X_{(k)}}(x)=\frac{n!}{(k-1)!(n-k)!}F(x)^{k-1}(1-F(x))^{n-k}f(x).
$$
For $n$ iid $U(0,1)$ variables, $X_{(k)}\sim\operatorname{Beta}(k,n+1-k)$ and $E[X_{(k)}]=k/(n+1)$.

### 8. Limits and bounds

1. **Markov** ($X\ge0$, $a>0$): $P(X\ge a)\le E[X]/a$.  
2. **Chebyshev** (finite variance, $t>0$): $P(|X-\mu|\ge t)\le\sigma^2/t^2$.  
3. **Weak LLN:** the sample average of iid finite-variance RVs converges in probability to $\mu$.  
4. **CLT:** for iid RVs with mean $\mu$ and finite, positive variance $\sigma^2$,
$$
\frac{\sum_{i=1}^nX_i-n\mu}{\sigma\sqrt n}\Rightarrow \mathcal N(0,1).
$$
LLN says where averages settle; CLT describes their $1/\sqrt n$-scale fluctuations.

### 9. Classic results

- **Birthday problem:** with $n$ equally likely birthdays and $k\le n$ people,
  $$
  P(\text{no match})=\prod_{i=0}^{k-1}\frac{n-i}{n},\qquad P(\text{at least one match})=1-P(\text{no match}).
  $$
  For $k>n$, the no-match probability is $0$.
- **Coupon collector:** with $n$ equally likely coupon types, $E[T]=nH_n=n\sum_{k=1}^n1/k=n\log n+\gamma n+O(1)$.
- **Fixed points:** a uniformly random permutation of $n$ items has expected number of fixed points $1$ (use indicators).
- **Markov chains:** for transition matrix $P$, a row distribution evolves as $\pi_t=\pi_0P^t$; a stationary distribution satisfies $\pi=\pi P$.
- **Fair gambler's ruin:** a simple symmetric walk on $\{0,1,\ldots,N\}$ starting at $i$ hits $N$ before $0$ with probability $i/N$, and its expected absorption time is $i(N-i)$.

## 10. Interview patterns

- **Symmetry:** equally likely labels/positions have equal probability; avoid unnecessary enumeration.
- **Condition on the first step / state:** if $h(s)$ is the expected remaining time from state $s$, write $h(s)=1+\sum_{s'}P(s\to s')h(s')$. This is the backbone of hitting-time problems.
- **First-step equations:** define the state completely, include boundary conditions such as $h(\text{target})=0$, then solve the linear equations. Self-loops belong in the equation.
- **Deferred decisions:** reveal random choices only when needed; often converts a hard dependency into a simple conditional probability.
- **Coupling or bijection:** pair outcomes to show equality, or map each outcome counted to exactly one counterpart.
- **Check edge cases:** $p=0,1$, $n=1$, dimensions/units, and whether probabilities sum to 1.

## 11. Compact problem-solving checklist

1. State the experiment, sample space, and what is random.
2. Identify whether outcomes are equally likely; distinguish with/without replacement.
3. Translate “and/or/given/at least” into $\cap,\cup,\mid$, complement.
4. Choose a representation: count, condition, indicator, recursion, symmetry, or distribution.
5. Define variables and assumptions before calculating; do not assume independence silently.
6. Use a complement for “at least one,” linearity for expected counts, and conditioning to break dependence.
7. Sanity-check range, limiting cases, and a small example; then communicate the cleanest derivation.
