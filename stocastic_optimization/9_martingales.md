
*based on [link][1]*
*created on: 2026-08-16 11:49:16*
## Martingales

We will define a martingale as a sequence of $\{Z_n:n\geq1\}$ given that $\mathbb{E}[|Z_n|]\leq\infty$ and 

$$\mathbb{E}[Z_{n}|Z_{n-1}, \dots , Z_1]=Z_{n-1} \text{ for all } n\geq1. \tag{1}$$

The martingale concept comes from gambling, where gamblers were trying to address if a game was "fair", as in, my expected winnings/losings are zero.

Condition $(1)$ might seem not that strict. While achieving an expected value equal to the previous state is not particularly hard. But condition $(1)$ also implies that the past values will not affect my current expected value. This is counterintuitive, given that the past condition on the expected value was conditioned by $Z_{n-k}$ and hence might look like it should affect the next values. 

**Lemma** For a martingale $\{Z_n:n\geq1\}$, for $n > i \geq 0$ we have that 
$$\mathbb{E}[Z_{n}|Z_{i}, \dots , Z_1]=Z_{i} \text{ for all } n > i \geq 0. \tag{2}$$

which also means that
$$\mathbb{E}[Z_n] = \mathbb{E}[Z_i] $$


This lemma may look counterintuitive at first, since it seems that the martingale definition tells us that past values before the current state do not affect the future expected value.

The key point is that the conditioning information is different in $(1)$ and $(2)$. In $(2)$ we are conditioning on *less* information than in $(1)$: we only "know" the values up to $Z_i$, i.e., information up to step $i$, whereas in $(1)$ we know the values up to step $n-1$. 

### Examples of martingales 

1. Zero mean random walk: Let $Z_n=X_1+\dots +X_n$ where $\{X_i; i \geq 1\}$ are IID and zero mean. 

$$
\begin{aligned}
\mathbb{E}[Z_{n}|Z_{n-1}, \dots , Z_1] &= \mathbb{E}[X_n+Z_{n-1}|Z_{n-1}, \dots , Z_1] \\
&=\mathbb{E}[X_n]+Z_{n-1} \\
&=Z_{n-1} 
\end{aligned}
$$

2. Sum of "arbitrary" dependent rv's: Suppose $\{X_i; i \geq 1\}$ satisfy $\mathbb{E}[X_i|X_{i-1}, \dots, X_1] = 0$. Then $\{Z_n; n \geq 1\}$ defined by $Z_n = X_1 + \dots + X_n$ is a martingale.
$$
\begin{aligned}
\mathbb{E}[Z_n|Z_{n-1}, \dots, Z_1] &= \mathbb{E}[X_n + Z_{n-1}|Z_{n-1}, \dots, Z_1] \\
&= \mathbb{E}[X_n|X_{n-1}, \dots, X_1] + Z_{n-1} \\
&= Z_{n-1}
\end{aligned}
$$

3. Let $X_i= U_iY_i$ where $\{U_i; i \geq 1\}$ are IID equiprobable $\pm 1$ The $Y_i$ are non-negative and independent of the $U_i$ but otherwise arbitrary. Then

$$
\begin{aligned}
\mathbb{E}[X_n|X_{n-1}, \dots, X_1] &= \mathbb{E}[U_nY_n|X_{n-1}, \dots, X_1] \\
&= \mathbb{E}[U_n]\mathbb{E}[Y_n|X_{n-1}, \dots, X_1] \\
&= 0
\end{aligned}
$$
Thus $\{Z_n; n \geq 1\}$ where $Z_n = X_1 + \dots + X_n$ is a martingale. where $Y_i$ is **any non-negative random variable**.

4. Product of martingales. Suppose $\{X_i;i\ge1\}$ is a sequence of IID unit mean rv's (i.e., $\mathbb{E}[X_i] = 1 \forall i\ge 1 $ ). Then $\{Z_n; n \geq 1\}$ where $Z_n = X_1 \cdots X_n$ is a martingale.
$$
\begin{aligned}
\mathbb{E}[Z_n|Z_{n-1}, \dots, Z_1] &= \mathbb{E}[X_n Z_{n-1}|Z_{n-1}, \dots, Z_1] \\
&= \mathbb{E}[X_n] Z_{n-1} \\
&= Z_{n-1}
\end{aligned}
$$

5. Special case of product from martingales. Let $X_i$ be IID and equiprobable 2 or 0.  $Z_n = X_1 \cdots X_n$ is a martingale because: 

$$\Pr\{Z_n = 2^n\} = \frac{1}{2^n} = 2^{-n}; \quad \Pr\{Z_n = 0\} = 1 - \frac{1}{2^n}, \quad \mathbb{E}[Z_n]=1$$
the expected value is equal to 1 because:
$$
\begin{aligned} E[Z_n] &=2^nP(Z_n=2^n)+0\cdot P(Z_n=0)\\
&=2^n\cdot2^{-n}\\
&=1.
\end{aligned}
$$

thus $\lim_{n\to \infty} Z_n = 0$ with probability 1, but $\mathbb{E}[Z_n] = 1$ for all $n$. hence $Z_n$ is a martingale with convergence to 0 but expected value equal to 1. 

#### submartigales and supermartingales

we will define a submartingale as (which is a martingale that explodes in value)
$$\mathbb{E}[Z_{n}|Z_{n-1}, \dots , Z_1]\geq Z_{n-1} \text{ for all } n\geq1. \tag{3}$$

Similarly, a supermartingale is defined as (which is a martingale that decreases in value)
$$\mathbb{E}[Z_{n}|Z_{n-1}, \dots , Z_1]\leq Z_{n-1} \text{ for all } n\geq1. \tag{4}$$

A martingale is both a submartingale and a supermartingale.

#### Jensen's inequality

given a convex function $h(x)$ and a random variable $X$, Jensen's inequality states that

$$\mathbb{E}[h(X)] \geq h(\mathbb{E}[X]).$$

**Theorem** if $\{Z_n; n \geq 1\}$ is a martingale and if $\mathbb{E}[|h(Z_n)|] < \infty$ and $h(x)$ is convex, then $\{h(Z_n); n \geq 1\}$ is a submartingale.

### Stopped martingales 

A stopping rule is any type or rule that for a random process the rule only depends on the past values of the process $Z_1, \dots, Z_n$.

We will define a stopping trial $J$ as random variable such that $\{J=n\}$ must by specified  by the values of $Z_1, \dots, Z_n$.

A defective stopping rv rule $J$ is defective if there's a non 0 probability that $\{J = \infty\}$ or $\{J < -\infty\}$. A possible defective stopping trial is a rule in which stopping might never happen. (for example a random walk RW with a single threshold)

A stopped process $\{Z^*_n; n \geq 1\}$ for a possible defective stopping time $J$ on a process $\{Z_n; n \geq 1\}$ satisfies $Z^*_n = Z_{n}$ if $n\leq J$ and $Z^*_n = Z_{J}$ if $n > J$. A stopped process a process that will stop given certain rule, even if the original rv continues. 

For example, a given gambling strategy, where $Z_n$ is the net worth at time $n$, could be modified to stop when $Z_n$ reaches some given value. Then $Z^*_n$ would remain at that value forever after while $Z_n$ follows the original strategy.

**Theorem** If $J$ is a possible defective stopping rule for a martingale (submartingale) $\{Z_n; n \geq 1\}$, then the stopped process $\{Z^*_n; n \geq 1\}$ is also a martingale (submartingale).

**Theorem** (Martingale convergence theorem) Let $\{Z_n; n \geq 1\}$ be a martingale and assume that there is some finite $M$ such that $\mathbb{E}[Z_n^2] \leq M$ for all $n$. Then there is a rv $Z$ such that $Z_n \to Z$ with probability 1.

[//]: <> (References)
[1]: <https://www.youtube.com/watch?v=GwVjWQykCDw&list=PLEEF5322B331C1B98&index=24>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)