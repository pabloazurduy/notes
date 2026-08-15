
*based on [link][1]*
*created on: 2026-08-14 11:24:59*



##  Markov Chains 

A markov chain is an integer time process $X_n, n \ge 0$ for which the sample values for each $X_n$ lie in a countable set $S$ and depend on the past only through the most recent rv $X_{n-1}$. More specifically, for all $n \ge 1$ and all $i_0, i_1, \ldots, i_n \in S$ we have

$$\Pr(X_n = i_n | X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = \Pr(X_n = i_n | X_{n-1} = i_{n-1})$$

Furthermore $\Pr(X_n = j | X_{n-1} = i)$ is independent of $n$ and we denote it by $p_{ij}$, which is the transition probability from state $i$ to state $j$. 

The initial state $X_0$ has an arbitrary probability distribution. A finite-state markov chain is a markov chain for which the state space $S$ is finite.

Without loss of generality you can take a random variable $Z_n$ which depends on the last $m$ states $Z_{n-1}, \ldots, Z_{n-m}$ and define a new markov chain $X_n = (Z_n, Z_{n-1}, \ldots, Z_{n-m})$ with a new state definition. 

### classification of states

We define a **walk** as any sequence of states from the Markov chain where there is a non-zero probability of moving from one state to the next.

We define a **path** as **a walk that does not repeat any states**.

Finally, a **cycle** is a walk that starts and ends at the same state, with no other node repeated. It is similar to a path, but since a path cannot repeat any node (including the start and end), a cycle does not qualify as a path.

A state $j$ is **accessible** from state $i$ if there is a walk from state $i$ to state $j$. This is a transitive property, meaning if $i \rightarrow j$ and $j \rightarrow k$, then $i \rightarrow k$.

Two states $i$ and $j$ are **communicating** if $i$ is accessible from $j$ and $j$ is accessible from $i$ denoted $i \leftrightarrow j$. That means $i\rightarrow j$ and $j\rightarrow i$. This property is also transitive, meaning if $i \leftrightarrow j$ and $j \leftrightarrow k$, then $i \leftrightarrow k$.

We will define a **class** $C$ as a non-empty set of states such that every pair of states in $C$ communicates with each other and no state in $C$ communicates with any state not in $C$. 
An important note, the set of states in the class can have transitions to states outside the class, as long as **they don't communicate** (bi-directionally) to a state inside the class. 

We will define a **recurrent state** as a state $i$ that is accessible from all the states that are accessible from it. In other words, if you can reach state $i$ from state $j$, then you can also reach state $j$ from state $i$. A recurrent state is also called a persistent state. A **transient state** is a state that is not recurrent.

An important property of recurrent states is that, if the chain ever enters a recurrent state it will come back with probability 1. Remember, a recurrent state will always have a path back from all the states that are accessible from it.
On the contrary, a transient state, given that has some states that can be access from it but they don't have a path back. hence the chain can leave a transient state and never come back.

#### Theorem 4.2.6 
For a finite-state markov chain, all states in a class are either all recurrent or all transient.

We will define the period of a state $d(i)$ as the greatest common divisor of those values $n$ for which $p_{ii}^{(n)} > 0$. A state is aperiodic if $d(i) = 1$ and periodic if $d(i) > 1$.

#### Theorem 4.2.7
In any finite-state markov chain or countably infinite-state markov chain, all states in a class have the same period.


For a finite Markov chain, an ergodic class is a class that is both recurrent and aperiodic. A Markov chain consisting entirely of one ergodic class is called an **ergodic chain**.

The ergodic chains hav ethe desirable property that $P_{ij}^{n}$ becomes independent of the the starting state $i$ as $n \to \infty$. 

### Matrix representation

The matrix $[P]$ of the transition probabilities of a Markov chain is called a stochastic matrix. A stochastic matrix is a square matrix with non-negative entries and each row sums to 1.

Consider the problem of learning what is the probability of being in state $j$ after $n$ steps. In the case of two steps we can write this probability as 

$$P_{ij}^{(2)} = \sum_{k \in S} P_{ik} P_{kj}$$

considering all the possible intermediate states $k$ that can be reached from $i$. You can see that this is the $i,j$ entry of the product of the matrix $[P]$ with itself. In general, we can write

$$P_{ij}^{(n+m)} = \sum_{k \in S} P_{ik}^{(n)} P_{kj}^{(m)}$$

This is known as the **Chapman-Kolmogorov equation**, and it shows that $[P^{(n+m)}] = [P^{(n)}][P^{(m)}]$, i.e. the $n$-step transition matrix is simply $[P]^n$, the $n$-th power of the one-step transition matrix $[P]$. An efficient approach to compute $[P]^n$ is to calculate first $[P]^2$, then $[P]^4$, then $[P]^8$, and so on, and then we can combine them to get $[P]^n$ in $O(\log n)$ matrix multiplications.

### Steady-state and $[P^n]$ for large $n$

The matrix $[P^n]$ is the $n$-step transition matrix. The entry $P_{ij}^{(n)}$ represents the probability of being in state $j$ after $n$ steps, $\Pr(X_n = j \mid X_0 = i)$. If the past dies out is possible that with $ n \to \infty$, $P_{ij}^n$ becomes independent of the starting state $i$ and $n$. which also means that all rows will tend to have the same values (given that $P_{ij}^n \to P_j = \pi_j$ fixed value). 

Given that the steady state matrix $\pi$ by definition will not have any change if we multiply by the original transition matrix $[P]$, $\pi [P] = \pi$, then it is easier to find the steady state matrix $\pi$ by solving the system of equations $\pi [P] = \pi$ rather than computing $[P^n]$ for large $n$.

A **Steady state vector** (or a steady state distribution) for an M-state markov chaing with transition matrix $[P]$ is a row vector $\pi$ that satisfies. 

$$\pi [P] = \pi, \quad \sum_{i=1}^{M} \pi_i = 1, \quad \pi_i \ge 0, 1 \le i \le M$$

if $\pi$ is taken as the initial probability mass function (PMF) of the nchain at time 0 then the PMF is maintained forever. This doesn't mean that $\pi$ is unique, two different $P$ can converge to different steady state vectors. However, For a finite state markov chain the equation $\pi = \pi[P]$ always has a probability vector solution. 

A **unichain** is a finite state markov chain that contains a single recurrent class plus perhaps some transient states. An **ergodic unichain** is a unichain for which the recurrent class is ergodic. 





[//]: <> (References)
[1]: <https://google.com>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)