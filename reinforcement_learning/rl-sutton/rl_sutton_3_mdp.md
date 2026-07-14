
*based on [link][1]*
*created on: 2026-07-12 19:19:02*

## Chapter 3: Finite Markov Decision Processes

We will define a Markov Decision Process (MDP) as a tuple $(S, A, P, R)$ where:
- $S_t$ represents the state of the environment at time $t$, which is a finite set of states
- $A_t$ represents the action taken at time $t$, which is a finite set of actions
- $R_t$ represents the reward received at time $t$, which is a finite set of rewards

We will define a "trajectory" as a sequence of states, actions, and rewards: 

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, ...$$

In a finite MDP, all of the sets ($S$, $A$, and $R$) are finite. Given that space we will have a distribution over the next state and reward given the current state and action, which we will denote as $P(S_{t+1}, R_{t+1} | S_t, A_t)$ as the transition probablity between states and rewards. We will call this distribution the "dynamics" of the environment.


$$ p(s^{\prime}, r | s, a) = Pr\{S_{t+1} = s^{\prime}, R_{t+1} = r | S_t = s, A_t = a\} $$


We will define the "state transition probability" as the probability of transitioning to state $s^{\prime}$ given that we are in state $s$ and take action $a$:

$$ p(s^{\prime} | s, a) = Pr\{S_{t+1} = s^{\prime} | S_t = s, A_t = a\} = \sum_{r \in R} p(s^{\prime}, r | s, a) $$


we can also compute the expected reward for state $s$ and action $a$ as follows:

$$ r(s, a) = E[R_{t} | S_{t-1} = s, A_{t-1} = a] = \sum_{r \in R} r * \sum_{s^{\prime} \in S} p(s^{\prime}, r | s, a) $$

and finally the expected reward for state $s$ and action $a$ given the next state $s^{\prime}$ as follows:

$$ r(s, a, s^{\prime}) = E[R_{t} | S_{t-1} = s, A_{t-1} = a, S_t = s^{\prime}] = \frac{\sum_{r \in R} r * p(s^{\prime}, r | s, a)}{p(s^{\prime} | s, a)} $$


That comes from the definition of conditional probability: 
$$ p(a | b) = \frac{p(a, b)}{p(b)} $$
$$ r(s, a, s^{\prime}) = E[R_{t} | S_{t-1} = s, A_{t-1} = a, S_t = s^{\prime}] = \sum_{r \in R} r * p(r | s, a, s^{\prime}) = \sum_{r \in R} r * \frac{p(s^{\prime}, r | s, a)}{p(s^{\prime} | s, a)} $$


In general when we place rewards for an agent we want to place them in a way that the reward signal *what the agent needs to achieve* rather than *how we want the agent to achieve it*. For example in chess, we don't want to give +1 for every piece captured,  because it will make the agent to try to capture pieces rather than winning the game. Instead we want to give +1 for winning the game and -1 for losing the game, because that is what we want the agent to achieve.

### Episodes and continuing tasks

We will define an "episode" as a set of states, actions, and rewards that ends in a terminal state. For example in chess, an episode is a game that ends in a win, loss, or draw.

In general:
- An episode will mark a return to the starting state
- The final state will not condition the rewards or actions of the next episode

If there's no final state in a scenario we will call this a "continuing task". For example, in a stock trading scenario, we can keep trading forever and there is no final state.

We have two ways of defining the sum of rewards (or the "objective function"). If the game is finite, we can define the return as the sum of rewards:

$$ G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T $$

where $T$ is the final time step of the episode. while if the game is infinite, we can define the return as the discounted sum of rewards:

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$
$$G_t = R_{t+1} + \gamma G_{t+1}$$

where $\gamma$ is the discount factor, which is a number between 0 and 1. The discount factor determines how much we care about future rewards. If $\gamma = 0$, we only care about immediate rewards. If $\gamma = 1$, we care about all future rewards equally. 

### 3.5 Policy and value functions

we will define a policy as a mapping from states to actions. $\pi(a|s)$, in general terms $\pi$ is the probability of taking action $a$ given that we are in state $s$.

The value function $v_{\pi}(s)$ is the expected return starting from state $s$ and following policy $\pi$. Is denoted as $v_{\pi}(s)$. we can define formally as:

$$ v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s] = E_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s] $$


we call $v_{\pi}(s)$ the **state-value function** for policy $\pi$.

We can define the **action-value function** $q_{\pi}(s, a)$ as the expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter. Formally:

$$ q_{\pi}(s, a) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a] = E_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) | S_t = s, A_t = a] $$

we call $q_{\pi}(s, a)$ the **action-value function** for policy $\pi$.

The values of $v_{\pi}(s)$ and $q_{\pi}(s, a)$ can be estimated via sampling average, if an agent follows a policy $\pi$ many times, and then averages the returns for each state and action. it will be able to estimate the value functions. if we also keep the action taken in each state, we can also estimate the action-value function. We call this method **Monte Carlo methods**. 

As pointed out before, the nature of the MDP allow us to define the value functions recursively. For example in the case of the state value function, we can define it as follows:

$$ v_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')] $$

This equation is called the **Bellman equation**. 

Based on these equations, adding a constant $c$ to every reward affects the value function differently depending on the task type. In a continuing task with $\gamma < 1$, every state's value increases by the constant $\frac{c}{1-\gamma}$, so the optimal policy is unchanged. In an episodic task, adding a constant $c$ to each reward generally changes the value function in a way that can alter the optimal policy. For example, adding a step cost of $-1$ in a maze encourages the agent to find the shortest path to the goal, whereas adding $+1$ per step encourages the longest path.

### 3.6 Optimal policies and value functions

For a given MDP there are many possible policies $\pi$. We say that policy $\pi$ is better than policy $\pi^{\prime}$ if $v_{\pi}(s) \ge v_{\pi^{\prime}}(s)$ for all states $s$. A policy $\pi_{*}$ is optimal if it is at least as good as all other policies. This optimum may not be unique; the optimal value function $v_{*}(s)$ is the value function of any optimal policy $\pi_{*}$:

$$ v_{*}(s) = \max_{\pi} v_{\pi}(s) $$

and also the optimal action-value function:

$$ q_{*}(s, a) = \max_{\pi} q_{\pi}(s, a) = E[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] $$

Because the optimal condition is self-consistent with the value function definition, we can express it without explicitly referring to an optimal policy. These expressions are called the **Bellman optimality equations**:

$$ v_{*}(s) = \max_{a} q_{*}(s, a) = \max_{a} E[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{*}(s')] $$

$$ q_{*}(s, a) = E[R_{t+1} + \gamma \max_{a'} q_{*}(S_{t+1}, a') | S_t = s, A_t = a] = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_{*}(s', a')] $$

As you can observe, these equations do not depend on a specific policy, but rather on the optimal value functions themselves. 

For finite MDPs there is always at least one optimal policy. Solving the Bellman optimality equations yields the optimal value function, from which we can derive an optimal policy. In practice this requires solving a system of equations whose size equals the number of states.

Once $v_{*}(s)$ is known for every state, an optimal policy chooses in each state the action that maximizes the expected return, if we know $q_{*}(s, a)$, the optimal action-value function, we can directly choose the action that maximizes $q_{*}(s, a)$ for each state $s$.

### 3.7 Optimality and Approximation

While solving the Bellman optimality equations is possible for small MDPs, it is rarely used in practice. This is because one or more of the following conditions are often not met:
1. the dynamics of the environment are accurately known
2. we have sufficient computational resources to solve the resulting system of equations
3. the states satisfy the Markov property





[//]:rl_sutton_2.md> (References)   
[1]: <https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf&ved=2ahUKEwi6xJ7YpbKVAxUfhf0HHTyjEKwQFnoECBYQAQ&usg=AOvVaw3bKK-Y_1kf6XQVwR-UYrBY>

[//]:rl_sutton_2.md> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)