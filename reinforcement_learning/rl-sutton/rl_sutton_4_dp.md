
*based on [link][1]*
*created on: 2026-07-14 22:45:28*
## Dynamic Programming

we will use the bellman equations to iteratively find the value function $v_*$, given a policy $\pi$. We will do this defining a random set of values for the initial value function $v_0$. 

We will update our estimate in the iteration $k+1$ using the following formula: 

$$ v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s^{\prime},r} p(s^{\prime},r|s,a)[r + \gamma v_k(s^{\prime})]$$

As you can see in the upper formula, we are using the previous estimate $v_k$ to update our new estimate $v_{k+1}$. we will keep iterating until the update is smaller than a threshold $\theta$. 

In pseudocode we can define the algorithm as follows:

$$
\begin{aligned}
& \textbf{Algorithm: } \text{Iterative Policy Evaluation} \\
& \text{Input: policy } \pi, \text{ threshold } \theta \\
& \text{Initialize } V(s) \leftarrow 0 \text{ for all } s \in \mathcal{S} \\
& \textbf{Repeat:} \\
& \quad \Delta \leftarrow 0 \\
& \quad \textbf{For each } s \in \mathcal{S}: \\
& \qquad v \leftarrow V(s) \\
& \qquad V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma V(s')] \\
& \qquad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\
& \textbf{Until } \Delta < \theta
\end{aligned}
$$


Similar to Value function we can also apply an iterative policy evaluation but for the action-value function $q_{\pi}(s, a)$. The update rule for the action-value function is as follows:

$$ q_{k+1}(s, a) = \sum_{s^{\prime},r} p(s^{\prime},r|s,a)[r + \gamma \sum_{a^{\prime}} \pi(a^{\prime}|s^{\prime})q_k(s^{\prime}, a^{\prime})]$$

### 4.2 Policy Improvement

After using DP to evaluate a policy $\pi$, we can wonder if the policy that we select is or not optimal. we can check if the policy is optimal by using the **policy improvement theorem**. We will say a policy $\pi^{\prime}$ is **better** than another policy $\pi$ if the action-value function for some action $a$ is greater than the value function for one state $s$. 

$$ q_{\pi^{\prime}}(s, \pi^{\prime}(s)) \ge v_{\pi}(s) $$

The theorem states that if the above condition holds for all possible states then the policy $\pi^{\prime}$ is optimal. 




[//]: <> (References)
[1]: <https://google.com>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)