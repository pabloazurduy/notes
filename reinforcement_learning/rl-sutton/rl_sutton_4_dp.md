
*based on [link][1]*
*created on: 2026-07-14 22:45:28*
## Dynamic Programming

we will use the bellman equations to iteratively find the value function $v_*$, given a policy $\pi$. We will do this defining a random set of values for the initial value function $v_0$. 

We will update our estimate in the itration $k+1$ using the following formula: 

$$ v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s^{\prime},r} p(s^{\prime},r|s,a)[r + \gamma v_k(s^{\prime})]$$

As you can see in the upper formula, we are using the previous estimate $v_k$ to update our new estimate $v_{k+1}$. we will keep iterating until the update is smaller than a threshold $\theta$. 

In pseudocode we can define the algorithm as follows:


<!-- Code-like block that still allows LaTeX rendering -->
<div style="background: #323436;border:1px solid #897878;padding:12px;border-radius:6px;font-family:SFMono-Regular,Menlo,Monaco,monospace;white-space:pre;overflow:auto;">
Input: policy $\pi$, threshold $\theta$, discount factor $\gamma$
Initialize $v(s)$ arbitrarily for all $s \in S$
Repeat:
    $\Delta \leftarrow 0$
    For each $s \in S$:
        $v_{old} \leftarrow v(s)$
        $v(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s^{\prime},r} p(s^{\prime},r|s,a)\left[r + \gamma v(s^{\prime})\right]$
        $\Delta \leftarrow \max(\Delta, |v_{old} - v(s)|)$
Until $\Delta < \theta$
</div>






[//]: <> (References)
[1]: <https://google.com>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)