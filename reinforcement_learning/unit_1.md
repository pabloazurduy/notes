
*based on [link][1]*
*created: 2023-04-16 11:31:42*
## Unit 1 


we define the RL loop as $S_0, A_0, R_1, S_1$ where:

1. $S_0$ : current and base state 
2. $A_0$ : action 
3. $R_1$ : reward (based on the action)
4. $S_1$ : Next State

We are gonna call `Observation` to a partial description of the state $O_i$. On the contrary, `State` $S_i$ it's a complete description of the state of the world.

We will define the expected reward as 
$R(\tau) = \sum_{k=0}^{\infin} r_{t+k+1} $ where $\tau = (S_i,A_i)_{i}$ 

### MDP a more formal approach
Considering a stochastic process ${S_i,A_i,R_i}$ we will say that it follows the Markov assumption if the information of all previous states is contained in the current state and it's 





[//]: <> (References)
[1]: <https://huggingface.co/deep-rl-course/unit1/rl-framework?fw=pt>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)