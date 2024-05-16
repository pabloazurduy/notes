
*notes based on [cs221 AI class][1]*
*created on: 2023-06-24 19:02:29*

## CS221: Search 2 - A*

Definition of Search Problem:

1. $S_{start}$ :  Starting State
2. $Actions(s)$: Possible Actions on state $s$
3. $Cost(s,a)$: action cost 
4. $Succ(s,a)$: successor state 
5. $IsEnd(s)$: reached end state ?

Objective: find the minimum cost path from $s_{start}$ to any $s$ satisfying $IsEnd(s)$

## Learning Costs

### Example: Transportation Problem

<img src="img_cs221/transportation_example.png" style='height:180px;'>

Additionally, if I'm in the state $s$ and $2s>N$ I can't take the "magic tram". Now imagine that I don't know the cost, but I only know the optimum path, now this will be a "learning problem" 

<img src="img_cs221/learning_costs.png" style='height:150px;'>

In this "inverse" problem, we have an optimum path, and we try to infer the cost of the graph. For example, we have some people's path, and we try to identify their cost function. 

|Problem| Formulation|
|-----------------------|----------------------------------------|
|Forward Problem (search)| $Cost(s,a) \rightarrow (a_1, ..., a_k)$|
|Invert Problem (learning)| $(a_1, ..., a_k) \rightarrow Cost(s,a)$|

Given a "initial guess" of costs (for example `walk=3 , tram=2`) we run the search algorithm to get the optimum path

<img src="img_cs221/tweaking_costs.png" style='height:400px;'>





[//]: <> (References)
[1]: <https://www.youtube.com/watch?v=aIsgJJYrlXk&list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)