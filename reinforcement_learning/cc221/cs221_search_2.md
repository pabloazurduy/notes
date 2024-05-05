
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

Instead of trying to identify the shortest (cheapest) path in a graph problem, we will now invert this problem from "search" to "learning". In this "inverse" problem, we have an optimum path, and we try to infer the cost of the graph. For example, we have some people's path, and we try to identify their cost function. 








[//]: <> (References)
[1]: <https://www.youtube.com/watch?v=aIsgJJYrlXk&list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)