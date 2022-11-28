
*based on [link][1]*
*created on: 2022-11-28 11:21:04*
## Instrumental Variables 

Instrumental variable design consist basically in using a variable that is a random confounder of the treatment as a proxy of a randomized experiment. Imagine that you have a confounder $Annoyance$ that is not measured and introducing some backdoor paths, imagine that is also another variable $Randomization$ that is **unrelated** with the backdoor variable $Annoyance$. Therefore, we could use the $Randomization$ variable as a source of a pseudo random treatment. 

<p align="center">
<img src="img/instrumentalvariables-ivdag-1.png" style='height:100px;align:center;'>
</p>

This is like the opposite as controlling for confounders (a more classical approach in causal inference methodologies) but we "build" a pseudo-randomized experiment using a known source of random influence in the treatment ($Randomization$ variable). 







[//]: <> (References)
[1]: <https://theeffectbook.net/ch-InstrumentalVariables.html>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)