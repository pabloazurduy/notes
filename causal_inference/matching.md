
*based on [link][1]*
*created on: 2023-05-29 21:46:10*
## Matching and Double Robust Estimator 

In the introduction of the chapter in the Effect there its a quote that explains the matching methodology very well:

Matching is the process of closing back doors between a treatment and an outcome by constructing comparison groups that are similar according to a set of matching variables. Usually this is applied to binary treated/untreated treatment variables, so you are picking a “control” group that is very similar to the group that happened to get “treated


### The problem with the Linear Regression

"Matching consists in choosing a sample in which there is no variation in a variable $W$ closes any open back doors that $W$ sits on, As opposed to regression’s “finding all the variation related to variation in $W$ and removing it closes any open back doors that $W$ sits on.”"

I strongly suggest reading [this article][2] that explains the differences between matching and regression and the limitations of both methods, mainly I will summarize the differences in two:

1. The regression **relies on a functional form**: If this form it's incorrect (non independence, missing covariants, non-linear effects, additive effects) we are subject to misspecification (and therefore to errors in our model)
2. The matching requires [**common support**][3]: This assumes that we have enough similar observations in the control/treatment to make matching.










[//]: <> (References)
[1]: <https://theeffectbook.net/ch-Matching.html>
[2]: <https://www.franciscoyira.com/post/matching-in-r-2-differences-regression/>
[3]: <https://theeffectbook.net/ch-Matching.html#common-support>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)