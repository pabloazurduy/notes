*based on ["Causal Inference The Mixtape"][1]*
*created on: 2024-11-16 20:00:43*

## Chapter 6: Regression Discontinuity (RDD)

In some situations, the treatment is assigned based on an arbitrary cohort or threshold score, $C_0$ over the "running variable" $X$. Examples include admission scores or penalties from alcohol tests. 

We are never able to observe units with the same value of the running variable $X$ on both groups (Control and Treatment), therefore we can't apply CIA (given that we are not fulfilling the common support assumption). 

We can assume that, if the threshold score was randomly assigned, (or at least independently of the outcome variable) we fulfill the **"continuity assumption"**, that is translated into: "at the cutoff the potential outcomes are continuous".

Having the Continuity assumption, and specification assumption (given that we will need to extrapolate) we are able to estimate LATE (local average treatment effect) on the units right on the cutoff. Assuming LATE is the same for all units, we can estimate the ATE (average treatment effect) for the whole population.

[//]:the_mixtape_summary_32.md> (References)
[1]: <https://mixtape.scunning.com/>
[2]: <https://stats.stackexchange.com/a/55891/274422>
[3]: <https://en.wikipedia.org/wiki/Ramsey_RESET_test>
[4]: <https://www.mattblackwell.org/files/teaching/s05-fisher.pdf>


[//]:the_mixtape_summary_32.md> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)