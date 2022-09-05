## Panel (causal Inference The mixtape)

A panel is a (randomly selected from a population) group of entities observed overtime. The outcome variable will be $Y_{i,t}$ for the unit $i \in N$ on the time $t \in T$. There is a covariate $D_{i,t}$ that represent the status of the treatment, and two other variables $X_i$ and $u_i$ both are different by entity but they both **don't change overtime** the only difference is that we know $X_i$ but we don't observe $u_i$. 

<img src="img/panel_data_dag.png" style='height:600px;'>

#### Assumptions:

1. The treatment $D_{i,t}$ affects the value of the outcome $Y_{i,t}$ but also affects the next week treatment $D_{i,t+1}$
2. $X_i$ and $u_i$ affect both the treatment ($D_{i,t}$) and the outcome($Y_{i,t}$), (this opens a backdoor), therefore the treatment is an endogenous variable when regressing, because $u_i$ (unobserved) will be absorbed by the error $\varepsilon$.
3. **Unobserved heterogeneity**: there is no unobserved variable that varies overtime, in other words, the panel is heterogenic ($u_i, X_i$) but the panel entities **don't change over time** (except for $D,Y$) or if they change that variable don't affect $D,Y$. **This is a very strong assumption**
4. **Past outcomes don't *directly* affect current outcomes** ($Y_{i,t-1}  \bot Y_{i,t}$). This is also a very strong assumption considering things like earnings/health/purchases, anything where $Y_{i,t} \sim Y_{i,t-1}$ 
5. Past treatments do not directly affect current treatments. That means ($Y_{i,t-1}  \bot D_{i,t}$). This is a less strong assumption compared to 3. and 4., but still might be concerning in things such as $Y$ earnings, $D$ education (for example).  
6. Past treatments do not affect present outcomes [not directly at least, but via $D_{i,t}$]($D_{i,t-1}  \bot Y_{i,t}$). This is also a more easy-to-meet assumption. 

Under this 6 assumptions we can estimate the effect of $D \to Y $ using the **Fixed Effect Estimator** (FE) or **pooled ordinary least squares** (POLS).

### Pooled OLS (POLS) - Not Useful - 

Our estimations model will be something like the following linear formula:

$$Y_{i,t} = \delta D_{i,t} + u_i + \varepsilon_{i,t}$$

we will call $u_i$ the _"unobserved heterogeneity"_ (that varies by entity but not over time). and $\varepsilon_{i,t}$ called the "idiosyncratic error" will be the time-varying unobserved factors that determinate $Y_{i,t}$

If we directly run a regression of $Y_{i,t} \sim D_{i,t}$ we will get something like this:

$$Y_{i,t} = \delta D_{i,t} + \eta_{i,t}$$ 

where $\eta_{i,t} = u_i + \varepsilon_{i,t}$ is a composite error. In order to obtain a **consistent estimator** for $\delta$ we will have to hold the following assumption:

$$\eta_{it}\mid D_{i1},D_{i2}, \dots, D_{iT}\big] = E\big[\eta_{it}\mid D_{it}\big] = 0 \quad \text{for $t=1,2,\dots, T$} $$

which means that $u_i \bot D_{i,t}$ This a assumption that is rarely hold in observational studies, the dag presented at the begging also explicitly link $u_i$ with $D_{i,t}$ breaking this premise.

### Fixed Effects (FE)
The idea is to make a transformation over the model $Y_{i,t} = \delta D_{i,t} + u_i + \varepsilon_{i,t}$ so we can remove the $u_i$ variable from it. Using the following transformations should be _numerically equivalent_ to the regression of $Y_{i,t} \sim D_{i,t} $ and **unit specific dummy variables** ($u_i$).

$$\widetilde{Y_{i,t}} = Y_{i,t} - \overline{Y_{i}}$$
$$\widetilde{D_{i,t}} = D_{i,t} - \overline{D_{i}}$$

where:
$$\overline{D}_i \equiv \dfrac{1}{T}\sum_{t=1}^TD_{it}$$
$$\bar{Y}_i \equiv \dfrac{1}{T} \sum_{t=1}^T Y_{it}$$

In other words adding this transformation is equivalent of adding a set of $u_i$ dummy variables to control for the _**Unobserved heterogeneity**_. From The Effect Book:

> Fixed effects is a method of controlling for all variables, whether they’re observed or not, as long as they stay constant within some larger category.

The idea is that, if we control for the larger category therefore we will be controlling that is constant over that category. For example, we want to study the effect of $D$ (some new policy in a Country) in $Y$ (GDP) but we have some unobserved variables $u_i$ that affect $D$ and $Y$ (unobserved heterogeneity) such as culture, geography, industries. We can control for it measuring many countries ("larger category") to control by $u_i$. **Note**: again this will not consider other variables that vary overtime $u_{i,t}$ such as Education, Climate change, etc. 

### Known issues with Fixed Effects (FE)
1. **Fixed effects cannot solve reverse causality**: If there is an inverse relationship between $Y_{i,t} \to D_{i,t} $ the FE estimator will not work, for example in $Y$:crime rates and $D$: police investment. Is it possible that our regression show that an increase in police investment increase the crime rates too. (but there is a reverse causality problem, also called _simultaneity bias_).

2. **Fixed Effects cannot address time-variant unobserved heterogeneity**: that means that our unobserved heterogeneity depends also on the time $u_{i,t}$. In that case FE estimator will also be biased by an omitted variable. 