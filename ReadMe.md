 # ADMM algorithm for solving Proximal operator of the Latent Overlapping Group (LOG) lasso penalty (version 2.0)

This Matlab function contains our ADMM implementation with the sharing scheme to compute proximal operator of the LOG penalty. For more informatio, please refer to [Zhang, D., Liu, Y., Davanloo Tajbakhsh, S. (2020), A first-order optimization algorithm for statistical learning with hierarchical sparsity structure](https://arxiv.org/abs/2001.03322)



## Content

```
.
├── Experiments
│   ├── Cancer // Section 4.2.2
│   │   ├── ADMM_L2.m
│   │   ├── CBCD_L2.m
│   │   ├── Figures
│   │   ├── RBCD_L2.m
│   │   ├── Results
│   │   ├── cancer_FISTA_ADMM.m	// algorithm code
│   │   ├── cancer_FISTA_CBCD.m // algorithm code
│   │   ├── cancer_FISTA_RBCD.m // algorithm code
│   │   ├── data_cancer.mat	// processed source data
│   │   └── process_result.m // code to plot the final result
│   ├── Simulations	// Section 4.1
│   │   ├── Data	// Simulated data
│   │   ├── Figures	// result plots
│   │   ├── Functions // compared algorithms
│   │   ├── Result	// result for all cases
│   │   ├── Result_R // result of R package 'HSM' 
│   │   ├── main_HSM.r // code to run 'HSM'
│   │   └── main_compare.m // main code that do the compare
│   └── Topics // Section 4.2.1
│       ├── Data
│       ├── FISTA_ADMM.m // algorithm code
│       ├── FISTA_CBCD.m // algorithm code
│       ├── FISTA_RBCD.m // algorithm code
│       ├── Figures
│       ├── Functions
│       ├── Results
│       └── process_result.m // code to plot the final result
├── driver.m	//Demo code for calling prox_ADMM
└── prox_ADMM.m //Solver for proximal operator of LOG penalty using ADMM

```



## Call the function

Function:

```
[beta,objval,penaltyval] = prox_ADMM(ancestor,b,iter_max,rho,lambda,w);
```

solves problem
$$
\min_{\beta} \frac{1}{2}||\beta-b||^2 + \lambda \Omega_{\text{LOG}}(\beta)
$$
Required inputs:

- ancestor: cell array that stores the ancestor nodes of each node
- b: vector
- iter_max: maximum number of iteration
- rho : stepsize of ADMM
- lambda: parameter > 0
- w : penalty value $\omega$ for each group



Outputs:

- beta : solution of proximal operator
- objval : objective value of proximal operator
- penaltyval : value of LOG penalty $ \lambda \Omega_{\text{LOG}}(\beta) $



Another way to set ancestor is provided in driver.m. Users can define an $n\times 2$ matrix 'indx_arc' with each row to be a directed edge of DAG. Then  the code below will generate the cell array ancestor.

```matlab
%% Generate index of ancestor for each node
ancestor = cell(n_DAG,1);
for i = 1:n_DAG
    idx_ancestor = indx_arc(:, 2) == i;
    ancestor{i} = indx_arc(idx_ancestor, 1);
    ancestor{i} = unique([i;cell2mat({ancestor{[indx_arc(idx_ancestor, 1); i]}}')]);
end

```









 

