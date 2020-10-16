 # ADMM algorithm for solving Proximal operator of the Latent Overlapping Group (LOG) lasso penalty (version 2.0)

This Matlab function contains our ADMM implementation with the sharing scheme to compute proximal operator of the LOG penalty. For more informatio, please refer to [Zhang, D., Liu, Y., Davanloo Tajbakhsh, S. (2020), A first-order optimization algorithm for statistical learning with hierarchical sparsity structure](https://arxiv.org/abs/2001.03322)



## Content

```
.
├── Compare_Algrhm //Code that compare different methods of proximal operator
├── ReadMe.md
├── driver.m	//Demo code for calling prox_ADMM
└── prox_ADMM.m //Solver for proximal operator of LOG penalty using ADMM

```

Function:

```
[beta,objval,penaltyval] = prox_ADMM(ancestor,b,iter_max,rho,lambda,w);
```

solves problem
$$
\min_{\beta} \frac{1}{2}||\beta-b||^2 + \lambda \Omega_{\text{LOG}}(\beta)
$$
Required input:

- ancestor: cell array that stores the ancestor nodes of each node
- b: vector
- iter_max: maximum number of iteration
- rho : stepsize of ADMM
- lambda: parameter > 0
- w : penalty value $\omega$ for each group



Output:

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









 

