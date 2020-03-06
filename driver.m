% *************************************************************************

% Description: Demo code for using ADMM to solve LOG penalty
% Notes:


% Input:
%   Arcs of DAG

% Output:
%   none

% Other m-files required: prox_ADMM.m
%   none

% -------------  LOG --------------


% : First created
% ---------------------------------


% Author: Liu, Yin
% Email: liu.6630 at osu(dot)edu
% Created with MATLAB ver.: 9.7.0.1296695 (R2019b) Update 4

% *************************************************************************


rng(8)
%% Generate random DAG with 40 nodes
n_DAG = 40;
indx_arc = randi([1, n_DAG], 400, 2);
indx_arc = sort(indx_arc, 2);
indx_arc = unique(indx_arc, 'rows');
indx_identical = indx_arc(:, 1) == indx_arc(:, 2);
indx_arc(indx_identical, :) = [];

%% Generate index of ancestor for each node
ancestor = cell(n_DAG,1);
for i = 1:n_DAG
    idx_ancestor = indx_arc(:, 2) == i;
    ancestor{i} = indx_arc(idx_ancestor, 1);
    ancestor{i} = unique([i;cell2mat({ancestor{[indx_arc(idx_ancestor, 1); i]}}')]);
end


%% Generate y and w
b = randn(n_DAG,1);
w = sqrt(cellfun('length',ancestor));   % penalty w for each group

%% Call prox_ADMM
iter_max = 1000;
rho = 0.9;
lambda = 0.1;

[beta,objval,penaltyval] = prox_ADMM(ancestor,b,iter_max,rho,lambda,w);