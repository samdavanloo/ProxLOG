% Solve objection function  min_beta 1/2 || beta - x || + lambda*Omega_LOG(beta) by ADMM
% Penalty is l_infinity norm

function [beta, P_LOG, t_parallel] = ADMM_LI(x, G_idx, w, lambda, rho, crt, ite_ADMM)

num_G = length(G_idx);

% v,z_bar primal var; u dual var
% beta_ADMM = sum(v,2), target solution

v = zeros(num_G, num_G);
v_bar = zeros(num_G, 1);
z_bar = zeros(num_G, 1);
u = zeros(num_G, 1);
beta = zeros(num_G, 1);
P_LOG = 0;
%  beta_ADMM = zeros(num_G, ite_ADMM); %target solution
%  f_ADMM = zeros(ite_ADMM, 1); %objective value
%  v_z_gap = zeros(ite_ADMM, 1); % primal var gap
t_ADMM = zeros(ite_ADMM, 1);
for k = 1:ite_ADMM
    beta_pre = beta;
    clock_parallel = tic;
    % Update v
    for i = 1:num_G % Can be parallelized
        idx = G_idx{i};
        T = v(idx, i) - v_bar(idx) + z_bar(idx) - u(idx);
        v(idx, i) = T - ProjectOntoL1Ball(T, lambda*w(i)/rho);
    end
    t_ADMM(k) = toc(clock_parallel);
    beta = sum(v, 2);
    v_bar = beta / num_G;

    % Update z_bar
    z_bar = (x + rho * u + rho * v_bar) / (num_G + rho);

    % Update u
    u = u + v_bar - z_bar;

    % stop criteria


    %  f_ADMM(k) = norm(x - beta).^2/2 + lambda * P_LOG;

    % v_z_gap(k) = norm(beta_ADMM(:, k) - z_bar * num_G);

    if k > 1 && norm(beta - beta_pre) < crt
        %  f_ADMM = f_ADMM(1:k);
        P_LOG = max(abs(v)) * w; % LOG penalty
        t_parallel = sum(t_ADMM);
        break
    elseif k == ite_ADMM
        P_LOG = max(abs(v)) * w; % LOG penalty
        t_parallel = sum(t_ADMM);
    end

end

end
