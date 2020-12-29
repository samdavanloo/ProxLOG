% Solve objection function  min_beta 1/2 || beta - x || + lambda*Omega_LOG(beta) by ADMM
% Penalty is L2 norm

function [beta, P_LOG, time_parallel] = ADMM_L2(x, G_idx, w, lambda, rho, crt, ite_ADMM)

num_G = length(G_idx);
num_v = size(x, 1);
% v,z_bar primal var; u dual var
% beta_ADMM = sum(v,2), target solution
t1_ADMM = zeros(ite_ADMM, 1);
v = zeros(num_v, num_G);
v_bar = zeros(num_v, 1);
z_bar = zeros(num_v, 1);
u = zeros(num_v, 1);
beta = zeros(num_v, 1);
P_LOG = 0;
%  beta_ADMM = zeros(num_G, ite_ADMM); %target solution
%f_ADMM = zeros(ite_ADMM, 1); %objective value
%  v_z_gap = zeros(ite_ADMM, 1); % primal var gap

for k = 1:ite_ADMM
    beta_pre = beta;
    % Update v
    clock_parallel = tic;
    for i = 1:num_G % Can be parallelized
        idx = G_idx{i};
        T = v(idx, i) - v_bar(idx) + z_bar(idx) - u(idx);
        v(idx, i) = T * max(0, 1-lambda*w(i)/rho/norm(T));
    end
    t1_ADMM(k) = toc(clock_parallel);
    beta = sum(v, 2);
    v_bar = beta / num_G;

    % Update z_bar
    z_bar = (x + rho * u + rho * v_bar) / (num_G + rho);

    % Update u
    u = u + v_bar - z_bar;

    % stop criteria
    % P_LOG = sqrt(sum(v.^2,1)) * w;
    % f_ADMM(k) = norm(x-beta).^2 / 2 + lambda * P_LOG;

    % v_z_gap(k) = norm(beta_ADMM(:, k) - z_bar * num_G);

    if k > 1 && norm(beta - beta_pre) < crt
        %  f_ADMM = f_ADMM(1:k);
        P_LOG = sqrt(sum(v.^2, 1)) * w; % LOG penalty
        time_parallel = sum(t1_ADMM);
        break
    elseif k == ite_ADMM
        P_LOG = sqrt(sum(v.^2, 1)) * w; % LOG penalty
        time_parallel = sum(t1_ADMM);

    end

end

end
