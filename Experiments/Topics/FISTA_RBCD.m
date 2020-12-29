% Solve  function  min_ai,D 1/2 sum_i || x-Dai || + lambda*Omega_LOG(a)
% Update ai using Proximal Gradient
% Update D using algorithm mentioned by Jenatton
%
%
% Author : Yin Liu
% History:
%   4/13/2019:
%       - Change function a_FISTA to ADMM, write FISTA part in main script.
%   4/6/2019 :
%       - Generate C code using Coder provided by Matlab
%       - Call a_FISTA since theory of 4/5 is not correct
%       - Only cluster articles of 2011-2015(1846 total) topic.mat is the
%       relative dataset
%   4/5/2019 :
%       - Increase crt to decrease iteration numbers(about 100 right now)
%       - Multiply norm(pinv(D)) to obj func to make it a simple proximal operator and solve it use ADMM
%
%   4/2/2019 :
%       - File Create
%       - Add dynamic plot for each f_FISTA value to ensure step size
%       is appropriate
fprintf('Start running at %s\n ', datestr(now))
currentFile = sprintf('./Results/FISTA_RBCD.mat');
tempFile = sprintf('./Results/FISTA_RBCD_temp.mat');

%% Settings

rng(6);
num_topic = 13;

lambda = 2^-15;
rho = 10;
step_size = 1; %step size of FISTA
crt = 1e-7;
crt_FISTA = 1e-5;
ite_max = 100; %maximum iteration for whole program
ite_FISTA = 500; % iteration for FISTA
ite_RBCD = 500; % iteration for inner ADMM

lambda_FISTA = step_size * lambda; %lambda for FISTA

%% Initialize
load('./Data/topic.mat')
[m, n] = size(X);
f = zeros(ite_max, 1);

grad_prox = zeros(ite_max, 1);
grad_prox_stepsize = zeros(ite_max, 1);
grad_D = zeros(ite_max, 1);
diff_A_D = zeros(ite_max, 1);

A_save = zeros(num_topic, n, ite_max);
D_save = zeros(m, num_topic, ite_max);

P_LOG = zeros(n, 1); %LOG pelnaty
time_real = zeros(ite_max, 1);
time_CPU = zeros(ite_max, 28);

D = rand(m, num_topic);
A = zeros(num_topic, n);

for i = 1:num_topic
    D(:, i) = ProjectOntoSimplex(D(:, i), 1);
end

%% Hierachial structure
G_idx = {1, [1, 2], [1, 3], [1, 4], [1, 5], [1, 2, 6], [1, 2, 7], [1, 3, 8], ...
    [1, 3, 9], [1, 4, 10], [1, 4, 11], [1, 5, 12], [1, 5, 13]};
num_G = length(G_idx);
w = zeros(num_G, 1);

for i = 1:num_G
    w(i) = sqrt(length(G_idx{i}));
end

poolobj = parpool(28)

for ite = 1:ite_max
    spmd, cpu_t = cputime; end
    tic
    A_pre = A;
    D_pre = D;

    %% Update a by FISTA
    parfor i = 1:n
        a = A(:, i);
        x = X(:, i);
        %f_FISTA = zeros(ite_FISTA, 1);
        t = 1;
        y = a;

        for j = 1:ite_FISTA
            grad = D' * (D * a - x);

            a_hat = max(0, y-step_size*grad);
            a_pre = a;

            %% Update prox_Omega(a-(ak-t*grad))
            [a, p_LOG] = RBCD_LI(G_idx, a_hat, lambda_FISTA, w, crt, ite_RBCD);
            %f_FISTA(j) = norm(x - D * a)^2/2 + lambda * p_LOG;
            t_pre = t;
            t = (1 + sqrt(1+4*t^2)) / 2;
            y = a + (t_pre - 1) / t * (a - a_pre);

            if j > 1 && norm(a-a_pre) < crt_FISTA
                %f_FISTA = f_FISTA(1:j);
                break
            end

        end

        A(:, i) = a;
        P_LOG(i) = p_LOG;

    end

    %% Update D
    An = A * A';
    Bn = X * A';

    [D, f_D] = D_update(D, An, Bn, X, A);
    f(ite) = f_D(end) + lambda * sum(P_LOG);

    time_real(ite) = toc;
    spmd, cpu_t = cputime - cpu_t; end
    time_CPU(ite, :) = [cpu_t{:}];

    % Calculate prox gradient of A and gradient of D and diff of A and D
    % per iteration
    grad_prox(ite) = norm(A_pre-A, 'fro');
    grad_prox_stepsize(ite) = norm((A_pre - A)/step_size, 'fro');
    grad_D(ite) = norm((X - D * A)*A', 'fro');
    diff_A_D(ite) = 1 / n * norm(A-A_pre, 'fro') + 1 / m * norm(D-D_pre, 'fro');
    A_save(:, :, ite) = A;
    D_save(:, :, ite) = D;

    fprintf('finish ite %g, f = %d ,time = %s\n', ite, f(ite), datestr(now, 15))

    if mod(ite, 20) == 0
        save(tempFile, 'f', 'A', 'D', 'A_save', 'D_save', 'time_real', 'time_CPU', 'grad_prox', 'grad_prox_stepsize', 'grad_D', 'diff_A_D')
    end
end

[~, idx] = maxk(D, 10);
topic = label(idx);

save(currentFile, 'f', 'A', 'D', 'A_save', 'D_save', 'topic', 'time_real', 'time_CPU', 'grad_prox', 'grad_prox_stepsize', 'grad_D', 'diff_A_D')
