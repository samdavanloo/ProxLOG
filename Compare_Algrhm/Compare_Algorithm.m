% Note:
%   - With backtracking  PGM and FISTA


% % clear
% % clc
rng(8);

%% setting
lambda = 0.5;
rho_ADMM = 0.8;
stepsize_PGM = 1; %step size for PGM and FISTA
stepsize_FISTA = 1;
iteration_BCD = 1000;
iteration_ADMM = 4000;
iteration_PGM = 6000;
iteration_FISTA = 6000;

iteration_RBCD = 1000;

maxNumCompThreads(1);

%% Generate index matrix
% % % % Two paths
% % % M = zeros(101);
% % % for i = 1:51
% % %     M(1:i, i) = 1;
% % % end
% % % for i = 52:101
% % %     M(52:i, i) = 1;
% % %     M(1, i) = 1;
% % % end

% % % %Two layer tree
% % % M = eye(101);
% % % M(1, :) = 1;

% % % % % Binary tree(7 layer, 127 nodes)
% % % M = eye(127);
% % % for i = 1:127
% % %     x = floor(i/2);
% % %     while x > 0
% % %                 M(x, i) = 1;
% % %
% % %         x = floor(x/2);
% % %     end
% % % end


% DAG
G = {1, [1, 2], [3], [3, 4], [3, 5], [3, 4, 5, 6], [1:7], [3, 4, 5, 6, 8]};
M = 0;
for g = 1:length(G)
    M(G{g}, g) = 1;
end

% % % %% Random DAG with 100 nodes
% % % % generate arcs
% % % n_DAG = 40;
% % % indx_arc = randi([1, n_DAG], 400, 2);
% % % indx_arc = sort(indx_arc, 2);
% % % indx_arc = unique(indx_arc, 'row');
% % % indx_identical = indx_arc(:, 1) == indx_arc(:, 2);
% % % indx_arc(indx_identical, :) = [];
% % %
% % % % generate  matrix
% % % M = zeros(n_DAG);
% % % ancestor = cell(n_DAG, 1);
% % % for i = 1:n_DAG
% % %     idx_ancestor = indx_arc(:, 2) == i;
% % %     ancestor{i} = indx_arc(idx_ancestor, 1);
% % %     ancestor{i} = unique(cell2mat({ancestor{[indx_arc(idx_ancestor, 1); i]}}'));
% % %
% % % end
% % %
% % % for i = 1:n_DAG
% % %     M(ancestor{i}, i) = 1;
% % %     M(i, i) = 1;
% % % end

%%
n_para = size(M, 1);

indx = {n_para, 1};
for i = 1:n_para
    indx{i} = find(M(:, i));
end

w = sqrt(sum(M, 1))';

y = randn(n_para, 10);

beta_BCD = zeros(n_para, iteration_BCD*n_para, 10);
t_BCD = zeros(iteration_BCD*n_para, 1, 10);
f_BCD = zeros(iteration_BCD*n_para, 1, 10);

beta_ADMM = zeros(n_para, iteration_ADMM, 10);
[t1_ADMM, t2_ADMM, f_ADMM, dual_ADMM, x_minus_z_ADMM] = deal(zeros(iteration_ADMM, 1, 10));

beta_PGM = zeros(n_para, iteration_PGM, 10);
[t_PGM, f_PGM] = deal(zeros(iteration_PGM, 1, 10));

beta_FISTA = zeros(n_para, iteration_FISTA, 10);
[t_FISTA, f_FISTA] = deal(zeros(iteration_FISTA, 1, 10));

beta_RBCD = zeros(n_para, iteration_BCD*n_para, 10);
t_RBCD = zeros(iteration_BCD*n_para, 1, 10);
f_RBCD = zeros(iteration_BCD*n_para, 1, 10);

%% opt for 10 experiments

for i = 1:10
    [beta_BCD(:, :, i), t_BCD(:, :, i), f_BCD(:, :, i), ~] ... ,
        = BCD_func(indx, y(:, i), lambda, w, iteration_BCD);

    [beta_ADMM(:, :, i), t1_ADMM(:, :, i), t2_ADMM(:, :, i), f_ADMM(:, :, i), dual_ADMM(:, :, i), x_minus_z_ADMM(:, :, i)] = ... ,
        ADMM_func(indx, y(:, i), iteration_ADMM, rho_ADMM, lambda, w);

    [beta_PGM(:, :, i), t_PGM(:, :, i), f_PGM(:, :, i)] = ... ,
        PGM_func(indx, y(:, i), iteration_PGM, lambda, w, 1);

    [beta_RBCD(:, :, i), t_RBCD(:, :, i), f_RBCD(:, :, i)] ... ,
        = RBCD_func(indx, y(:, i), lambda, w, iteration_RBCD);

    [beta_FISTA(:, :, i), t_FISTA(:, :, i), f_FISTA(:, :, i)] = ... ,
        FISTA_func(indx, y(:, i), iteration_FISTA, lambda, w, 1);
end

%% Calculate residual of beta

beta_opt = beta_BCD(:, end, :);

temp = beta_BCD - beta_opt;
beta_res_BCD_avg = mean(sqrt(sum(temp.^2, 1)), 3);

temp = beta_ADMM - beta_opt;
beta_res_ADMM_avg = mean(sqrt(sum(temp.^2, 1)), 3);

temp = beta_PGM - beta_opt;
beta_res_PGM_avg = mean(sqrt(sum(temp.^2, 1)), 3);

temp = beta_FISTA - beta_opt;
beta_res_FISTA_avg = mean(sqrt(sum(temp.^2, 1)), 3);

temp = beta_RBCD - beta_opt;
beta_res_RBCD_avg = mean(sqrt(sum(temp.^2, 1)), 3);

%% average time over 10 experiments
t_BCD_avg = sum(t_BCD, 3) / 10;

t1_ADMM_avg = sum(t1_ADMM, 3) / 10;
t2_ADMM_avg = sum(t2_ADMM, 3) / 10;
%x_minus_z_ADMM_avg = sum(x_minus_z_ADMM, 3) / 10;

t_PGM_avg = sum(t_PGM, 3) / 10;

t_FISTA_avg = sum(t_FISTA, 3) / 10;

t_RBCD_avg = sum(t_RBCD, 3) / 10;

%% calculate time
time_BCD_avg = cumsum(t_BCD_avg);
time_ADMM_avg = cumsum(t1_ADMM_avg+t2_ADMM_avg);
time_PGM_avg = cumsum(t_PGM_avg);
time_FISTA_avg = cumsum(t_FISTA_avg);

time_RBCD_avg = cumsum(t_BCD_avg);

% % % %calculate residual of beta
% % %
% % % beta_opt_avg = beta_BCD_avg(:, end);
% % % temp = beta_BCD_avg - repmat(beta_opt_avg, 1, length(beta_BCD_avg(1, :)), 1);
% % % beta_res_BCD_avg = sqrt(sum(temp.^2, 1))';
% % %
% % % temp = beta_ADMM_avg - repmat(beta_opt_avg, 1, length(beta_ADMM_avg(1, :)), 1);
% % % beta_res_ADMM_avg = sqrt(sum(temp.^2, 1))';
% % %
% % % temp = beta_PGM_avg - repmat(beta_opt_avg, 1, length(beta_PGM_avg(1, :)), 1);
% % % beta_res_PGM_avg = sqrt(sum(temp.^2, 1))';
% % %
% % % temp = beta_FISTA_avg - repmat(beta_opt_avg, 1, length(beta_FISTA_avg(1, :)), 1);
% % % beta_res_FISTA_avg = sqrt(sum(temp.^2, 1))';
% % %
% % % temp = beta_RBCD_avg - repmat(beta_opt_avg, 1, length(beta_RBCD_avg(1, :)), 1);
% % % beta_res_RBCD_avg = sqrt(sum(temp.^2, 1))';

%% Plot
figure(1)
semilogy(beta_res_ADMM_avg, '-o', 'MarkerIndices', 1:500:length(beta_res_ADMM_avg), 'MarkerSize', 10, 'LineWidth', 2);
hold on
%semilogy(beta_res_BCD_avg, '-+', 'MarkerIndices', 1:5000:length(beta_res_BCD_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(beta_res_FISTA_avg, 'LineWidth', 2);
%semilogy(beta_res_RBCD_avg, '-s', 'MarkerIndices', 1:5000:length(beta_res_RBCD_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(beta_res_PGM_avg, '-^', 'MarkerIndices', 1:1000:length(beta_res_PGM_avg), 'LineWidth', 2, 'MarkerSize', 10);

legend('ADMM', 'ACC-PGM', 'PGM');
xlabel('iteration k')
ylabel('||\beta_k-\beta^*||_2')
hold off


set(gca, ...
    'Units', 'normalized', ...
    'FontUnits', 'points', ...
    'FontWeight', 'normal', ...
    'FontSize', 11, ...
    'FontName', 'Arial')
set(gcf, ...
    'Units', 'inches', ...
    'Position', [0, 0, 4, 3])

box off

figure(2)

semilogy(time_ADMM_avg, beta_res_ADMM_avg, '-o', 'MarkerIndices', 1:300:length(beta_res_ADMM_avg), 'MarkerSize', 10, 'LineWidth', 2);
hold on

semilogy(time_FISTA_avg, beta_res_FISTA_avg, 'LineWidth', 2);

semilogy(time_PGM_avg, beta_res_PGM_avg, '-^', 'MarkerIndices', 1:800:length(beta_res_PGM_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(time_BCD_avg, beta_res_BCD_avg, '-+', 'MarkerIndices', 1:10000:length(beta_res_BCD_avg), 'LineWidth', 2, 'MarkerSize', 10)
semilogy(time_RBCD_avg, beta_res_RBCD_avg, '-s', 'MarkerIndices', 1:10000:length(beta_res_RBCD_avg), 'LineWidth', 2, 'MarkerSize', 10)

xlabel('time')
ylabel('||\beta_k-\beta^*||_2')
legend('ADMM', 'ACC-PGM', 'PGM', 'C-BCD', 'R-BCD');
hold off


set(gca, ...
    'Units', 'normalized', ...
    'FontUnits', 'points', ...
    'FontWeight', 'normal', ...
    'FontSize', 11, ...
    'FontName', 'Arial')
set(gcf, ...
    'Units', 'inches', ...
    'Position', [0, 0, 4, 3])

box off
