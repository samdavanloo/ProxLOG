% *************************************************************************

% Description: Code for experiment 4.1. Compare ADMM, ACC-PGM, RGM, C-BCD, R_BCD and HSM on 6 different cases
% Notes:


% Input:qq
%   none

% Output:
%   none

% Other m-files required:
%   ADMM_func.m
%   FISTA_func.m
%   PGM_func.m
%   BCD_func.m
%   RBCD_func.m

% -------------  LOG --------------


%   2020-10-03: Re-write for Github publish
% ---------------------------------


% Author: Liu, Yin
% Email: liu.6630 at osu(dot)edu
% Created with MATLAB ver.: 9.7.0.1296695 (R2019b) Update 4

% *************************************************************************
maxNumCompThreads(1); % confine to single thread

rng(8);

type = 6; % select DAG type
update = 1; % 1 if use all algs. 0 only ADMM

%% setting
lambda = 0.1;
%rho_ADMM = 0.5;
stepsize_PGM = 1; % step size for PGM
stepsize_FISTA = 1; % step size for FISTA
iteration_BCD = 1000;
iteration_RBCD = 1000;

iteration_ADMM = 3000;
iteration_PGM = 3000;
iteration_FISTA = 3000;

n_exp = 10;

%% Generate index of groups
switch type
    case 1
        % two layer tree
        n_para = 101;
        index_group = cell(n_para, 1);
        index_group{1} = 1;
        index_group(2:end) = mat2cell([ones(1, n_para-1); 2:n_para]', ones(1, n_para-1));

        rho_ADMM = 20;

    case 2
        % one root two paths
        n_para = 101;
        index_group = cell(n_para, 1);
        for i = 1:51
            index_group{i} = [1:i];
        end
        for i = 52:101
            index_group{i} = [1, 52:i];
        end

        rho_ADMM = 1;

    case 3
        % Binary tree
        n_para = 127;
        index_group = cell(n_para, 1);
        M = eye(127);
        for i = 1:127
            x = floor(i/2);
            while x > 0
                M(x, i) = 1;

                x = floor(x/2);
            end
        end

        for i = 1:n_para
            index_group{i} = find(M(:, i));
        end

        rho_ADMM = 10;
        clear M
    case 4 % Reverse binary tree
        n_para = 127;
        temp = repmat([65:127], 2, 1);
        index_arc = [[1:126]', temp(:)];

        index_group = cell(n_para, 1);

        for i = 1:n_para
            index_ancestor = index_arc(:, 2) == i;
            temp = index_arc(index_ancestor, 1);
            index_group{i} = unique([index_group{temp}]);
            index_group{i} = [index_group{i}, i];

        end
        clear index_arc index_ancestor

        rho_ADMM = 0.5;

    case 5 % Asymmetric tree
        n_para = 201;
        index_group = cell(n_para, 1);
        for i = 1:100
            temp = [1:i];
            index_group{i} = temp;
        end

        for i = 101:201
            temp = [1, i];
            index_group{i} = temp;
        end

        rho_ADMM = 0.5;

    case 6 % Random DAG with 100 node and 98 arcs
        n_para = 100;
        n_arc = 100;

        index_arc = randi([1, n_para], n_arc, 2);
        index_arc = sort(index_arc, 2);
        index_arc = unique(index_arc, 'row');
        index_identical = index_arc(:, 1) == index_arc(:, 2);
        index_arc(index_identical, :) = [];
        clear index_identical

        % generate  index of group
        index_group = cell(n_para, 1);

        for i = 1:n_para
            index_ancestor = index_arc(:, 2) == i;
            temp = index_arc(index_ancestor, 1);
            index_group{i} = unique([index_group{temp}]);
            index_group{i} = [index_group{i}, i];

        end
        clear index_arc index_ancestor

        rho_ADMM = 5;

end

clear i ans temp

w = sqrt(cellfun(@length, index_group));

y = randn(n_para, n_exp);

%% Pre-allocate
if update == 1
    beta_BCD = zeros(n_para, iteration_BCD, n_exp);
    t_BCD = zeros(iteration_BCD, 1, n_exp);
    f_BCD = zeros(iteration_BCD, 1, n_exp);

    beta_RBCD = zeros(n_para, iteration_BCD, n_exp);
    t_RBCD = zeros(iteration_BCD, 1, n_exp);
    f_RBCD = zeros(iteration_BCD, 1, n_exp);

    beta_PGM = zeros(n_para, iteration_PGM, 10);
    [t_PGM, f_PGM] = deal(zeros(iteration_PGM, 1, 10));

    beta_FISTA = zeros(n_para, iteration_FISTA, 10);
    [t_FISTA, f_FISTA] = deal(zeros(iteration_FISTA, 1, 10));
end
beta_ADMM = zeros(n_para, iteration_ADMM, n_exp);
[t1_ADMM, t2_ADMM, f_ADMM, dual_ADMM, x_minus_z_ADMM] = deal(zeros(iteration_ADMM, 1, n_exp));

%% opt for n_exp experiments

for i = 1:n_exp
    if update == 1
        [beta_BCD(:, :, i), t_BCD(:, :, i), f_BCD(:, :, i), ~] ... ,
            = BCD_func(index_group, y(:, i), lambda, w, iteration_BCD);

        [beta_RBCD(:, :, i), t_RBCD(:, :, i), f_RBCD(:, :, i)] ... ,
            = RBCD_func(index_group, y(:, i), lambda, w, iteration_BCD);
        [beta_PGM(:, :, i), t_PGM(:, :, i), f_PGM(:, :, i)] = ... ,
            PGM_func(index_group, y(:, i), iteration_PGM, lambda, w, stepsize_PGM);

        [beta_FISTA(:, :, i), t_FISTA(:, :, i), f_FISTA(:, :, i)] = ... ,
            FISTA_func(index_group, y(:, i), iteration_FISTA, lambda, w, stepsize_FISTA);
    end

    [beta_ADMM(:, :, i), t1_ADMM(:, :, i), t2_ADMM(:, :, i), f_ADMM(:, :, i), dual_ADMM(:, :, i), x_minus_z_ADMM(:, :, i)] = ... ,
        ADMM_func(index_group, y(:, i), iteration_ADMM, rho_ADMM, lambda, w);

end

%% average time over experiments
n_core = 1; % Potential to parallelize
t_BCD_avg = sum(t_BCD, 3) / n_exp;
t_RBCD_avg = sum(t_RBCD, 3) / n_exp;

t_PGM_avg = sum(t_PGM, 3) / n_exp;
t_FISTA_avg = sum(t_FISTA, 3) / n_exp;

t1_ADMM_avg = sum(t1_ADMM, 3) / n_exp / n_core;
t2_ADMM_avg = sum(t2_ADMM, 3) / n_exp;
%x_minus_z_ADMM_avg = sum(x_minus_z_ADMM, 3) / 10;


% calculate time
time_BCD_avg = cumsum(t_BCD_avg);
time_ADMM_avg = cumsum(t1_ADMM_avg+t2_ADMM_avg);
time_RBCD_avg = cumsum(t_RBCD_avg);

time_PGM_avg = cumsum(t_PGM_avg);
time_FISTA_avg = cumsum(t_FISTA_avg);

% Calculate optimality gap of obj value based on lowest value among all
% algorithms

f_opt = zeros(1, 1, n_exp);

for i = 1:n_exp
    [f_opt(1, 1, i), win(i)] = min([f_PGM(end, :, i), f_BCD(end, :, i), f_ADMM(end, :, i), f_FISTA(end, :, i), f_RBCD(end, :, i)]);
end


temp = f_BCD - f_opt;
temp = abs(temp);
for i = 1:n_exp
    temp(:, :, i) = temp(:, :, i) / norm(f_opt(:, :, i));
end
f_res_BCD_avg = mean(temp, 3);

temp = f_ADMM - f_opt;
temp = abs(temp);
for i = 1:n_exp
    temp(:, :, i) = temp(:, :, i) / norm(f_opt(:, :, i));
end
f_res_ADMM_avg = mean(temp, 3);

temp = f_PGM - f_opt;
temp = abs(temp);
for i = 1:n_exp
    temp(:, :, i) = temp(:, :, i) / norm(f_opt(:, :, i));
end
f_res_PGM_avg = mean(temp, 3);

temp = f_FISTA - f_opt;
temp = abs(temp);
for i = 1:n_exp
    temp(:, :, i) = temp(:, :, i) / norm(f_opt(:, :, i));
end
f_res_FISTA_avg = mean(temp, 3);

temp = f_RBCD - f_opt;
temp = abs(temp);
for i = 1:n_exp
    temp(:, :, i) = temp(:, :, i) / norm(f_opt(:, :, i));
end
f_res_RBCD_avg = mean(temp, 3);


figure(1)
semilogy(f_res_ADMM_avg, '-o', 'MarkerIndices', 1:500:length(f_res_ADMM_avg), 'MarkerSize', 10, 'LineWidth', 2);
hold on
%semilogy(f_res_BCD_avg, '-+', 'MarkerIndices', 1:5000:length(f_res_BCD_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(f_res_FISTA_avg, 'LineWidth', 2);
%semilogy(f_res_RBCD_avg, '-s', 'MarkerIndices', 1:5000:length(f_res_RBCD_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(f_res_PGM_avg, '-^', 'MarkerIndices', 1:500:length(f_res_PGM_avg), 'LineWidth', 2, 'MarkerSize', 10);


legend('ADMM', 'ACC-PGM', 'PGM', 'BCD', 'RBCD');
xlabel('iteration k')
ylabel('$\frac{\|f_k-f^*||_2}{\|f^*\|_2}$', 'Interpreter', 'Latex')
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

semilogy(time_ADMM_avg, f_res_ADMM_avg, '-o', 'MarkerIndices', 1:500:length(f_res_ADMM_avg), 'MarkerSize', 10, 'LineWidth', 2);
hold on

semilogy(time_FISTA_avg, f_res_FISTA_avg, 'LineWidth', 2);

semilogy(time_PGM_avg, f_res_PGM_avg, '-^', 'MarkerIndices', 1:500:length(f_res_PGM_avg), 'LineWidth', 2, 'MarkerSize', 10);
semilogy(time_BCD_avg, f_res_BCD_avg, '-+', 'MarkerIndices', 1:10000:length(f_res_BCD_avg), 'LineWidth', 2, 'MarkerSize', 10)
semilogy(time_RBCD_avg, f_res_RBCD_avg, '-s', 'MarkerIndices', 1:10000:length(f_res_RBCD_avg), 'LineWidth', 2, 'MarkerSize', 10)

xlabel('time (seconds)')
ylabel('$\frac{\|f_k-f^*||_2}{\|f^*\|_2}$', 'Interpreter', 'Latex')
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

%% Add HSM Result
switch type
    case 1
       
        temp = readmatrix('Result_R/objval_two_tree.csv');
        temp(:, 1) = [];
        f_HSM = temp';
        temp = f_HSM - f_opt(:)';
        temp = abs(temp);

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

        temp = readmatrix('Result_R/time_two_tree.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);

    case 2
       

        temp = readmatrix('Result_R/time_two_path_tree.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);

        temp = readmatrix('Result_R/objval_two_path_tree.csv');
        temp(:, 1) = [];
        f_HSM = temp';
        temp = f_HSM - f_opt(:)';
        temp = abs(temp);
        temp = temp .* (f_HSM ~= 0);
        temp(temp == 0) = nan;
        temp = fillmissing(temp, 'previous');

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

    case 3
       

        temp = readmatrix('Result_R/time_binary_tree.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);


        temp = readmatrix('Result_R/objval_binary_tree.csv');
        temp(:, 1) = [];
        f_HSM = temp';
        temp = f_HSM - f_opt(:)';
        temp = abs(temp);

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

    case 4

        temp = readmatrix('Result_R/objval_reverse_binary_127.csv');
        temp(:, 1) = [];
        f_HSM = temp';

        f_HSM(f_HSM == 0) = nan;
        f_HSM = fillmissing(f_HSM, 'previous');


        temp = f_HSM - f_opt(:)';
        temp = abs(temp);

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

        temp = readmatrix('Result_R/time_reverse_binary_127.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);

    case 5
        temp = readmatrix('Result_R/objval_long_wide.csv');
        temp(:, 1) = [];
        f_HSM = temp';
        temp = f_HSM - f_opt(:)';
        temp = abs(temp);

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

        temp = readmatrix('Result_R/time_long_wide.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);

    case 6
        temp = readmatrix('Result_R/objval_DAG_100.csv');
        temp(:, 1) = [];
        f_HSM = temp';

        f_HSM(f_HSM == 0) = nan;
        f_HSM = fillmissing(f_HSM, 'previous');


        temp = f_HSM - f_opt(:)';
        temp = abs(temp);

        for i = 1:10
            temp(:, i) = temp(:, i) / norm(f_opt(:, :, i));
        end
        f_res_HSM_avg = mean(temp, 2);

        temp = readmatrix('Result_R/time_DAG_100.csv');
        t_HSM_avg = mean(temp(:, 2:end));
        time_HSM_avg = cumsum(t_HSM_avg);

end

figure(1)
hold on
%semilogy(beta_res_HSM_avg, '-x', 'MarkerIndices', 1:100:length(beta_res_HSM_avg), 'MarkerSize', 10, 'LineWidth', 2, 'color', [0.30, 0.75, 0.93]);
semilogy(f_res_HSM_avg, '-x', 'MarkerIndices', 1:500:length(f_res_HSM_avg), 'MarkerSize', 10, 'LineWidth', 2, 'color', [0.30, 0.75, 0.93]);

hold off
legend('ADMM', 'ACC-PGM', 'PGM', 'HSM');

figure(2)
hold on
semilogy(time_HSM_avg, f_res_HSM_avg, '-x', 'MarkerIndices', 1:500:length(f_res_HSM_avg), 'LineWidth', 2, 'MarkerSize', 10);
hold off
legend('ADMM', 'ACC-PGM', 'PGM', 'C-BCD', 'R-BCD', 'HSM');
