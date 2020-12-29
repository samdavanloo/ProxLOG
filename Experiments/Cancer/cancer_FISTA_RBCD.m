% Cancer meta classification problem
% use weight to balance samples
% FISTA method, admm as proximal operator solver
% Vant veer breast cancer dataset (295 samples,78 meta)

%% Setting
currentFile = sprintf('./Results/cancer_FISTA_RBCD.mat');
lambda = 0.05;
rho = 100;
step_size = 0.001; %step size of PGM
crt = 1e-8;
ite_max = 1000; % iteration for PGM
ite_BCD = 1500; % iteration for inner ADMM

load('data_cancer.mat')
group_newindex = zeros(size(group));

for i = 1:500
    group_newindex(group == gene(i)) = i;
end

%% Visualize connection of 500 genes
group_newindex = sort(group_newindex, 2);
%  M = zeros(500);
% idx = sub2ind(size(M), group_newindex(:, 1), group_newindex(:, 2));
% M(idx) = 1;
% M_viso = M + M';
% spy(M_viso);
% title('correlations of genes')

%% Find not connected genes
range_gene = [1:500];
idx = ismember(range_gene, unique(sort(group_newindex(:))));
node_nconct = range_gene(~idx);

%% Generate group cells

% sigle nodes
G1 = mat2cell(range_gene(~idx)', ones(sum(~idx), 1));
% connected nodes
G2 = mat2cell(group_newindex, ones(size(group_newindex, 1), 1));
% total group
G_idx = [G1; G2];

%% Seperate training and testing set
rng(6)
n = size(expression, 2);
temp = randperm(n);
idx_train = temp(1:n*0.8);
idx_test = temp(n*0.8+1:end);

data_train = expression(:, idx_train);
data_test = expression(:, idx_test);
label_train = labels(idx_train);
label_test = labels(idx_test);

clear('G1', 'G2', 'idx', 'temp');

%% Initialize
f = zeros(ite_max, 1);
time_real = zeros(ite_max, 1);
time_CPU = zeros(ite_max, 1);

P_LOG = zeros(ite_max, 1); %LOG pelnaty
time_parallel = zeros(ite_max, 1); %time that can be paralllazed.
num_G = length(G_idx);
w = ones(num_G, 1);

X = [ones(size(data_train, 2), 1)'; data_train];
theta = zeros(size(X, 1), 1);
Y = label_train';
% add balance weight to each sample
b = sum(labels) / length(labels);
a = 1 - b;
w_b = zeros(size(label_train))';
w_b(logical(label_train)) = a;
w_b(~logical(label_train)) = b;

t = 1;
y = theta;

%% FISTA

for ite = 1:ite_max
    tic
    time = cputime;
    theta_pre = theta;

    h = 1 ./ (1 + exp(-theta'*X));
    grad = X * ((h - Y) .* w_b)';

    prox = y - step_size * grad;
    [theta, P_LOG(ite)] = RBCD_L2(G_idx, prox, lambda*step_size, w, crt, ite_BCD);
    t_pre = t;
    t = (1 + sqrt(1+4*t^2)) / 2;
    y = theta + (t_pre - 1) / t * (theta - theta_pre);

    h = 1 ./ (1 + exp(-theta'*X));
    f(ite) = -(Y * a * log(h)' + (1 - Y) * b * log(1-h)') + lambda * P_LOG(ite);

    % Stop criteria
    if ite > 1 && norm(theta_pre-theta) < crt
        f = f(1:ite);
        P_LOG = P_LOG(1:ite);
        break
    end

    time_CPU(ite) = cputime - time;
    time_real(ite) = toc;
    fprintf('finish ite %g, f = %d ,time = %s\n', ite, f(ite), datestr(now, 15))

end

%% Test

X_test = [ones(size(data_test(1, :))); data_test];
h_test = 1 ./ (1 + exp(-theta'*X_test));
result = label_test' == (h_test >= 0.5);
sum(result) / length(result)

save(currentFile, 'f', 'theta', 'time_CPU', 'time_real', 'time_parallel')
