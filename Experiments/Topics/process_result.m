
%% Load results

load('Results/FISTA_ADMM_rho5_201014.mat')

f_ADMM = f;
time_real_ADMM = time_real;
time_CPU_ADMM = time_CPU;
grad_prox_ADMM = grad_prox;
grad_D_ADMM = grad_D;
diff_ADMM = diff_A_D;

load('Results/FISTA_RBCD_201012.mat')
f_RBCD = f(1:100);
time_real_RBCD = time_real(1:100);
time_CPU_RBCD = time_CPU(1:100, :);
grad_prox_RBCD = grad_prox(1:100);
grad_D_RBCD = grad_D(1:100);
diff_RBCD = diff_A_D(1:100);


load('Results/FISTA_CBCD_201012.mat')
f_CBCD = f;
time_real_CBCD = time_real;
time_CPU_CBCD = time_CPU;
grad_prox_CBCD = grad_prox;
grad_D_CBCD = grad_D;
diff_CBCD = diff_A_D;

clear f time_real time_CPU grad_prox grad_D diff_A_D A A_save D D_save topic grad_prox_stepsize

%% Process time

t_ADMM_real = cumsum(time_real_ADMM);
t_RBCD_real = cumsum(time_real_RBCD);
t_CBCD_real = cumsum(time_real_CBCD);


t_ADMM_CPU = sum(time_CPU_ADMM, 2);
t_ADMM_CPU = cumsum(t_ADMM_CPU);

t_RBCD_CPU = sum(time_CPU_RBCD, 2);
t_RBCD_CPU = cumsum(t_RBCD_CPU);

t_CBCD_CPU = sum(time_CPU_CBCD, 2);
t_CBCD_CPU = cumsum(t_CBCD_CPU);

t_ADMM_prl = sum(time_CPU_ADMM, 2) - sum(t_ADMM, 2) + sum(t_ADMM, 2) / 13;
t_ADMM_prl = cumsum(t_ADMM_prl);

%% Plot 
figure(1)
clf

% for line style legend
semilogy(NaN, NaN, 'color', 'black','Marker','o','MarkerSize',10)
hold on
semilogy(NaN,NaN,'color','black')
semilogy(NaN, NaN, 'linestyle', '--', 'color', 'black')
semilogy(NaN, NaN, 'linestyle', ':', 'color', 'black')

% for line color legend
patch(NaN, NaN, [0.00, 0.45, 0.74])
patch(NaN, NaN, [0.85, 0.33, 0.10])

% plot lines
semilogy(t_ADMM_CPU, grad_prox_ADMM, 'color', [0.00, 0.45, 0.74],...
    'Marker','o','MarkerSize',10, 'MarkerIndices',1:10:100)
semilogy(t_ADMM_prl, grad_prox_ADMM,'color', [0.00, 0.45, 0.74])

semilogy(t_RBCD_CPU, grad_prox_RBCD, '--', 'color', [0.00, 0.45, 0.74])
semilogy(t_CBCD_CPU, grad_prox_CBCD, ':', 'color', [0.00, 0.45, 0.74])

semilogy(t_ADMM_CPU, diff_ADMM, 'color', [0.85, 0.33, 0.10],...
    'Marker','o','MarkerSize',10, 'MarkerIndices',1:10:100)
semilogy(t_ADMM_prl, diff_ADMM, 'color',[0.85, 0.33, 0.10])

semilogy(t_RBCD_CPU, diff_RBCD, '--', 'color', [0.85, 0.33, 0.10])
semilogy(t_CBCD_CPU, diff_CBCD, ':', 'color', [0.85, 0.33, 0.10])

legend('ADMM','ADMM(parallelized)', 'R-BCD','C-BCD', '$\|\tilde{\nabla}f(A^k,D^k)\|$', '$\frac{1}{n}\|A^k-A^{k-1}\|_F+\frac{1}{m}\|D^k-D^{k-1}\|_F$', 'Interpreter', 'latex')
xlabel('time (seconds)')


set(gca, ...
    'Units', 'normalized', ...
    'FontUnits', 'points', ...
    'FontWeight', 'normal', ...
    'FontSize', 11, ...
    'FontName', 'Arial')
set(gcf, ...
    'Units', 'inches', ...
    'Position', [0, 0, 8, 6])

box off

%expfig('topic_CPU.pdf')




