%% Load result
load('Results/cancer_FISTA_ADMM.mat')
f_ADMM = f;
time_CPU_ADMM = time_CPU;
time_prl_ADMM = time_parallel;

load('Results/cancer_FISTA_CBCD.mat')
f_CBCD = f;
time_CPU_CBCD = time_CPU;

load('Results/cancer_FISTA_RBCD.mat')
f_RBCD = f;
time_CPU_RBCD = time_CPU;


%% Process time

t_ADMM_CPU = cumsum(time_CPU_ADMM);
t_ADMM_prl_1 = time_CPU_ADMM-time_prl_ADMM + time_prl_ADMM/28;
t_ADMM_prl_1 = cumsum(t_ADMM_prl_1);
t_RBCD_CPU = cumsum(time_CPU_RBCD);

t_CBCD_CPU = cumsum(time_CPU_CBCD);

%% Plot

figure(2)
clf
semilogy(t_ADMM_CPU,abs(f_ADMM-f_ADMM(end))/f_ADMM(end))
hold on
semilogy(t_ADMM_prl_1,abs(f_ADMM-f_ADMM(end))/f_ADMM(end))

semilogy(t_RBCD_CPU,abs(f_RBCD-f_ADMM(end))/f_ADMM(end))

semilogy(t_CBCD_CPU,abs(f_CBCD-f_ADMM(end))/f_ADMM(end))

xlabel('time (seconds)')
ylabel('$f(\theta^k)-f^*/|f^*|$','Interpreter','latex')
legend('ADMM','ADMM (parallelized)','R-BCD','C-BCD')
