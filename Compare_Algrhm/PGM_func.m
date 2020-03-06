function [beta_PGM, t_PGM, f_PGM] = PGM_func(indx, y, iteration, lambda, w, alpha0)

n_para = length(indx);
n_iteration = iteration;

x_PGM = zeros(n_para, n_para);

x_PGM_new = zeros(n_para, n_para);
t_PGM = zeros(n_iteration, 1);
f_PGM = zeros(n_iteration, 1);
beta_PGM = zeros(n_para, n_iteration);
alpha = alpha0; % step-size

for k = 1:n_iteration
    tic
    beta_pre = sum(x_PGM, 2);
    grad = beta_pre - y;
    while (1)

        for i = 1:n_para
            temp = x_PGM(indx{i}, i) - alpha * grad(indx{i});
            x_PGM_new(indx{i}, i) = temp * max(0, 1-alpha*lambda*w(i)/norm(temp));
        end
        beta = sum(x_PGM_new, 2);
        F = 0.5 * norm(y-beta)^2;
        %  Q_pre = 0.5 * norm(y-beta_pre)^2 + sum(grad'*(x_PGM-x_PGM)) + 1 / (2 * alpha) * norm(beta_pre-beta_pre,'fro')^2;

        Q = 0.5 * norm(y-beta_pre)^2 + sum(grad'*(x_PGM_new - x_PGM)) + 1 / (2 * alpha) * norm(x_PGM_new-x_PGM, 'fro')^2;
        if F <= Q
            break
        else
            alpha = 0.8 * alpha;
        end
    end
    t_PGM(k) = toc;

    beta_PGM(:, k) = beta;
    x_PGM = x_PGM_new;
    f_PGM(k) = 0.5 * norm(y-beta_PGM(:, k))^2 + lambda * sqrt(sum(x_PGM(:, :).^2, 1)) * w;


end
