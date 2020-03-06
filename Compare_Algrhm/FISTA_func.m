function [beta_FISTA, time_FISTA, f_FISTA] = FISTA_func(indx, y, iteration, lambda, w, alpha)

n_para = length(indx);
n_iteration = iteration;

t = 1;
x_FISTA = zeros(n_para, n_para);
time_FISTA = zeros(n_iteration, 1);
f_FISTA = zeros(n_iteration, 1);
beta_FISTA = zeros(n_para, n_iteration);
y_FISTA = zeros(n_para, n_para);

for k = 1:n_iteration
    tic
    grad = sum(y_FISTA, 2) - y;
    x_FISTA_prev = x_FISTA;

    while (1)

        for i = 1:n_para
            temp = y_FISTA(indx{i}, i) - alpha * grad(indx{i});
            x_FISTA(indx{i}, i) = temp * max(0, 1-alpha*lambda*w(i)/norm(temp));
        end


        F = 0.5 * norm(y-sum(x_FISTA, 2))^2;
        Q_pre = 0.5 * norm(y-y_FISTA)^2 ;

        Q = 0.5 * norm(y-sum(y_FISTA, 2))^2 + sum(grad'*(x_FISTA - y_FISTA)) + 1 / (2 * alpha) * norm(x_FISTA-y_FISTA, 'fro')^2;
        if F <= Q
            break
        else
            alpha = 0.8 * alpha;
        end


    end
    t_prev = t;
    t = (1 + sqrt(1+4*t_prev^2)) / 2;
    y_FISTA = x_FISTA + (t_prev - 1) / t * (x_FISTA - x_FISTA_prev);

    time_FISTA(k) = toc;

    beta_FISTA(:, k) = sum(x_FISTA, 2);

    f_FISTA(k) = 0.5 * norm(y-beta_FISTA(:, k))^2 + lambda * sqrt(sum(x_FISTA(:, :).^2, 1)) * w;

    
    
end
