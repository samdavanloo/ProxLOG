function [beta_RBCD, t_RBCD, f_RBCD] = RBCD_func(indx, y, lambda, w, max_iteration)
%[beta_BCD,t_BCD, f_BCD] = BCD_func(indx,y,lambda,w,max_iteration)


n_para = length(indx);
n_iteration = max_iteration;

v_BCD = zeros(n_para, n_para);
t_RBCD = zeros(n_iteration, 1);
f_RBCD = zeros(n_iteration, 1);
beta_RBCD = zeros(n_para, n_iteration);

temp_beta = zeros(n_para, 1);

for k = 1:max_iteration
    tic
    for j = 1:n_para
        i = randi([1, n_para], 1);


        temp_beta(indx{i}) = temp_beta(indx{i}) - v_BCD(indx{i}, i);


        temp = y(indx{i}) - temp_beta(indx{i});
        v_BCD(indx{i}, i) = temp * max(0, 1-lambda*w(i)/norm(temp));
        temp_beta(indx{i}) = temp_beta(indx{i}) + v_BCD(indx{i}, i);


    end

    t_RBCD(k) = toc;
    beta_RBCD(:, k) = temp_beta;
    f_RBCD(k) = 0.5 * norm(y-temp_beta)^2 + lambda * sqrt(sum(v_BCD.^2, 1)) * w;


end


%