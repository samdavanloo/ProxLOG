function [beta_BCD, t_BCD, f_BCD, P_LOG] = BCD_func(indx, y, lambda, w, max_iteration)
%[beta_BCD,t_BCD, f_BCD] = BCD_func(indx,y,lambda,w,max_iteration)


n_para = length(indx);
n_iteration = max_iteration;

v_BCD = zeros(n_para, n_para);
t_BCD = zeros(n_iteration, 1);
f_BCD = zeros(n_iteration, 1);
beta_BCD = zeros(n_para, n_iteration);
temp_beta = zeros(n_para, 1);

for k = 1:max_iteration
    tic
    for i = 1:n_para


        temp_beta(indx{i}) = temp_beta(indx{i}) - v_BCD(indx{i}, i);


        temp = y(indx{i}) - temp_beta(indx{i});
        v_BCD(indx{i}, i) = temp * max(0, 1-lambda*w(i)/norm(temp));
        temp_beta(indx{i}) = temp_beta(indx{i}) + v_BCD(indx{i}, i);


        %          temp_beta = temp_beta- v_BCD(:,i);
        %          temp = (y-temp_beta) .* indx(:,i);
        %          v_BCD(:,i) = temp * max(0, 1-lambda * w(i)/norm(temp));
        %          temp_beta = temp_beta + v_BCD(:,i);
    end

    t_BCD(k) = toc;
    beta_BCD(:, k) = temp_beta;
    f_BCD(k) = 0.5 * norm(y-temp_beta)^2 + lambda * sqrt(sum(v_BCD.^2, 1)) * w;

  

end
P_LOG = sqrt(sum(v_BCD.^2, 1)) * w;


%