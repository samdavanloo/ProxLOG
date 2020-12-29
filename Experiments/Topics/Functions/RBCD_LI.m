function [beta_RBCD, P_LOG] = RBCD_LI(indx, y, lambda, w, crt, max_iteration)
%[beta_RBCD, P_LOG] = RBCD_func(indx, y, lambda, w, crt, max_iteration)


n_para = length(indx);


v_BCD = zeros(n_para, n_para);
%t_RBCD = zeros(n_iteration, 1);
%f_RBCD = zeros(n_iteration, 1);
beta_RBCD = zeros(n_para, 1);
P_LOG = 0;
temp_beta = zeros(n_para, 1);

for k = 1:max_iteration
    beta_pre = beta_RBCD;

    for j = 1:n_para
        i = randi([1, n_para], 1);

        idx = indx{i};
        temp_beta(idx) = temp_beta(idx) - v_BCD(idx, i);


        T = y(idx) - temp_beta(idx);
        v_BCD(idx, i) = T - ProjectOntoL1Ball(T, lambda*w(i));
        temp_beta(idx) = temp_beta(idx) + v_BCD(idx, i);


    end


    beta_RBCD = temp_beta;

    if k > 1 && norm(beta_RBCD-beta_pre) < crt
        P_LOG = max(abs(v_BCD)) * w;
        break
    elseif k == max_iteration
        P_LOG = max(abs(v_BCD)) * w;
    end

end


%