function [beta_CBCD, P_LOG] = CBCD_LI(indx, y, lambda, w, crt, max_iteration)
%[beta_CBCD, P_LOG] = CBCD_func(indx, y, lambda, w, crt, max_iteration)


n_para = length(indx);


v_BCD = zeros(n_para, n_para);
%t_CBCD = zeros(n_iteration, 1);
%f_CBCD = zeros(n_iteration, 1);
beta_CBCD = zeros(n_para, 1);
P_LOG = 0;
temp_beta = zeros(n_para, 1);

for k = 1:max_iteration
    beta_pre = beta_CBCD;

    for i = 1:n_para

        idx = indx{i};
        temp_beta(idx) = temp_beta(idx) - v_BCD(idx, i);


        T = y(idx) - temp_beta(idx);
        v_BCD(idx, i) = T - ProjectOntoL1Ball(T, lambda*w(i));
        temp_beta(idx) = temp_beta(idx) + v_BCD(idx, i);


    end


    beta_CBCD = temp_beta;

    if k > 1 && norm(beta_CBCD-beta_pre) < crt
        P_LOG = max(abs(v_BCD)) * w;
        break
    elseif k == max_iteration
        P_LOG = max(abs(v_BCD)) * w;
    end

end


%