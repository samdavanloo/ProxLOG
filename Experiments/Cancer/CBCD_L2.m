function [beta_BCD, P_LOG] = CBCD_L2(indx, y, lambda, w, crt, max_iteration)
%[beta_BCD,t_BCD, f_BCD] = BCD_func(indx,y,lambda,w,max_iteration)


num_G = length(indx);
num_v = size(y,1);
v_BCD = zeros(num_v,num_G);
%t_BCD = zeros(n_iteration, 1);
%f_BCD = zeros(n_iteration, 1);
beta_BCD = zeros(num_v, 1);
temp_beta = zeros(num_v, 1);

for k = 1:max_iteration
    beta_pre = beta_BCD;
    for i = 1:num_G

        idx = indx{i};
        temp_beta(idx) = temp_beta(idx) - v_BCD(idx, i);


        temp = y(idx) - temp_beta(idx);
        v_BCD(idx, i) = temp * max(0, 1-lambda*w(i)/norm(temp));
        temp_beta(idx) = temp_beta(idx) + v_BCD(idx, i);


    end

    beta_BCD = temp_beta;
    %f_BCD(k) = 0.5 * norm(y-temp_beta)^2 + lambda * sqrt(sum(v_BCD.^2, 1)) * w;

    if k > 1 && norm(beta_BCD-beta_pre) < crt
        %  f_ADMM = f_ADMM(1:k);
        P_LOG = sqrt(sum(v_BCD.^2, 1)) * w; % LOG penalty
        break
    elseif k == max_iteration
        P_LOG = sqrt(sum(v_BCD.^2, 1)) * w; % LOG penalty
    end

end


%