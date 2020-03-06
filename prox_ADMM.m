function [beta_out,f_out,penalty_out] = prox_ADMM(group,b,iter_max,rho,lambda,w)
% [beta_ADMM,t1_ADMM, t2_ADMM, f_ADMM, dual_ADMM, x_minus_z_ADMM] = 
%        ADMM_func(indx,y,iteration,rho,lambda,w)
% t1_ADMM -- update x; t2_ADMM -- update z and u
option.Toldual = 1e-6;
n_para = length(group);

x_ADMM = zeros(n_para,n_para);
z_ADMM = zeros(n_para,1);   %z_bar
u_ADMM = zeros(n_para,1);

beta_ADMM = zeros(n_para, iter_max); %beta value for each iteration
x_minus_z_ADMM = zeros(iter_max,1); %norm(x-z)
penalty_ADMM = zeros(iter_max,1);   % penalty value for each iteration
f_ADMM = zeros(iter_max,1);    %f value for each iteration
dual_ADMM = zeros(iter_max,1); %dual value for each iteration
t1_ADMM = zeros(iter_max,1);  %tic toc value for x
t2_ADMM = zeros(iter_max,1);  %tic toc value for z and u

x_bar = zeros(n_para,1);

k = 1;
while k <=iter_max 
    temp_u=u_ADMM;  %used for dual value
    
    %update x
    %tic
    for i = 1:n_para
        temp_x = x_ADMM(group{i},i) - x_bar(group{i}) + z_ADMM(group{i}) - u_ADMM(group{i});
        x_ADMM(group{i},i) = temp_x * max(0, 1-lambda * w(i)/rho/norm(temp_x));
    end
   % t1_ADMM(k) = toc;
    
    %update z and u
   % tic
    x_bar = sum(x_ADMM,2)/n_para;
    z_ADMM = 1/(n_para+rho)*(b+ rho*(u_ADMM+x_bar));
    u_ADMM = u_ADMM + x_bar - z_ADMM;
   % t2_ADMM(k) = toc;
    
    
    dual_ADMM(k) = sqrt(sum(x_ADMM.^2,1))*lambda* w + 0.5*norm(z_ADMM*n_para - b)^2 +rho*n_para*temp_u'*(sum(x_ADMM,2)/n_para - z_ADMM)+0.5*rho*n_para*norm(sum(x_ADMM,2)/n_para - z_ADMM)^2;
    
    beta_ADMM(:,k) = sum(x_ADMM,2);
    x_minus_z_ADMM(k) = norm(sum(x_ADMM,2) - n_para*z_ADMM);
    
    penalty_ADMM(k) = sqrt(sum(x_ADMM.^2,1))* lambda* w;
    f_ADMM(k) = penalty_ADMM(k) + 0.5*norm(z_ADMM*n_para - b).^2;
    if x_minus_z_ADMM(k) <=option.Toldual
        beta_ADMM = beta_ADMM(:,1:k);
        x_minus_z_ADMM = x_minus_z_ADMM(1:k);
        dual_ADMM = dual_ADMM(1:k);
        
        t1_ADMM = t1_ADMM(1:k);
        t2_ADMM = t2_ADMM(1:k);
        penalty_ADMM = penalty_ADMM(1:k);
        f_ADMM = f_ADMM(1:k);
        break
    end
    k = k+1;
end

beta_out = beta_ADMM(:,end);
f_out = f_ADMM(end);
penalty_out = penalty_ADMM(end);

