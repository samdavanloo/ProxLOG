function [beta_ADMM,t1_ADMM, t2_ADMM, f_ADMM, dual_ADMM, x_minus_z_ADMM] = ADMM_func(indx,y,iteration,rho,lambda,w)
% [beta_ADMM,t1_ADMM, t2_ADMM, f_ADMM, dual_ADMM, x_minus_z_ADMM] = 
%        ADMM_func(indx,y,iteration,rho,lambda,w)
% t1_ADMM -- update x; t2_ADMM -- update z and u

n_para = length(indx);
iter_max = iteration;

x_ADMM = zeros(n_para,n_para);
z_ADMM = zeros(n_para,1);   %z_bar
u_ADMM = zeros(n_para,1);

beta_ADMM = zeros(n_para, iter_max); %beta value for each iteration
x_minus_z_ADMM = zeros(iter_max,1); %norm(x-z)
f_ADMM = zeros(iter_max,1);    %f value for each iteration
dual_ADMM = zeros(iter_max,1); %dual value for each iteration
t1_ADMM = zeros(iter_max,1);  %tic toc value for x
t2_ADMM = zeros(iter_max,1);  %tic toc value for z and u

x_bar = zeros(n_para,1);
k = 1;
while k <=iter_max 
    temp_u=u_ADMM;  %used for dual value
    
    %update x
    tic
    for i = 1:n_para
        temp_x = x_ADMM(indx{i},i) - x_bar(indx{i}) + z_ADMM(indx{i}) - u_ADMM(indx{i});
        x_ADMM(indx{i},i) = temp_x * max(0, 1-lambda * w(i)/rho/norm(temp_x));
    end
    t1_ADMM(k) = toc;
    
    %update z and u
    tic
    x_bar = sum(x_ADMM,2)/n_para;
    z_ADMM = 1/(n_para+rho)*(y+ rho*(u_ADMM+x_bar));
    u_ADMM = u_ADMM + x_bar - z_ADMM;
    t2_ADMM(k) = toc;
    
    
    dual_ADMM(k) = sqrt(sum(x_ADMM.^2,1))*lambda* w + 0.5*norm(z_ADMM*n_para - y)^2 +rho*n_para*temp_u'*(sum(x_ADMM,2)/n_para - z_ADMM)+0.5*rho*n_para*norm(sum(x_ADMM,2)/n_para - z_ADMM)^2;
    
    beta_ADMM(:,k) = sum(x_ADMM,2);
    x_minus_z_ADMM(k) = sqrt(n_para*norm(sum(x_ADMM,2)/n_para - z_ADMM)^2);
    f_ADMM(k) = sqrt(sum(x_ADMM.^2,1))* lambda* w + 0.5*norm(z_ADMM*n_para - y).^2;
    
  
    k = k+1;
end



