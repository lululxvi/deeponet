function [pts_cell0, coeff_mat]=matrix_for_frac_Lap(y,alpha)

M = size(y,1);
M_theta = 16;
[KESI,W]=GJ_generate(M_theta,0,0);

lam_N = 8000;
index = (0:lam_N)';
gk=wk(alpha,lam_N+10);
h = 1.0e-3;
pts_cell = cell(1,M);
coeff_cell = cell(1,M);

parfor i=1:M
    x0=y(i,1);
    y0=y(i,2);
    theta1=pi+pi*KESI;
     W1=pi*W; 
    
     vec = [x0-index*h*cos(theta1(1)) y0-index*h*sin(theta1(1))];
     id = find(sum(vec.^2,2)>1);
     id0 = id(1)-1; 
     vec0 = vec(1:id0,:);
     coeff0 = W1(1)*h^(-alpha)* gk(1:id0);
     
     
     for k2=2:M_theta
          vec = [x0-index*h*cos(theta1(k2)) y0-index*h*sin(theta1(k2))];
          id = find(sum(vec.^2,2)>1);
          id0 = id(1)-1; 
          vec0 = [vec0; vec(1:id0,:)];
          coeff = W1(k2)*h^(-alpha)*gk(1:id0);
          coeff0 = [coeff0; coeff];
          
%           [t_vec, r_vec] = cart2pol(x_vec, y_vec);
%           v = fun(r_vec,t_vec,coeff);     
%           s1(k2)=h^(-alpha)*(gk(1:id0)'*v);
     end
       
     pts_cell{i} = vec0;
     coeff_cell{i} = coeff0;
end

max_size = 0;
for i=1:M
    temp = size(pts_cell{i},1);
    if temp>max_size
        max_size = temp;
    end
end

coeff_mat = zeros(M, max_size);
pts_cell0 = cell(1,M);
for i =1:M
    len = size(coeff_cell{i},1);
    coeff_mat(i,1:len)=coeff_cell{i};
    pts_cell0{i} = [pts_cell{i}; zeros(max_size-len,2)];
end

% u_aux = zeros(max_size, M);
% for i=1:M
%     len = size(pts_cell{i},1);
%     u_aux(1:len,i) = u_exact(pts_cell{i},alpha);
% end
ratio=gamma((1-alpha)/2)*gamma((2+alpha)/2)/sqrt(pi)/2/pi;
coeff_mat = coeff_mat * ratio;
% f_appr = ratio*diag(coeff_mat*u_aux);
% 
% f_exa = f_sour(y, alpha);
% 
% err = norm(f_appr-f_exa)/norm(f_exa)
    

function y=wk(q,K)
    y=zeros(K+1,1);
    y(1)=1;

   parfor kk=1:K  
       b=1:kk;
       a=-q+b-1;
       y(kk+1)=prod(a./b);
   end
   
   
        function y=f_sour(x,alpha)
            d=2;
        y=2^alpha*gamma(alpha/2+2)*gamma((d+alpha)/2)*gamma(d/2)^(-1)*(1-(1+alpha/d)*(x(:,1).^2+x(:,2).^2));
    
    
    function y=u_exact(x,alpha)
        y=(1-x(:,1).^2-x(:,2).^2+1.0e-15).^(1+alpha/2);