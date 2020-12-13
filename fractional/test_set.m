function [u, alpha, x, y, Lu] = test_set(Num_u, Num_x, Num_y, Num_alpha, operator_type, space_type, space_paras)
degree = space_paras(1);
size_cube = space_paras(2);
if strcmp(operator_type, '1D_Caputo')
    x = linspace(-1,1, Num_x)';
    
    if strcmp(space_type, 'Orthogonal')
            Phi = Orthogonal_polynomials(x, degree, 'unit_interval');
            q = qrandstream('sobol',degree+1,'Leap',17,'Skip',3000);   
            temp=-size_cube+2*size_cube*qrand(q, Num_u+1);
            coeff_u = temp(2:end,:);
            u = Phi * coeff_u';
            y = linspace(-0.99, 0.99, Num_y)';
            alpha = linspace(0.01, 0.99,Num_alpha)'; 
            Lu = zeros(Num_y, Num_u*Num_alpha);
            for i = 1:Num_alpha
                L_Phi = operator_Orthogonal_polynomials(y, operator_type,[alpha(i)],degree);
                Lu(:, (1+(i-1)*Num_u):(i*Num_u)) = L_Phi * coeff_u';
            end
    end
     save('test_Lu.txt', 'Lu','-ascii','-double');
     save('test_u.txt', 'u','-ascii','-double');
     save('test_y.txt', 'y','-ascii', '-double');
     save('test_alpha.txt', 'alpha', '-ascii', '-double');  
     
     x0 = linspace(-1,1,degree+1)';
     Phi0 = Orthogonal_polynomials(x0, degree, 'unit_interval');
     coeff_u0 = Phi0\(x0+1).^1.58;
     u0 = Phi* coeff_u0;
     Num_y0 = Num_y;
     y0 = linspace(-0.99, 0.99, Num_y0)';
     Num_alpha0 = Num_alpha;
     alpha0 = linspace(0.01, 0.99,Num_alpha0)'; 

     Num_u0 = 1;
     Lu0 = zeros(Num_y0, Num_u0*Num_alpha0);
     Lu_exact = Lu0;
    for i = 1:Num_alpha
        L_Phi0 = operator_Orthogonal_polynomials(y0, operator_type,[alpha0(i)], degree);
        Lu0(:, (1+(i-1)*Num_u0):(i*Num_u0)) = L_Phi0 * coeff_u0;
        Lu_exact(:, (1+(i-1)*Num_u0):(i*Num_u0)) = gamma(2.58)/gamma(2.58-alpha0(i))*(y0+1).^(1.58-alpha0(i));
    end
    
    error = norm(Lu0 - Lu_exact)/norm(Lu_exact)
     save('test_Lu0.txt', 'Lu0','-ascii','-double');
     save('test_u0.txt', 'u0','-ascii','-double');
     save('test_y0.txt', 'y0','-ascii', '-double');
     save('test_alpha0.txt', 'alpha0', '-ascii', '-double');  
    
elseif strcmp(operator_type, '2D_fLap_disk')
    degree = space_paras(1);
    size_cube = space_paras(2);    
   if strcmp(space_type, 'Orthogonal')
            [x_r, x_t] = meshgrid(linspace(0,0.95,Num_x), linspace(0,2*pi,Num_x));
            x = [x_r(:), x_t(:)];
            Phi = Orthogonal_polynomials(x, degree, 'unit_disk');
            q = qrandstream('sobol',degree+1,'Leap',24,'Skip',3);   
            temp=-size_cube+2*size_cube*qrand(q, Num_u+1);
            coeff_u = temp(2:end,:);
            u = Phi * coeff_u';
            
            [y_r, y_t] = meshgrid(linspace(0,0.95,Num_y), linspace(0,2*pi,Num_y)); 
            y = [y_r(:).*cos(y_t(:)), y_r(:).*sin(y_t(:))];
            alpha = linspace(0.01, 0.99,Num_alpha)'; 
            Lu = zeros(Num_y^2, Num_u*Num_alpha);
            for i = 1:Num_alpha
                L_Phi = operator_Orthogonal_polynomials(y, operator_type,[alpha(i)], degree);
                Lu(:, (1+(i-1)*Num_u):(i*Num_u)) = L_Phi * coeff_u';
            end
  
         save('test_Lu.txt', 'Lu','-ascii');
         save('test_u.txt', 'u','-ascii');
         save('test_y.txt', 'y','-ascii');
         save('test_alpha.txt', 'alpha', '-ascii');          

            Num_u0 = 1;
            Num_x0 = Num_x;
            [x_r0, x_t0] = meshgrid(linspace(0,1,60), linspace(0,2*pi,60));
            x0 = [x_r0(:), x_t0(:)];
            Phi0 = Orthogonal_polynomials(x0, degree, 'unit_disk');
           coeff_u0 = Phi0 \ (1-x_r0(:).^2).^(1+1.5/2);
            u0 = Phi* coeff_u0;
            err = norm(Phi0*coeff_u0 - (1-x_r0(:).^2).^(1+1.5/2))/norm((1-x_r0(:).^2).^(1+1.5/2))
             Num_y0 = Num_y;           
            [y_r0, y_t0] = meshgrid(linspace(0,0.95,Num_y0), linspace(0,2*pi,Num_y0)); 
            y0 = [y_r0(:).*cos(y_t0(:)), y_r0(:).*sin(y_t0(:))];
            alpha0 = [1.5]; 
            Num_alpha0 = 1;
            Lu0 = zeros(Num_y0^2, Num_u0*Num_alpha0);
            Lu_exact = Lu0;
            for i = 1:Num_alpha0
                L_Phi0 = operator_Orthogonal_polynomials(y0, operator_type,[alpha0(i)], degree);
                Lu0(:, (1+(i-1)*Num_u0):(i*Num_u0)) = L_Phi0 * coeff_u0;
                Lu_exact(:, (1+(i-1)*Num_u0):(i*Num_u0))  = 2^alpha0(1)*gamma(alpha0(1)/2+2)*gamma((2+alpha0(1))/2)...
                      *(1-(1+alpha0(1)/2)*y_r0(:).^2);
            end
%             coeff00 = L_Phi0 \ Lu_exact;
%     norm(L_Phi0 * coeff00-Lu_exact)/norm(Lu_exact)
            err = norm(Lu0-Lu_exact)/norm(Lu_exact)
%             aaa=3;
   end
     save('test_Lu0.txt', 'Lu0','-ascii');
     save('test_u0.txt', 'u0','-ascii');
     save('test_y0.txt', 'y0','-ascii');
     save('test_alpha0.txt', 'alpha0', '-ascii');          
end
 
end
