function [u, alpha, x, y, Lu] = training_set(Num_u, Num_x, Num_y, Num_alpha, operator_type, space_type, space_paras)
if strcmp(operator_type, '1D_Caputo')
    degree = space_paras(1);
    size_cube = space_paras(2);
    
    if strcmp(space_type, 'Orthogonal')
            x = linspace(-1,1, Num_x)';
            Phi = Orthogonal_polynomials(x, degree, 'unit_interval');
            q = qrandstream('sobol',degree+1,'Leap',24,'Skip',3);   
            temp=-size_cube+2*size_cube*qrand(q, Num_u+1);
            coeff_u = temp(2:end,:);
            u = Phi * coeff_u';
            y = linspace(-0.99, 0.99, Num_y)';
            alpha = linspace(0.01, 0.99,Num_alpha)'; 
            Lu = zeros(Num_y, Num_u*Num_alpha);
            for i = 1:Num_alpha
                L_Phi = operator_Orthogonal_polynomials(y, operator_type,[alpha(i)], degree);
                Lu(:, (1+(i-1)*Num_u):(i*Num_u)) = L_Phi * coeff_u';
            end
    end
     save('training_Lu.txt', 'Lu','-ascii','-double');
     save('training_u.txt', 'u','-ascii','-double');
     save('training_y.txt', 'y','-ascii', '-double');
     save('training_alpha.txt', 'alpha', '-ascii', '-double');   

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
   end
%      save('training_Lu.txt', 'Lu','-ascii','-double');
%      save('training_u.txt', 'u','-ascii','-double');
%      save('training_y.txt', 'y','-ascii', '-double');
%      save('training_alpha.txt', 'alpha', '-ascii', '-double');      
     save('training_Lu.txt', 'Lu','-ascii');
     save('training_u.txt', 'u','-ascii');
     save('training_y.txt', 'y','-ascii');
     save('training_alpha.txt', 'alpha', '-ascii');       
    
end
