function L_Phi = operator_Orthogonal_polynomials( Loc_u, operator_type, operator_paras, degree)

     if strcmp(operator_type, '1D_Caputo')
         alpha = operator_paras(1);
         M = size(Loc_u,1);
         N = 1000;
          pt_cell = cell(1,M);
         coeff_mat = zeros(M, N+1);
     
         for i = 1:M
             pt_cell{i}= linspace(-1, Loc_u(i),N+1)';
             coeff = zeros(1,N+1);
             h = (Loc_u(i)+1)/N;
             index = 0:(N-1);
             cc = (N-index).^(1-alpha)-(N-index-1).^(1-alpha);
             coeff (2:(N+1)) = cc;
             coeff(1:N)=coeff(1:N) - cc;
             coeff_mat(i,:) = coeff * h^(-alpha)/gamma(2-alpha);
         end  
         L_Phi = zeros(M, degree+1);
         for i = 1:M
             Phi = Orthogonal_polynomials(pt_cell{i}, degree, 'unit_interval');
             L_Phi(i,:) = coeff_mat(i,:) * Phi;   
         end
   
     elseif strcmp(operator_type, '2D_fLap_disk')
          alpha = operator_paras(1);
          M = size(Loc_u,1);
          [pt_cell, coeff_mat]=matrix_for_frac_Lap(Loc_u, alpha); 
         L_Phi = zeros(M, degree+1);
         for i = 1:M
             [t0,r0]=cart2pol(pt_cell{i}(:,1), pt_cell{i}(:,2));
             Phi = Orthogonal_polynomials([r0,t0], degree, 'unit_disk');
             L_Phi(i,:) = coeff_mat(i,:) * Phi;   
         end
     end
   