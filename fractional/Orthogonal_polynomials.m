function Phi = Orthogonal_polynomials( Loc_u, degree, domain_type)

     if strcmp(domain_type, 'unit_interval')
         syms x
         expr = legendreP(0:degree, x); % Legendre polynomial
%      expr =(1 + x).^0.5.* jacobiP(0:degree, -0.5, 0.5, x);  %       poly-fractonomial
         
         x = Loc_u;
         Phi = double(subs(expr));
         
     elseif strcmp(domain_type, 'unit_disk')
         Phi = Zp(Loc_u(:,1), Loc_u(:,2), degree);  % Loc_u 's in polar coordinate system
     end
   