function [root, poly,Tri_mat]=rootsofjacobi(N,varargin) 
% ROOTSOFJACOBI  Zeros of Jacobi polynomials, polynomial coefficients 
%                and the tridiagonal matrix corresponding to 
%                the three-term recursive relation of Jacobi polynomials.
%    ROOT=ROOTSOFJACOBI(N,'Legendre') returns the zeros of Legendre 
%    polynomial of degree N>=2.
%
%    ROOT=ROOTSOFJACOBI(N,'Chebyshev1') returns the zeros of Chebyshev
%    polynomial of the first kind.
%
%    ROOT=ROOTSOFJACOBI(N,'Chebyshev2') returns the zeros of Chebyshev
%    polynomial of the second kind.
%
%    ROOT=ROOTSOFJACOBI(N,miu,lamed) returns the zeros of Jacobi polynomial
%    with double-precision parameters miu and lamed both larger than -1.0.
%
%   [ROOT,POLY]=ROOTSOFJACOBI(...) returns the polynomial coefficients with
%   the leading coefficient in POLY(1).
%
%   [ROOT,POLY,TRI_MAT]=ROOTSOFJACOBI(...) returns the tridiagonal matrix
%   that corresponds to the three-term recursive relation of Jacobi
%   polynomials.

%   See also GJ_GENERATE, GJL_GENERATE.
%   Ref: G.H. Golub and J.H. Welsch, " Calculation of Gass quadrature",
%   1967.
%   G.F. Pang, 27/08/2012


if N<1 || abs(round(N)-N)>eps 
    error('************N must be a positive integer************')
end


if length(varargin)==2 && strcmp(class(varargin{1}),'double') && strcmp(class(varargin{2}),'double')
    miu=varargin{1};
    lamed=varargin{2};
    C=ones(1,N+1);
    if miu<=-1.0 || lamed<=-1.0
        error ('***********Miu and lamed must be in (-1,inf)***********')
    end
elseif length(varargin)==1 && ischar(class(varargin))
    switch varargin{:}
        case 'Legendre'
        miu=0.;
        lamed=miu;
        % C is a constant factor relating Jacobi polynomials to some special
        % orthogonal polynomials such as Legendre and Chebyshev
        % polynomials.
        C=ones(1,N+1);
        case 'Chebyshev1'
        miu=-0.5;
        lamed=miu;
        C=[1 2.^(2:2:2*N).*beta(2:(N+1),2:(N+1)).*(2*(1:N)+1)];
        case 'Chebyshev2'
        miu=0.5;
        lamed=0.5;
        C=[1 2.^(2:2:2*N).*beta(3:(N+2),2:(N+1)).*(2*(1:N)+2)];
        otherwise
            error(' Polynomial type must be in {Legendre, Chebyshev1, Chebyshev2}')  
    end
else
    error('***********imporper arguments!************') 
end
    
 
format long

coeff_matrix=zeros(N+1,N+1);
coeff_matrix(1,N+1)=1;
coeff_matrix(2,(end-1):end)=conv([an(1,miu,lamed),bn(1,miu,lamed)],coeff_matrix(1,N+1));


% Generate the coefficients of Jacobi polynomials from degree 2 to N using
% three-term recursive relation.
for i=3:N+1
    coeff_matrix(i,(end-i+1):end)= ...
    conv([an(i-1,miu,lamed),bn(i-1,miu,lamed)],coeff_matrix(i-1,(end-i+2):end)) ... 
    -[0 0 cn(i-1,miu,lamed)*coeff_matrix(i-2,(end-i+3):end)];
end


% Transform Jacobi polynomials to Legendre/Chebyshev polynomials.
for j=1:N+1
    coeff_matrix(j,:)=coeff_matrix(j,:)*C(j);
end
poly=coeff_matrix(end,:);


% Generate the tridiagonal matrix, zeros and poly.
A=zeros(1,N);
B=zeros(1,N-1);
   for n=1:N
      A(n)=-bn(n,miu,lamed)/an(n,miu,lamed);
   end
   
   for n=1:N-1
     B(n)=(cn(n+1,miu,lamed)/an(n,miu,lamed)/an(n+1,miu,lamed))^0.5;
   end 
Tri_mat=diag(A)+diag(B,1)+diag(B,-1);

% Obtain zeros by computing the eigenvalues of matrix.
root=sort(eig(Tri_mat));



%---Coefficients in three-term recursive relation-------------------------

function y=an(n,miu,lamed)
    if abs(miu+lamed+1)<eps
        if n>=2
        y=(2*n-1)/n;
        else
            y=0.5;
        end
    else
        y=(2*n+miu+lamed-1)*(2*n+miu+lamed)/(2*n*(n+miu+lamed));
    end
    
%--------------------------------------------------------------------------

function y=bn(n,miu,lamed)  
     if abs(miu+lamed+1)<eps
        if n>=2
        y=(miu^2-lamed^2)/n/(2*n-3);
        else
            y=(miu^2-lamed^2)*(-0.5);
        end
     elseif abs(abs(miu)-abs(lamed))<eps
         y=0;
     else
        y=(2*n+miu+lamed-1)*(miu^2-lamed^2)/(2*n*(n+miu+lamed)*(2*n+miu+lamed-2));
    end
    
%--------------------------------------------------------------------------
    
function y=cn(n,miu,lamed)
    y=(n+miu-1)*(n+lamed-1)*(2*n+miu+lamed)/(n*(n+miu+lamed)*(2*n+miu+lamed-2));
          
%--------------------------------------------------------------------------