function [point,weight]=GJ_generate(N,varargin)
%GJ_GENERATE  Gauss-Jacobi quadrature points and weights.
%   [POINT,WEIGHT]=GJ_GENERATE(N) returns N points and N corresponding
%   weights of Gauss-Legendre quadrature rules.

%   [POINT,WEIGHT]=GJ_GENERATE(N,miu,lamed) returns the points and weights
%   of Gauss-Jacobi rules with double-precision parameters miu and lamed.

%   See also ROOTSOFJACOBI, GJL_GENERATE.
%   Ref: G.H. Golub and J.H. Welsch, " Calculation of Gass quadrature",
%   1967
%   G.F. Pang, 27/08/2012

  if abs(round(N)-N)>eps 
       error('**********N must be an integer*************')
   end
   if length(varargin)==2 && strcmp(class(varargin{1}),'double') && strcmp(class(varargin{2}),'double')
        miu=varargin{1};
        lamed=varargin{2};
        if miu<=-1.0 || lamed<=-1.0
        error ('***********Miu and lamed must be in (-1,inf)***********')
        end     
   elseif isempty(varargin)
        % Default case: Gauss-Legendre rule.
        miu=0.;
        lamed=miu;
   else
        error('***********Imporper arguments!************') 
   end
   
   % Quadrature points.
   [x_nodes,pp,T]=rootsofjacobi(N,miu,lamed);
   point=x_nodes;    
   
   % Weights.
   [V,pp]=eig(T);
   miu0=2^(miu+lamed+1)*gamma(miu+1)*gamma(lamed+1)/gamma(miu+lamed+2);
   w=zeros(N,1);
   for i=1:N
       w(i)=miu0*V(1,i)^2;
   end
   weight=w;