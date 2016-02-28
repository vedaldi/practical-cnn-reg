function z = proj(x,p)
%PROJ  Project a tensor onto anotehr
%   Z = PROJ(X,P) computes the projection Z of tensor X onto P.
%
%   Remark: if X and P contain multiple tensor instances
%   (concatenated along the foruth dimension), then the
%   result Z contains a scalar projection for each.

prods = x .* p ;
z = sum(prods(:)) ;
