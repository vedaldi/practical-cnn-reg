setup() ;

% Create a random input image
x = randn(10,10,'single') ;

% Define a filter
w = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

% Forward mode: evaluate the convolution
y = vl_nnconv(x, w, []) ;

% Pick a random projection tensor
p = randn(size(y), 'single') ;

% Backward mode: projected derivatives
[dx,dw] = vl_nnconv(x, w, [], p) ;

% Check the derivative numerically
delta = 0.01 ;
dx_numerical = zeros(size(dx), 'single') ;
for i = 1:numel(x)
  xp = x ; 
  xp(i) = xp(i) + delta ;
  yp = vl_nnconv(xp,w,[]) ;
  dx_numerical(i) =  p(:)' * (yp(:) - y(:)) / delta ;
end

figure(1) ; clf('reset') ;
subplot(1,3,1) ; bar3(dx) ; zlim([-20 20]) ;
title('dx') ;
subplot(1,3,2) ; bar3(dx_numerical) ; zlim([-20 20]) ;
title('dx (numerical)') ;
subplot(1,3,3) ; bar3(abs(dx-dx_numerical)) ; zlim([-20 20]) ;
title('absolute difference') ;

% Forward mode: evaluate the conv + ReLU
y = vl_nnconv(x, w, []) ;
z = vl_nnrelu(y) ;

% Pick a random projection tensor
p = randn(size(z), 'single') ;

% Backward mode: projected derivatives
dy = vl_nnrelu(z, p) ;
[dx,dw] = vl_nnconv(x, w, [], dy) ;

% Check the derivative numerically
delta = 0.01 ;
dx_numerical = zeros(size(dx), 'single') ;
for i = 1:numel(x)
  xp = x ; 
  xp(i) = xp(i) + delta ;
  yp = vl_nnconv(xp,w,[]) ;
  zp = vl_nnrelu(yp) ;
  dx_numerical(i) =  p(:)' * (zp(:) - z(:)) / delta ;
end

figure(2) ; clf('reset') ;
subplot(1,3,1) ; bar3(dx) ; zlim([-20 20]) ;
title('dx') ;
subplot(1,3,2) ; bar3(dx_numerical) ; zlim([-20 20]) ;
title('dx (numerical)') ;
subplot(1,3,3) ; bar3(abs(dx-dx_numerical)) ; zlim([-20 20]) ;
title('absolute difference') ;

