setup() ;

%% Part 2.1:  Backward mode verification

% Create a random input image batch
x = randn(10,10,1,2,'single') ;

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
figure(1) ; clf('reset') ;
set(gcf,'name','Part 2.1: single layer backrpop') ;
checkDerivativeNumerically(@(x) proj(p, vl_nnconv(x, w, [])), x, dx) ;

%% Part 2.2: Backpropagation

% Create a random input image batch
x = randn(10,10,1,2,'single') ;

% Forward mode: evaluate the conv follwed by ReLU
y = vl_nnconv(x, w, []) ;
z = vl_nnrelu(y) ;

% Pick a random projection tensor
p = randn(size(z), 'single') ;

% Backward mode: projected derivatives
dy = vl_nnrelu(z, p) ;
[dx,dw] = vl_nnconv(x, w, [], dy) ;

% Check the derivative numerically
figure(2) ; clf('reset') ;
set(gcf,'name','Part 2.2: two layers backrpop') ;
func = @(x) proj(p, vl_nnrelu(vl_nnconv(x, w, []))) ;
checkDerivativeNumerically(func, x, dx) ;

%% Part 2.3: Design and verify your own layer

x0 = randn(size(x), 'single') ;

y = customLayerForward(x, x0) ;

p = randn(size(y), 'single') ;
dx = customLayerBackward(x, x0, p) ;

% Check the derivative numerically
figure(3) ; clf('reset') ;
set(gcf,'name','Part 2.3: custom layer') ;
func = @(x) proj(p, customLayerForward(x, x0)) ;
checkDerivativeNumerically(func, x, dx) ;
