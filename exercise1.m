setup() ;  

%% Part 1.1: convolution

% Part 1.1.1: convolution by a single filter

% Load an image and convert it to gray scale and single precision
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

% Define a filter
w = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

% Apply the filter to the image
y = vl_nnconv(x, w, []) ;

% Visualize the results
figure(1) ; clf ; colormap gray ;
set(gcf,'name','P1.1: convolution') ;

subplot(1,3,1) ;
imagesc(x) ;
axis off image ;
title('input image x') ;

subplot(1,3,2) ;
imagesc(w) ;
axis off image ;
title('filter w') ;

subplot(1,3,3) ;
subplot(1,3,3) ;
imagesc(y) ;
axis off image ;
title('output image y') ;

% Part 1.1.2: convolution by a bank of filters

% Concatenate three fitlers in a bank
w1 = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

w2 = single([
  -1 0 +1
  -1 0 +1
  -1 0 +1]) ;

w3 = single([
  -1 -1 -1
  0 0 0 
  +1 +1 +1]) ;
  
wbank = cat(4, w1, w2, w3) ;

% Apply convolution
y = vl_nnconv(x, wbank, []) ;

% Show feature channels
figure(2) ; clf ; set(gcf,'name','P1.1.2: channels') ;
colormap gray ;
showFeatureChannels(y) ;

%% Part 1.2: non-linear activation functions (ReLU)

% Part 1.2.1: Laplacian and ReLU

% Convolve with the negated Laplacian
y = vl_nnconv(x, - w, []) ;

% Apply the ReLU operator
z = vl_nnrelu(y) ;

figure(4) ; clf ; set(gcf,'name','P1.2.1: Laplacian anr ReLU') ;
colormap gray ;
subplot(1,3,1); imagesc(x) ; axis off image ; title('image x') ;
subplot(1,3,2); imagesc(y) ; axis off image ; title('Laplacian y')
subplot(1,3,3); imagesc(z) ; axis off image ; title('ReLU z') ;

% Part 1.2.2: adding a bias

bias = single(- 0.2) ;
y = vl_nnconv(x, - w, bias) ;
z = vl_nnrelu(y) ;

figure(5) ; clf ; set(gcf,'name','P1.2.2: adding a bias') ;
colormap gray ;
subplot(1,3,1); imagesc(x) ; axis off image ; title('image x') ;
subplot(1,3,2); imagesc(y) ; axis off image ; title('Laplacian y with bias')
subplot(1,3,3); imagesc(z) ; axis off image ; title('ReLU z') ;
