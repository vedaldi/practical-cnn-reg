function go()

run matconvnet/matlab/vl_setupnn
addpath matconvnet/examples

net.layers = { } ;

% convolution part
net.layers{end+1} = struct('name', 'conv1', 'type', 'conv', ...
  'weights', {{ones(3,3,1,1,'single')/9,[]}}, ... %xavier(3,3,1,8)}, ...
  'pad', 1, ...
  'stride', 2, ...
  'learningRate', [0 0.0001], 'weightDecay', [1 0 ]) ;
%net.layers{end+1} = struct('name', 'relu1', 'type', 'relu') ;
%net.layers{end+1} = struct('name', 'pool1', 'type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2, 'pad', 1) ;

%net.layers{end+1} = struct('name', 'conv2', 'type', 'conv', 'weights', {xavier(3,3,8,16)}, 'pad', 1) ;
%net.layers{end+1} = struct('name', 'relu2', 'type', 'relu') ;
%net.layers{end+1} = struct('name', 'pool2', 'type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2, 'pad', 1) ;

%net.layers{end+1} = struct('name', 'conv3', 'type', 'conv', 'weights', {xavier(3,3,16,32)}, 'pad', 1) ;
%net.layers{end+1} = struct('name', 'relu3', 'type', 'relu') ;
%net.layers{end+1} = struct('name', 'pool3', 'type', 'pool', 'method', 'max', 'pool', [3 3], 'stride', 2, 'pad', 1) ;

% deconvolution part
%net.layers{end+1} = struct('name', 'convt4', 'type', 'convt', 'weights', {xaviert(3,3,16,32)}, 'upsample', 2, 'crop', [0 1 0 1]) ;
%net.layers{end+1} = struct('name', 'relu4', 'type', 'relu') ;

%net.layers{end+1} = struct('name', 'convt5', 'type', 'convt', 'weights', {xaviert(3,3,8,16)}, 'upsample', 2, 'crop', [0 1 0 1]) ;
%net.layers{end+1} = struct('name', 'relu5', 'type', 'relu') ;

f = single([.5 1 .5]' * [.5 1 .5]) ;
f = ones(3,3,'single')/9 ;

net.layers{end+1} = struct('name', 'convt6', 'type', 'convt', ...
  'weights', {{f,[]}}, ... %xaviert(3,3,1,8)}, ...
  'upsample', 2, 'crop', [1 0 1 0], ...
  'learningRate', [1 0.00001], 'weightDecay', [1 0]) ;
%net.layers{end+1} = struct('name', 'relu6', 'type', 'relu') ;

net.layers{end+1} = struct('name', 'loss', 'type', 'pdist', 'noRoot', true, 'aggregate', true) ;
net = vl_simplenn_tidy(net) ;

sigma = 5 ;
numSamples = 256 ;
maxNumObjects = 20 ;
scale = 128 * 5 * 5 ;

for k = 1 : numSamples
  n = randi(maxNumObjects) ;
  [im, loc] = sampleImage(n) ;
  im = im / scale ;
  density = computeDensity(loc, sigma) ;
  imdb.images.image(:,:,1,k) = im ;
  imdb.images.label(:,:,1,k) = density ;
  imdb.images.n(k) = n ;
  imdb.images.loc(:,:,1,k) = loc ;
end
imdb.images.set = [ones(1,numSamples/4*3), 2*ones(1,numSamples/4)] ;

k = 1 ;
net.layers{end}.class = imdb.images.label(:,:,k) ;
res = vl_simplenn(net, imdb.images.image(:,:,k), [], [],  'conserveMemory', false) ;

net = cnn_train(net, imdb, @getBatch,  'errorFunction', 'none', 'batchSize', 16, ...
  'learningRate', [0.1 * ones(1,10), 0.01*ones(1,10)], ...
  'numEpochs', 20, ...
  'continue', true, 'expDir', ...
  'data/exp3', ...
  'plotDiagnostics', true, ...
  'conserveMemory', false) ;

net.layers{end-1}.precious = 1 ;

figure(2) ;
error = 0 ;
for k = 1 : numSamples
  im = imdb.images.image(:,:,1,k) ;
  label = imdb.images.label(:,:,1,k) ;
  net.layers{end}.class = label ;
  
  res = vl_simplenn(net, im) ;
  error = error + res(end).x ;
  
  clf ;
  subplot(3,1,1) ; imagesc(im) ; axis image ;
  subplot(3,1,2) ; imagesc(label) ; axis image ;  
  subplot(3,1,3) ; imagesc(res(end-1).x) ; axis image ;
  pause(1) ;
end
error = error / numSamples ;

function [im, label] = getBatch(imdb, batch)
im = imdb.images.image(:,:,:,batch) ;
label = imdb.images.label(:,:,:,batch) ;

function density = computeDensity(loc, sigma)
m = round(sigma*6) ;
z = fspecial('gaussian', m, sigma) ;
density = conv2(loc,z,'same') ;

function [im, loc] = sampleImage(n)
% sample locations
w = 128 ;
loc = zeros(w,'single') ;
s = randperm(numel(loc)) ;
loc(s(1:n)) = 1 ;
% create rendition
im = computeDensity(loc, 5) ;
im = im / max(im(:)) * 128 ;
%im = im > max(im(:))/2 ;

function weights = xavier(varargin)
rng(0) ;
filterSize = [varargin{:}] ;
scale = sqrt(2/prod(filterSize(1:3))) ;
filters = randn(filterSize, 'single') * scale ;
biases = zeros(filterSize(4),1,'single') ;
weights = {filters, biases} ;

function weights = xaviert(varargin)
rng(0) ;
filterSize = [varargin{:}] ;
scale = sqrt(2/prod(filterSize(1:3))) ;
filters = randn(filterSize, 'single') * scale ;
biases = zeros(filterSize(3),1,'single') ;
weights = {filters, biases} ;
    