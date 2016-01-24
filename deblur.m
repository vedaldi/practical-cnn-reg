function deblur()

%% Initialization
opts.dataDir = 'data/deblur' ;
opts.expDir = 'data/deblur-exp6' ;

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

imdbPath = fullfile(opts.dataDir, 'imdb4.mat') ;
if exist(imdbPath) ;
  imdb = load(imdbPath) ;
else
  imdb = getImdb(opts.dataDir) ;
  save(imdbPath, '-struct', 'imdb') ;
end

figure(100) ; clf ;
subplot(1,2,1) ; imagesc(imdb.images.data(:,:,:,1)) ; axis off image ; title('input (blurred)') ;
subplot(1,2,2) ; imagesc(imdb.images.label(:,:,:,1)) ; axis off image ; title('desired output (sharp)') ;
colormap gray ;

%% Network
net.layers = { } ;

net.layers{end+1} = struct(...
  'name', 'conv1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,1,32)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,32,64)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv3', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,64,64)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu3', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'prediction', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,64,1)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'loss', ...
  'type', 'pdist', ...
  'p', 2, ...
  'aggregate', true, ...
  'instanceWeights', 1/(64*64)) ;

net.meta.inputSize = [64 64 1 1] ;

% check and prepare network
net = vl_simplenn_tidy(net) ;

% display network
vl_simplenn_display(net) ;

net.layers{end}.class = ones(1,1,1,1,'single') ;
res = vl_simplenn(net, imdb.images.data(:,:,:,1)) ;

net = cnn_train(net, imdb, @getBatch, ...
  'expDir', opts.expDir, ...
  'numEpochs', 30, ...
  'learningRate', 0.01 * [ones(1,30)], ...
  'continue', true, ...
  'plotDiagnostics', false, ...
  'conserveMemory', false, ...
  'batchSize', 16, ...
  'gpus', [2], ...
  'errorFunction',  'none') ;

% evalaute
net.layers(end) =  [] ; %drop loss layer

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

figure(101) ; set(101,'name','Resluts on the training set') ; clf ;
show(net, imdb, train(1:30:151)) ;

figure(102) ; set(102,'name','Resluts on the validation set') ; clf ;
show(net, imdb, val(1:30:151)) ;

% -------------------------------------------------------------------------
function show(net, imdb, subset)
% -------------------------------------------------------------------------
res = vl_simplenn(net, imdb.images.data(:,:,:,subset)) ;
preds = res(end).x ;
n = numel(subset) ;
for i = 1 : n
  j = subset(i) ;
  subplot(n,3,1+3*(i-1)) ;
  imagesc(imdb.images.data(:,:,:,j),[-1 0]) ;
  axis off image ; title('original') ;
  subplot(n,3,2+3*(i-1)) ;
  imagesc(imdb.images.label(:,:,:,j),[-1 0]) ;
  axis off image ; title('expected') ;
  subplot(n,3,3+3*(i-1)) ;
  imagesc(preds(:,:,:,i),[-1 0]) ;
  axis off image ; title('achieved') ;
end
colormap gray ;

% -------------------------------------------------------------------------
function imdb = getImdb(dataDir)
% -------------------------------------------------------------------------
imdb.images.id = {} ;
imdb.images.data = {} ;
imdb.images.set = {} ;
imdb.images.label = {} ;

names = dir(fullfile(dataDir, '*.png')) ;
names = {names.name}  ;

for i = 1:numel(names)
  im = imread(fullfile(dataDir, names{i})) ;
  im = im2single(im) ;
  if size(im,3) > 1, im = rgb2gray(im) ; end
  im = im - 1 ; % make white = 0
  label = im ;

  G = fspecial('gaussian', [5 5], 2);
  im = imfilter(label,G,'same') ;
  s = 1+ (i > numel(names)*.75) ;

  % further break each image in 64 x 64 tiles
  for i = 0:7
    for j = 0:7
      si = i*64 + (1:64) ;
      sj = j*64 + (1:64) ;
      im_ = im(si,sj) ;
      label_ = label(si,sj) ;
      % drop if nothing in the patch
      if std(im_(:)) < 0.05, continue ; end
      imdb.images.id{end+1} = numel(imdb.images.id) + 1 ;
      imdb.images.set{end+1} = s ;
      imdb.images.label{end+1} = label_ ;
      imdb.images.data{end+1} = im_ ;
    end
  end
end

imdb.images.id = horzcat(imdb.images.id{:}) ;
imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.label = cat(4, imdb.images.label{:}) ;
imdb.images.data = cat(4, imdb.images.data{:}) ;

%m = mean(imdb.images.data(imdb.images.set==1)) ;
%imdb.images.data = imdb.images.data - m ;
%imdb.images.label = imdb.images.label - m ;

% -------------------------------------------------------------------------
function weights = xavier(varargin)
% -------------------------------------------------------------------------
rng(1) ;
filterSize = [varargin{:}] ;
scale = sqrt(2/prod(filterSize(1:3))) ;
filters = randn(filterSize, 'single') * scale ;
biases = zeros(filterSize(4),1,'single') ;
weights = {filters, biases} ;

% -------------------------------------------------------------------------
function [im, label] = getBatch(imdb, batch)
% -------------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
label = imdb.images.label(:,:,:,batch) ;
