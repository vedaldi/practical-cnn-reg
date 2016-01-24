function deblur()

%% Initialization
opts.dataDir = 'data/deblur' ;
opts.expDir = 'data/deblur-exp3' ;

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

imdbPath = fullfile(opts.dataDir, 'imdb.mat') ;
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
  'weights', {xavier(3,3,1,8)}, ...
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
  'weights', {xavier(3,3,8,16)}, ...
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
  'weights', {xavier(3,3,16,32)}, ...
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
  'weights', {xavier(3,3,32,1)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'loss', ...
  'type', 'pdist', ...
  'p', 2, ...
  'aggregate', true, ...
  'instanceWeights', 1/(512*512)) ;

net.meta.inputSize = [512 512 1 1] ;

% check and prepare network
net = vl_simplenn_tidy(net) ;

% display network
vl_simplenn_display(net) ;

net.layers{end}.class = ones(1,1,1,1,'single') ;
res = vl_simplenn(net, imdb.images.data(:,:,:,1)) ;

net = cnn_train(net, imdb, @getBatch, ...
  'expDir', opts.expDir, ...
  'numEpochs', 40, ...
  'learningRate', 0.01 * [ones(1,20), 0.1*ones(1,20)], ...
  'continue', false, ...
  'plotDiagnostics', false, ...
  'conserveMemory', false, ...
  'batchSize', 1, ...
  'train', 1*ones(1,20), ...
  'val', 1, ...
  'errorFunction',  'none') ;

% evalaute
net.layers(end) =  [] ; %drop loss layer
res = vl_simplenn(net, imdb.images.data) ;

figure(101) ; clf ;
subplot(1,3,1) ; imagesc(imdb.images.data(:,:,:,end)) ; axis off image ; title('input (blurred)') ;
subplot(1,3,2) ; imagesc(imdb.images.label(:,:,:,end)) ; axis off image ; title('output desired (sharp)') ;
subplot(1,3,3) ; imagesc(res(end).x(:,:,:,end)) ; axis off image ; title('output achieved') ;
colormap gray ;

keyboard

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
  label = im ;
   
  G = fspecial('gaussian', [5 5], 2);
  im = imfilter(label,G,'same') ;
  s = 1+ (i > numel(names)*.75) ;

  imdb.images.id{end+1} = numel(imdb.images.id) + 1 ;
  imdb.images.set{end+1} = s ;
  imdb.images.label{end+1} = label ;
  imdb.images.data{end+1} = im ;
end
    
imdb.images.id = horzcat(imdb.images.id{:}) ;
imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.label = cat(4, imdb.images.label{:}) ;
imdb.images.data = cat(4, imdb.images.data{:}) ;

m = mean(imdb.images.data(imdb.images.set==1)) ;
imdb.images.data = imdb.images.data - m ;
imdb.images.label = imdb.images.label - m ;

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
    

