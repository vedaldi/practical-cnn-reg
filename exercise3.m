function exercise2()

%% Initialization
%
% The `setup()` command initializes the practical by including
% MatConvNet. Next, a database of images `text_imdb.mat` is loaded.

% Initialize the practical
setup() ;

% Choose a directory to save the experiment files
opts.expDir = 'data/text-exp-1' ;

% Learning parameters
trainOpts.expDir = opts.expDir ;
trainOpts.batchSize = 16 ;
trainOpts.learningRate = 0.01 ;
trainOpts.numEpochs = 30 ;
trainOpts.gpus = [] ;
trainOpts.errorFunction = 'none' ;


%% Part 2.1: Prepare the data

% Load a database of blurred images to train from
imdb = load('data/text_imdb.mat') ;

% Visualize the first image in the database
figure(100) ; clf ;

subplot(1,2,1) ; imagesc(imdb.images.data(:,:,:,1)) ;
axis off image ; title('input (blurred)') ;

subplot(1,2,2) ; imagesc(imdb.images.label(:,:,:,1)) ;
axis off image ; title('desired output (sharp)') ;

colormap gray ;


%% Part 2.2: Create a network architecture
%
% The expected input size (a single 64 x 64 x 1 image patch). This is
% used for visualization purposes.

net.meta.inputSize = [64 64 1 1] ;

% Add one layer at a time

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


% Consolidate the network, fixing any missing option
% in the specification above

net = vl_simplenn_tidy(net) ;

% Display network

vl_simplenn_display(net) ;

%% Step 3: learn the model

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deployment: remove the last layer
net.layers(end) = [] ;


%% Step 4: evaluate the model

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

figure(101) ; set(101,'name','Resluts on the training set') ;
showDeblurringResult(net, imdb, train(1:30:151)) ;

figure(102) ; set(102,'name','Resluts on the validation set') ;
showDeblurringResult(net, imdb, val(1:30:151)) ;

% -------------------------------------------------------------------------
function [im, label] = getBatch(imdb, batch)
% -------------------------------------------------------------------------
% The GETBATCH() function is used by the training code to extract the
% data required fort training the network.

im = imdb.images.data(:,:,:,batch) ;
label = imdb.images.label(:,:,:,batch) ;
