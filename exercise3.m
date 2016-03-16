setup() ;

%% Part 3.1: Prepare the data

% Load a database of blurred images to train from
imdb = load('data/text_imdb.mat') ;

% Visualize the first image in the database
figure(2) ; set(2, 'name', 'Part 3.1: Data') ; clf ;

subplot(1,2,1) ; imagesc(imdb.images.data(:,:,:,1)) ;
axis off image ; title('Input (blurred)') ;

subplot(1,2,2) ; imagesc(imdb.images.label(:,:,:,1)) ;
axis off image ; title('Desired output (sharp)') ;

colormap gray ;

%% Part 3.2: Create a network architecture
%
% The expected input size (a single 64 x 64 x 1 image patch). This is
% used for visualization purposes.

net = initializeSmallCNN() ;
%net = initializeLargeCNN() ;

% Display network
vl_simplenn_display(net) ;

% Evaluate network on an image
res = vl_simplenn(net, imdb.images.data(:,:,:,1)) ;

figure(3) ; clf ; colormap gray ;
set(gcf,'name', 'Part 3.2: network input') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

set(gcf,'name', 'Part 3.2: network output') ;
subplot(1,2,2) ;
imagesc(res(end).x) ; axis image off  ;
title('CNN output (not trained yet)') ;

%% Part 3.3: learn the model

% Add a loss (using our custom layer), if not already present
if ~strcmp(net.layers{end}.type, 'loss')
  net.layers{end+1} = getCustomLayer() ;
end

% Train
trainOpts.expDir = 'data/text-small' ;
trainOpts.batchSize = 16 ;
trainOpts.learningRate = 0.02 ;
trainOpts.plotDiagnostics = false ;
trainOpts.numEpochs = 20 ;
trainOpts.gpus = [] ;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 3.4: evaluate the model

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

figure(4) ; set(4, 'name', 'Part 3.4: Results on the training set') ;
showDeblurringResult(net, imdb, train(1:30:151)) ;

figure(5) ; set(5, 'name', 'Part 3.4: Results on the validation set') ;
showDeblurringResult(net, imdb, val(1:30:151)) ;

figure(6) ;
set(6, 'name', 'Part 3.4: Larger example on the validation set') ;
colormap gray ;
subplot(1,2,1) ; imagesc(imdb.examples.blurred{1}, [-1, 0]) ;
axis image off ;
title('CNN input') ;
res = vl_simplenn(net, imdb.examples.blurred{1}) ;
subplot(1,2,2) ; imagesc(res(end).x, [-1, 0]) ;
axis image off ;
title('CNN output') ;
