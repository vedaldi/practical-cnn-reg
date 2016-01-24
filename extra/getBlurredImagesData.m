function imdb = getBlurredImagesData(dataDir)
%GETBLURREDIMAGESDATA  Get the data for the text deblurring exercise
%   IMDB = GETBLURREDIMAGESDATA(DATADIR) reads a directory of PNG
%   images DATADIR and returns a corresponding IMDB structure.

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
