function preprocess()
% Run the Makefile first

opts.dataDir = 'data/text/' ;
opts.imdbPath = 'data/text_imdb.mat' ;

setup() ;

if ~exist(opts.imdbPath) ;
  imdb = getBlurredImagesData(opts.dataDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

