function showDeblurringResult(net, imdb, subset)
%SHOWDEBLURRINGRESULT  Show a few examples of deblurred images
%   SHOWDEBLURRINGRESULTS(NET, IMDB, SUBSET) uses the CNN NET to
%   deblur a few images in the IMDB database and visualzie the result
%   in a figure. SUBSET is a vector of image indexes to display.

% Evaluate the CNN to obtain deblurring results
res = vl_simplenn(net, imdb.images.data(:,:,:,subset)) ;
preds = res(end).x ;

% Visualize the results in a figure
clf ;
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
