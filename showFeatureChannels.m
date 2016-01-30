function showFeatureChannels(x)
%SHOWFEATURECHANNELS  Display the feature channels in the tensor x

k = size(x,3) ;
n = ceil(sqrt(k)) ;
m = ceil(k/n) ;

for i = 1:k
  subplot(m,n,i) ; imagesc(x(:,:,i)) ;
  title(sprintf('feature channel %d',i)) ; axis image ;
end