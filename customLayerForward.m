function y = customLayerForward(x,r)

dif = x - r ;
y = sum(dif(:).^2) ;

y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
