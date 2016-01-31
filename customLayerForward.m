function y = customLayerForward(x,r)
y = sum(sum(sum((x - r).^2, 1), 2), 3) ;
y = y / (size(x,1) * size(x,2)) ;