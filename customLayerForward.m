function y = customLayerForward(x,r)
y = sum(sum(sum((x - r).^2, 1), 2), 3) ;
