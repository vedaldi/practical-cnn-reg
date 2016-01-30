function y = customLayerForward(x, x0)
y = sum(sum(sum((x - x0).^2, 1), 2), 3) ;
y = y / (size(x,1) * size(x,2)) ;