function dx = customLayerBackward(x,x0,p)
dx = 2 * bsxfun(@times, p, x - x0) ;
dx = dx / (size(x,1) * size(x,2)) ;