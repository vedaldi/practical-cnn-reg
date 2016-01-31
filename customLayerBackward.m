function dx = customLayerBackward(x,r,p)
dx = 2 * bsxfun(@times, p, x - r) ;
dx = dx / (size(x,1) * size(x,2)) ;