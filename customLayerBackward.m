function dx = customLayerBackward(x,r,p)
dx = 2 * bsxfun(@times, p, x - r) ;
