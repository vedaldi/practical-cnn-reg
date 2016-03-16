function dx = l1LossBackward(x,r,p)
% TODO Comment the next line and implement
dx = rand(size(x), 'like', x);
dx = dx / (size(x,1) * size(x,2)) ;  % normalize by image size
