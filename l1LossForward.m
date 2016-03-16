function y = l1LossForward(x,r)
% TODO Comment the next line and implement
y = rand(size(x), 'like', x);
y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
