function weights = xavier(varargin)
%XAVIER  Xavier filter initialization.
%   WEIGHTS = XAVIER(H, W, C, N) initializes N filters of support H x
%   W and C channels using Xavier method. WEIGHTS = {FILTERS,BIASES}is
%   a cell array containing both filters and biases.

rng(1) ;
filterSize = [varargin{:}] ;
scale = sqrt(2/prod(filterSize(1:3))) ;
filters = randn(filterSize, 'single') * scale ;
biases = zeros(filterSize(4),1,'single') ;
weights = {filters, biases} ;
