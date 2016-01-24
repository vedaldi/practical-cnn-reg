function setup(varargin)
%SETUP()  Initialize the practical
%   SETUP() initializes the practical. SETUP('useGpu', true) does
%   the same, but compiles the GPU supprot as well.

base = fileparts(mfilename('fullpath')) ;
run(fullfile(base, 'matconvnet', 'matlab', 'vl_setupnn')) ;

opts.useGpu = false ;
opts.verbose = false ;
opts = vl_argparse(opts, varargin) ;

addpath(fullfile(base, 'matconvnet', 'examples')) ;

try
  vl_nnconv(single(1),single(1),[]) ;
catch
  warning('VL_NNCONV() does not seem to be compiled. Trying to compile it now.') ;
  vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
end

if opts.useGpu
  try
    vl_nnconv(gpuArray(single(1)),gpuArray(single(1)),[]) ;
  catch
    warning('GPU support does not seem to be compiled in MatConvNet. Trying to compile it now.') ;
    vl_compilenn('enableGpu', opts.useGpu, 'verbose', opts.verbose) ;
  end
end
