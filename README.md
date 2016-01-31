Convolutional neural network practical (2)
==========================================

A computer vision practical by the Oxford Visual Geometry group,
authored by Andrea Vedaldi, Karel Lenc, and Joao Henriques.

Start from `doc/instructions.html`.

> Note that this practical requires compiling the (included)
> MatConvNet library. This should happen automatically (see the
> `setup.m` script), but make sure that the compilation succeeds on
> the laboratory computers.

Package contents
----------------

The practical consists of four exercises, organized in the following
files:

* `exercise1.m` -- Part 1: Building blocks: convolution and ReLU
* `exercise2.m` -- Part 2: Derivatives and backpropagation
* `exercise3.m` -- Part 3: Learning a CNN for text deblurring

The practical runs in MATLAB and uses
[MatConvNet](http://www.vlfeat.org/matconvnet). This package contains
the following MATLAB functions:

* `checkDerivativeNumerically.m`: check a layer derivatives numerically.
* `customLayerForward.m` and `customLayerBackward.m`: code (partially) implementing a custom layer.
* `getBatch.m:`: get a batch of images for training.
* `getCustomLayer.m`: get the custom layer in SimpleNN format.
* `initializeSmallCNN.m` and `initializeLargeCNN.m`: initialize CNN models for text deblurring.
* `setup.m`: setup MATLAB environment.
* `showDeblurringResult.m`: show results for the deblurring network.
* `showFeatureChannels.m`: show the feature channels in a tensor.
* `xavier.m`: Xaiver's initialization of the network weights.

Appendix: Installing from scratch
---------------------------------

The practical requires both VLFeat and MatConvNet. VLFeat comes with
pre-built binaries, but MatConvNet does not.

0. Set the current directory to the practical base directory.
1. From Bash:
   1. Run `git submodule update -i` to download the submodules.
   2. Run `make -f ./extras/Makefile preproc`. This will create a copy
      of the data for the practical
2. From MATLAB run `addpath extra ; preprocess ;`. This will create
   `data/text_imdb.mat`.
3. Test the practical: from MATLAB run all the exercises in order.

Changes
-------

* *2016a* - Initial edition

License
-------

    Copyright (c) 16 Andrea Vedaldi, Karel Lenc, and Joao Henriques

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
