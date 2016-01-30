# VGG Convolutional Neural Networks Practical

*By Andrea Vedaldi and Andrew Zisserman*

This is an [Oxford Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg) computer vision practical, authored by [Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi/) and Andrew Zisserman (Release 2015a).

<img height=400px src="images/cover.png" alt="cover"/>

*Convolutional neural networks* are an important class of learnable representations applicable, among others, to numerous computer vision problems. Deep CNNs, in particular, are composed of several layers of processing, each involving linear as well as non-linear operators, that are learned jointly, in an end-to-end manner, to solve a particular tasks. These methods are now the dominant approach for feature extraction from audiovisual and textual data.

This practical explores the basics of learning (deep) CNNs. The first part introduces typical CNN building blocks, such as ReLU units and linear filters, with a particular emphasis on understanding back-propagation. The second part looks at learning two basic CNNs. The first one is a simple non-linear filter capturing particular image structures, while the second one is a network that recognises typewritten characters (using a variety of different fonts). These examples illustrate the use of stochastic gradient descent with momentum, the definition of an objective function, the construction of mini-batches of data, and data jittering. The last part shows how powerful CNN models can be downloaded off-the-shelf and used directly in applications, bypassing the expensive training process.

[TOC]

$$
   \newcommand{\bx}{\mathbf{x}}
   \newcommand{\by}{\mathbf{y}}
   \newcommand{\bz}{\mathbf{z}}
   \newcommand{\bw}{\mathbf{w}}
   \newcommand{\bp}{\mathbf{p}}
   \newcommand{\cP}{\mathcal{P}}
   \newcommand{\cN}{\mathcal{N}}
   \newcommand{\vc}{\operatorname{vec}}
   \newcommand{\vv}{\operatorname{vec}}
$$

## Getting started

Read and understand the [requirements and installation instructions](../overview/index.html#installation). The download links for this practical are:

* Code and data: [practical-cnn-reg-2016a.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-reg-2016a.tar.gz)
* Code only: [practical-cnn-reg-2016a-code-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-reg-2016a-code-only.tar.gz)
* Data only: [practical-cnn-reg-2016a-data-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-reg-2016a-data-only.tar.gz)
* [Git repository](https://github.com/vedaldi/practical-cnn) (for lab setters and developers)

After the installation is complete, open and edit the script `exercise1.m` in the MATLAB editor. The script contains commented code and a description for all steps of this exercise, for [Part I](#part1) of this document. You can cut and paste this code into the MATLAB window to run it, and will need to modify it as you go through the session. Other files `exercise2.m`, `exercise3.m`, and `exercise4.m` are given for [Part II](#part2), [III](#part3), and [IV](part4).

Each part contains several **Questions** (that require pen and paper) and **Tasks** (that require experimentation or coding) to be answered/completed before proceeding further in the practical.

## Part 1: CNN bludling blocks {#part1}

In this part we will explore two fundamental bulding blocks of CNNs, linear convolution and non-linear activation functions. Open `exercise1.m` and run the `setup()` command as explained above.

### Part 1.1: convolution {#part1.1}

A *convolutional neural network* (CNN) is a sequence of linear and non-linear  convolutional operators. The most important example of a convolutional operator is *linear convolution*. In this part, we will explore linear convolution and see how to use it in MatConvNet. 

#### Part 1.1.1: convolution by a single filter {#part1.1.1}

Start by identifying and then running the following code fragment in `exercise1.m`:

```.language-matlab
% Load an image and convert it to gray scale and single precision
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

% Define a filter
w = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

% Apply the filter to the image
y = vl_nnconv(x, w, []) ;
```

The code loads the image `data/ray.jpg` and applies to it a linear filter using the linear convolution operator. The latter is implemented by the MatConvNet function `vl_nnconv()`. Note that all variables `x`, `w`, and `y` are in single precision; while MatConvNet supports double precision arithmetic too, single precision is usually preferred in applications where memory is a bottleneck. The result can be visualized as follows:

```.language-matlab
% Visualize the results
figure(1) ; clf ; colormap gray ;
set(gcf,'name','P1.1: convolution') ;

subplot(1,3,1) ;
imagesc(x) ;
axis off image ;
title('input image x') ;

subplot(1,3,2) ;
imagesc(w) ;
axis off image ;
title('filter w') ;

subplot(1,3,3) ;
subplot(1,3,3) ;
imagesc(y) ;
axis off image ;
title('output image y') ;
```

> **Task:** Run the code above and examine the result, which should look like the following image:
> <img height=400px src="images/conv.png" alt="cover"/>

Let's examine know what happened. The input $\bx$ to the linear convolution operator is an $M \times N$ matrix, which can be interpreted as a gray-scale image. The filter $\bw$ is the $3 \times 3$ matrix
$$
\bw = 
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0 \\
\end{bmatrix}
$$
The output of the convolution is a new matrix $\by$ given by
$$
y_{ij} = \sum_{uv} w_{uv}\ x_{i+u,\ j+v}
$$
> **Remark**: if you are familiar with convolution as defined in mathematics and signal processing, you might expect to find the index $i-u$ instead of $i+u$ in this expression. The convention $i+u$, which is often used in CNNs, is  often referred to as correlation.

> **Questions:**
> 
> 1. If $H \times W$ is the size of the input image, $H' \times W'$ the size of the filter, what is the size $H'' \times W''$ of the output image?
> 2. The filter $\bw$ given above is a discretized Laplacian operator, so that the output image is the Laplacian of the input. This filter respond particularly strongly to certain structures in the image. Which ones?

#### Part 1.1.2: convolution by a filter bank {#part1.1.2}

In neural networks, one usually operates with *filter banks* instead of individual filters. Each filter can be though of as computing a different *feature channel*, characterizing a particular statistical property of the input image.

To see how to define and use a filter bank, create a bank of three filters as follows:

```.language-matlab
% Concatenate three filters in a bank
w1 = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

w2 = single([
  -1 0 +1
  -1 0 +1
  -1 0 +1]) ;

w3 = single([
  -1 -1 -1
  0 0 0 
  +1 +1 +1]) ;
  
wbank = cat(4, w1, w2, w3) ;
```

The first filter $\bw_1$ is the Laplacian operator seen above; two additional filters $\bw_2$ and $\bw_3$ are horizontal and vertical image derivatives, respectively. Note that the same command `vl_nnconv(x, wbank, [])` works with a filter bank as well. However, the output `y` is not just a matrix, but a 3D  array (often called a *tensor* in the CNN jargon). This tensor has dimensions $H \times W \times K$, where $K$ is the number of *feature channels*.

> **Question:** What is the number of feature channels $C$ in this example? Why?

> **Task:** Run the code above and visualize the individual feature channels in the tensor `y` by using the provided function `showFeatureChannels()`. Do the channel responses make sense given the filter used to generate them?

In a CNN, not only the output tensor, but also the input tensor `x` and the filters `wbank` can have multiple feature channels. In this case, the convolution formula becomes:
$$
y_{ijk} = \sum_{uvp} w_{uvpk}\ x_{i+u,\ j+v,\ p}
$$

> **Questions:** 
> 
> * If the input tensor $\bx$ has $C$ 
> * In the code above, the command `wbank = cat(4, w1, w2, w3)` concatenates the tensors `w1`, `w2`, and `w3` along the *fourth dimension*. Why is that given that filters should have three dimensions?

### Part 1.2: non-linear activation (ReLU) {#part1.2}

CNNs are obtained by composing several operators, individually called *layers*. In addition to convolution and other linear layers, CNNs should contain non-linear layers as well.

> **Question:** What happens if all layers are linear?

The simplest non-linearity is given by scalar activation functions, which are applied independently to each element in a tensor. Perhaps the simples, and one of the most effective, examples is the *Rectified Linear Unit* (ReLU) operator:
$$
   y_{ijk} = \max \{0, x_{ijk}\}
$$
which simply cuts-off any negative value.

In MatConvNet, ReLU is implemented by the `vl_nnrelu` function. To demonstrate its use, we convolve the test image with the negated Laplacian, and then apply ReLU to the result:

```.language-matlab
% Convolve with the negated Laplacian
y = vl_nnconv(x, - w, []) ;

% Apply the ReLU operator
z = vl_nnrelu(y) ;
```

> **Task:** Run this code and visualize images `x`, `y`, and `z`.

> **Questions:** 
> 
> * Which kind of image structures are preferred by this filter? 
> * Why did we negate the Laplacian?

ReLU has a very important effect as it implicitly sets to zero the majority of the filter responses. In a certain sense, ReLU works as a detector, with the implicit convention that a certain pattern is detected when a corresponding filter response is large enough (greater than zero).

In practice, while signals are centered and therefore a threshold of zero is reasonable, in practice there is no particular reason why this should always be appropriate. For this reason, the convolution code allows to specify *a bias term* for each filter response. Let's use this term to make the response of ReLU more selective:

```.language-matlab
bias = single(- 0.2) ;
y = vl_nnconv(x, - w, bias) ;
z = vl_nnrelu(y) ;
```

There is only one `bias` term because there is only one filter in the bank (note that, as for the reset of the data, `bias` is a single precision number). The bias is applied after convolution, effectively subtracting 0.2 from the filter responses. Hence, now a response is not suppressed by the subsequent ReLU operator only if it is at least 0.2.

> **Task:** Run this code and visualize images `x`, `y`, and `z`.

> **Question:** Is the response now more selective?

> **Remark:** There are many other building blocks used in CNNs, the most important of which is perhaps max pooling. However, convolution and ReLU can solve already many problems, as we will see in the remainder of the practical.

## Part 2: backpropagation {#part2}

Training CNNs is normally done using a gradient-based optimization method. The CNN $f$ is the composition of $L$ layers $f_l$ each with parameters $\bw_l$, which in the simplest case of a chain looks like:
$$
 \bx_0
 \longrightarrow 
 \underset{\displaystyle\underset{\displaystyle\bw_1}{\uparrow}}{\boxed{f_1}} 
 \longrightarrow
 \bx_1
 \longrightarrow
 \underset{\displaystyle\underset{\displaystyle\bw_2}{\uparrow}}{\boxed{f_2}}
 \longrightarrow
 \bx_2 
 \longrightarrow
 \dots
 \longrightarrow
 \bx_{L-1}
 \longrightarrow
 \underset{\displaystyle\underset{\displaystyle\bw_2}{\uparrow}}{\boxed{f_L}}
 \longrightarrow
 \bx_L
$$
During learning, the last layer of the network is the *loss function* that should be minimized. Hence, the output $\bx_L = x_L$ of the network is a **scalar** quantity.

The gradient is easily computed using using the **chain rule**. If *all* network variables and parameters are scalar, this is given by[^derivative]:
$$
 \frac{\partial f}{\partial w_l}(x_0;w_1,\dots,w_L)
 =
 \frac{\partial f_L}{\partial x_L}(x_L;w_L) \times
 \cdots
 \times
 \frac{\partial f_{l+1}}{\partial x_l}(x_l;w_{l+1}) \times
 \frac{\partial f_{l}}{\partial w_l}(x_{l-1};w_l) 
$$
With tensors, however, there are some complications. Consider for instance the derivative of a function $\by=f(\bx)$ where both $\by$ and $\bx$ are tensors; this is formed by taking the derivative of each scalar element in the output $\by$ w.r.t. each scalar element in the input $\bx$. If $\bx$ has dimensions $H \times W \times C$ and $\by$ has dimensions $H' \times W' \times C'$, then the derivative contains $HWCH'W'C'$ elements, which is often unmanageable (often of the order of several GBs of memory).

Importantly, all intermediate derivatives in the chain rule are affected by this size explosion, but, since the output is a scalar, the output derivative are not.

> **Question:** The output derivatives have the same size as the parameters in the network. Why?

**Back-propagation** allows computing the output derivatives in a memory-efficient manner. To see how, the first step is to generalize the equation above to tensors using a matrix notation. This is done by converting tensors into vectors by using the $\vv$ (stacking)[^stacking] operator:
$$
 \frac{\partial \vv f}{\partial \vv^\top \bw_l}
 =
 \frac{\partial \vv f_L}{\partial \vv^\top \bx_L} \times
 \cdots
 \times
 \frac{\partial \vv f_{l+1}}{\partial \vv^\top \bx_l} \times
 \frac{\partial \vv f_{l}}{\partial \vv^\top \bw_l} 
$$
The next step is to *project* the derivative with respect to a tensor $\bp_L = 1$ as follows:
$$
 (\vv \bp_L)^\top \times \frac{\partial \vv f}{\partial \vv^\top \bw_l}
 =
 (\vv \bp_L)^\top
 \times
 \frac{\partial \vv f_L}{\partial \vv^\top \bx_L} \times
 \cdots
 \times
 \frac{\partial \vv f_{l+1}}{\partial \vv^\top \bx_l} \times
 \frac{\partial \vv f_{l}}{\partial \vv^\top \bw_l} 
$$
Note that $\bp_L=1$ has the same dimension as $\bx_L$ (the scalar loss) and, being the identity, does not change anything. This gets interesting when products are evaluated from the left to the right, i.e. *backward from the output to the input* of the CNN. The first such factors is given by:
\begin{equation}
\label{e:factor}
 (\vv \bp_{L-1})^\top = (\vv \bp_L)^\top
 \times
 \frac{\partial \vv f_L}{\partial \vv^\top \bx_L}
\end{equation}
This results in a new projection vector $\bp_{L-1}$ which can in turn be multiplied to obtain $\bp_{L-2}, \dots, \bp_l$. The last projection $\bp_l$ is the desired derivative. Crucially, each projection $\bp_q$ takes as much memory as the corresponding variable $\bp_q$.

The most attentive reader might have noticed that, while projections remain small, each factor \eqref{e:factor} does contain one such large derivatives. The trick is that CNN toolboxes implement, for each layer, a backward mode that computes the projected derivative without explicitly computing these factors.

In particular, for any building block function $\by=f(\bx;\bw)$, a CNN toolbox implements:

* A **forward mode** computing the function $\by=f(\bx;\bw)$.
* A **backward mode** computing the derivatives of the projected function $\langle \bp, f(\bx;\bw) \rangle$ with respect to the input $\bx$ and parameter $\bw$:

$$
\frac{\partial}{\partial \bx} \left\langle \bp, f(\bx;\bw) \right\rangle,
\qquad
\frac{\partial}{\partial \bw} \left\langle \bp, f(\bx;\bw) \right\rangle.
$$

### Backpropagation interface {#part2.1}

All the building blocks in MatConvNet support forward and backward computation and hence can be used in backpropagation. This is how it looks like for the convolution operator:

```.language-matlab
y = vl_nnconv(x,w,b) ; % forward mode (get output)
p = randn(size(y), 'single') ; % random projection
[dx,dw,db] = vl_nnconv(x,w,b,p) ; % backward mode (get projected derivatives)
```

and this is hat it looks like for ReLU operator:

```.language-matlab
y = vl_nnreli(x) ;
p = randn(size(y), 'single') ;
dx = vl_nnrelu(x,p) ;
```

### Backward mode in one layer {#part2.2}

Implementing new building blocks in a network is conceptually quite easy. However, it is also quite easy to make silly mistakes in computing the derivatives analytically, or in implementing them in software. Therefore, it is *highly recommended* to check derivatives numerically if you implement your own. This is also useful to understand what the backward mode does, so we look at this problem next.

Identify in `exercise2.m` the following code fragment and evaluate it, up to the visualization.

```.language-matlab
% Forward mode: evaluate the convolution
y = vl_nnconv(x, w, []) ;

% Pick a random projection tensor
p = randn(size(y), 'single') ;

% Backward mode: projected derivatives
[dx,dw] = vl_nnconv(x, w, [], p) ;

% Check the derivative numerically
delta = 0.01 ;
dx_numerical = zeros(size(dx), 'single') ;
for i = 1:numel(x)
  xp = x ; 
  xp(i) = xp(i) + delta ;
  yp = vl_nnconv(xp,w,[]) ;
  dx_numerical(i) =  p(:)' * (yp(:) - y(:)) / delta ;
end
```

> **Tasks:**
> 
> 1.   Run the code, visualizing the results. Convince yourself that the numerical and analytical derivatives are nearly identical.
> 2.   Modify the code to compute the derivative of the *first element* of the output tensor $\by$ with respect to *all the elements* of the input tensor $\bx$. **Hint:** it suffices to change the value of $\bp$.
> 2.   Modify the code to compute the derivative w.r.t. the convolution parameters $\bw$ instead of the convolution input $\bx$.

### Backward mode in two or more layers {#part2.3}
The backward mode is more interesting when at least two network layer are involved. Next, we append a ReLU layer to the convolutional one:

```.language-matlab
% Forward mode: evaluate the conv + ReLU
y = vl_nnconv(x, w, []) ;
z = vl_nnrelu(y) ;

% Pick a random projection tensor
p = randn(size(z), 'single') ;

% Backward mode: projected derivatives
dy = vl_nnrelu(z, p) ;
[dx,dw] = vl_nnconv(x, w, [], dy) ;
```

> **Question (important)** In the code above, in backward mode the projection `p` is fed to the `vl_nnrelu` operator. However, the `vl_nnconv` operator now receives `dy` as projection. Why?

> **Tasks:**
>
> 1.  Run the code and visualize the analytical and numerical derivatives. Do they differ?
> 2.  (Optional) Modify the code above to a chain of three layers: conv + ReLU + conv.

## Part 3: Learning a CNN for text deblurring {#part3}

In this part of the practical, we will learn a CNN that generates an image instead of performing classification. This is a simple demonstration of how CNNs can be used well beyond classification tasks.

The goal of the exercise is to learn a function that takes a blurred text as input and produces a crispier version as output. This problem is generally known as *deblurring* and is widely studied in computer vision, image processing, and computational photography. Here, instead of constructing a deblurring filter from first principles, we simply learn it from data. A key advantage is that the learned function can incorporate a significant amount of domain-specifc knowledge and perform particularly well on the particualr domain of interst.

Start by opening in your MATLAB editor `exercise2.m`.

### Part 3.1: preparing the data {#part3.1}

The first task is to load the training and validation data and to understand its format. The code responsible for loading such data is

```.language-matlab
imdb = load('data/text_imdb.mat') ;
```

The variable `imdb` is a structure containing $n$ images. The structure has the following fields:

* `imdb.images.data`: a $64 \times 64 \times 1 \times n$ array of grayscale images.
* `imdb.images.label`: a $64 \times 64 \times 1 \times n$ of image "labels"; for this problem, a label is also a grayscale image.
* `imdb.images.set`: a $1 \times n$ vector containing a 1 for training images and an 2 for validation images.

Each trainig datapoint is a blurred image of text (extracted from a scientific paper). Its "label" is the sharp version of the same image: learning the deblurring function is formulated as the problem of regressing the sharp image from the blurred one.

Run the following code, which displays the first image and corresponding label in the dataset:

```.language-matlab
figure(100) ; clf ;

subplot(1,2,1) ; imagesc(imdb.images.data(:,:,:,1)) ;
axis off image ; title('input (blurred)') ;

subplot(1,2,2) ; imagesc(imdb.images.label(:,:,:,1)) ;
axis off image ; title('desired output (sharp)') ;

colormap gray ;
```

It should produce the following output:

![Data example](images/text.png)

The images are split in 75% training images and 25% validation images, as indicated by the flag `imdb.images.set`.

> **Task:** make sure you understand the data format. How many training and validation images are there? What is the resolution of the individual images?

### Part 3.2: preparing the network

The next task is to construct a network `net` and initialize its weights. We are going to use the SimpleNN wrapper in MatConvNet (more complex architectures can be implemented using the DagNN wrapper).

A network is simply a sequence of functions. We initialize this as the empty list:

```.language-matlab
net.layers = { } ;
```

Layers need to be listed in `net.layers` in the order of execution, from first to last. For example, the following code adds the first layer of the network, a convolutional one:

```.language-matlab
net.layers{end+1} = struct(...
  'name', 'conv1', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,1,32)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;
```

The `name` field specifies a name for the layer, useful for debugging but otherwise arbitrary. The `type` field specifised which type of layer we want, in this case convolution. The `weights` layer is a cell array containing two arrays, one for the filters and one for the biases. In this example, the filter array has dimension $3 \times 3 \times 1 \times 32$, which is a filter bank of 32 filters, with $3\times 3$ spatial support, and operating on 1 input channel. The biases array has dimension $32 \times 1$ as it specifies one bias per filter. These two arrays are initialize randomly by the `xavier` function, using Xiavier's method.

The `pad` and `stride` options specify the filter padding and stride. Here the stride is dense (one pixel) and there is a one pixel padding. In this manne, the output tensor has exactly the same spatial dimensions of the input one.

Finally, the `learningRate` and `weightDecay` options specify filter-specific multipliers for the learning rate and weight decay for the filters and the biases.


The next layer to be added is simply a ReLU activation function, which is non-linear:

```.language-matlab
net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;
```

The architecture consists of a number of such convolution-ReLU blocks.

> **Question:** The last block, generating the final image, is only convolutional, and it has exactly one filter. Why?

This is still not sufficient to learn the model. We need in facto a loss function too. For this task, we use the Euclidean distance between the generated image and the desired one. This is implemented by the `pdist` block.

```.language-matlab
net.layers{end+1} = struct(...
  'name', 'loss', ...
  'type', 'pdist', ...
  'p', 2, ...
  'aggregate', true, ...
  'instanceWeights', 1/(64*64)) ;
```

Here `p` is the exponent of the p-distance (set to 2 for Eucliden), `aggregate` means that the individual squared pixel differences should be summed in a grand total for the whole image, and `instanceWeights` is a (homogeneous) scaling factors for all the pixels, set to the inverse area of an image. The latter two options make it so that the squared difference are averaged across pixels, resulting in a normalized Euclidean distance between generated and target images.

We add one last parameter

```.language-matlab
net.meta.inputSize = [64 64 1 1] ;
```

which  specifies the expected dimensions of the network input. Finally, we call the `vl_simplenn_tidy()` function to check the network parameters and fill in default values for any parameter that we did not specify yet:

```.language-matlab
net = vl_simplenn_tidy(net) ;
```

Finally, we display the parameters of the network just created:


```.language-matlab
vl_simplenn_display(net) ;
```

> **Questions:** Look carefully at the generated table and answer the following questions:
>
> 1. How many layers are in this network?
> 3. What is the dimensionality of each intermediate feature map? How is that related with the number of filters in each convolutional layer?
> 4. Is there a special relationship between number of channels in a feature map and a certain dimension of the following filter bank?
> 5. What is the receptive field size of each feature? Is that proportionate to the size of a character?

### Part 3.3: learning the network {#part3.3}

```.language-matlab
trainOpts.expDir = opts.expDir ;
trainOpts.batchSize = 16 ;
trainOpts.learningRate = 0.01 ;
trainOpts.numEpochs = 30 ;
trainOpts.gpus = [] ;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;
```

The `getBatch()` function is a particularly important one. The training script `cnn_train` takes a *handle* to this function and uses it whenever a new batch of data is required. By passing appropriate custom functions, `cnn_train` does not need to know almost anything about the dataset, making it fairly general. In practice, `getBatch()` is often simple:


```.language-matlab
function [im, label] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
label = imdb.images.label(:,:,:,batch) ;
```

The function takes as input the `imdb` structure defined above and a list `batch` of image indexes that should be returned for training. It does so by copying the relative images to the array `im`; it also copies the corresponding labels to `label`. In more complex cases, it may load images from disk on the fly, and-or apply some kind of transformation to the data (for example for data augmentation purposes).

## Links and further work

* The code for this practical is written using the software package [MatConvNet](http://www.vlfeat.org/matconvnet). This is a software library written in MATLAB, C++, and CUDA and is freely available as source code and binary.
* The ImageNet model is the *VGG very deep 16* of Karen Simonyan and Andrew Zisserman.

## Acknowledgements

* Beta testing by: Karel Lenc and Carlos Arteta.
* Bugfixes/typos by: Sun Yushi.

## History

* Used in the Oxford AIMS CDT, 2015-16.
* Used in the Oxford AIMS CDT, 2014-15.

[^derivative]: The derivative is computed with respect to a certain assignment $x_0$ and $(w_1,\dots,w_L)$ to the network input and parameters; furthermore, the intermediate derivatives are computed at points $x_1,\dots,x_L$ obtained by evaluating the network at $x_0$.

[^stacking]: The stacking operator $\vv$ simply unfolds a tensor in a vector by stacking its elements in some pre-defined order.

[^lattice]: A two-dimensional *lattice* is a discrete grid embedded in $R^2$, similar for example to a checkerboard.
