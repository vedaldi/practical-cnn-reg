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
   \newcommand{\cP}{\mathcal{P}}
   \newcommand{\cN}{\mathcal{N}}
   \newcommand{\vc}{\operatorname{vec}}
$$

## Getting started

Read and understand the [requirements and installation instructions](../overview/index.html#installation). The download links for this practical are:

* Code and data: [practical-cnn-2015a.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a.tar.gz)
* Code only: [practical-cnn-2015a-code-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a-code-only.tar.gz)
* Data only: [practical-cnn-2015a-data-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a-data-only.tar.gz)
* [Git repository](https://github.com/vedaldi/practical-cnn) (for lab setters and developers)

After the installation is complete, open and edit the script `exercise1.m` in the MATLAB editor. The script contains commented code and a description for all steps of this exercise, for [Part I](#part1) of this document. You can cut and paste this code into the MATLAB window to run it, and will need to modify it as you go through the session. Other files `exercise2.m`, `exercise3.m`, and `exercise4.m` are given for [Part II](#part2), [III](#part3), and [IV](part4).

Each part contains several **Questions** (that require pen and paper) and **Tasks** (that require experimentation or coding) to be answered/completed before proceeding further in the practical.

## Part 2: Learning a CNN for text deblurring {#part1}

In this part of the practical, we will learn a CNN that generates an image instead of performing classification. This is a simple demonstration of how CNNs can be used well beyond classification tasks.

The goal of the exercise is to learn a function that takes a blurred text as input and produces a crispier version as output. This problem is generally known as *deblurring* and is widely studied in computer vision, image processing, and computational photography. Here, instead of constructing a deblurring filter from first principles, we simply learn it from data. A key advantage is that the learned function can incorporate a significant amount of domain-specifc knowledge and perform particularly well on the particualr domain of interst.

Start by opening in your MATLAB editor `exercise2.m`.

### Part 2.1: preparing the data {#part1.1}

The first task is to load the training and validation data and to understand its format. The code responsible for loading such data is

```matlab
imdb = load('data/text_imdb.mat') ;
```

The variable `imdb` is a structure containing $n$ images. The structure has the following fields:

* `imdb.images.data`: a $64 \times 64 \times 1 \times n$ array of grayscale images.
* `imdb.images.label`: a $64 \times 64 \times 1 \times n$ of image "labels"; for this problem, a label is also a grayscale image.
* `imdb.images.set`: a $1 \times n$ vector containing a 1 for training images and an 2 for validation images.

Each trainig datapoint is a blurred image of text (extracted from a scientific paper). Its "label" is the sharp version of the same image: learning the deblurring function is formulated as the problem of regressing the sharp image from the blurred one.

Run the following code, which displays the first image and corresponding label in the dataset:

```matlab
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

## Part 2.2: preparing the network

The next task is to construct a network `net` and initialize its weights. We are going to use the SimpleNN wrapper in MatConvNet (more complex architectures can be implemented using the DagNN wrapper).

A network is simply a sequence of functions. We initialize this as the empty list:

```matlab
net.layers = { } ;
```

Layers need to be listed in `net.layers` in the order of execution, from first to last. For example, the following code adds the first layer of the network, a convolutional one:

```matlab
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

```matlab
net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;
```

The architecture consists of a number of such convolution-ReLU blocks.

> **Question:** The last block, generating the final image, is only convolutional, and it has exactly one filter. Why?

This is still not sufficient to learn the model. We need in facto a loss function too. For this task, we use the Euclidean distance between the generated image and the desired one. This is implemented by the `pdist` block.

```matlab
net.layers{end+1} = struct(...
  'name', 'loss', ...
  'type', 'pdist', ...
  'p', 2, ...
  'aggregate', true, ...
  'instanceWeights', 1/(64*64)) ;
```

Here `p` is the exponent of the p-distance (set to 2 for Eucliden), `aggregate` means that the individual squared pixel differences should be summed in a grand total for the whole image, and `instanceWeights` is a (homogeneous) scaling factors for all the pixels, set to the inverse area of an image. The latter two options make it so that the squared difference are averaged across pixels, resulting in a normalized Euclidean distance between generated and target images.

We add one last parameter

```matlab
net.meta.inputSize = [64 64 1 1] ;
```

which  specifies the expected dimensions of the network input. Finally, we call the `vl_simplenn_tidy()` function to check the network parameters and fill in default values for any parameter that we did not specify yet:

```matlab
net = vl_simplenn_tidy(net) ;
```

Finally, we display the parameters of the network just created:


```matlab
vl_simplenn_display(net) ;
```

> **Questions:** Look carefully at the generated table and answer the following questions:
>
> 1. How many layers are in this network?
> 2. What is the sampling density of each layer?
> 3. What is the dimensionality of each intermediate feature map? How is that related with the number of filters in each convolutional layer?
> 4. Is there a special relationship between number of channels in a feature map and a certain dimension of the following filter bank?
> 5. What is the receptive field size of each feature? Is that proportionate to the size of a character?

## Links and further work

* The code for this practical is written using the software package [MatConvNet](http://www.vlfeat.org/matconvnet). This is a software library written in MATLAB, C++, and CUDA and is freely available as source code and binary.
* The ImageNet model is the *VGG very deep 16* of Karen Simonyan and Andrew Zisserman.

## Acknowledgements

* Beta testing by: Karel Lenc and Carlos Arteta.
* Bugfixes/typos by: Sun Yushi.

## History

* Used in the Oxford AIMS CDT, 2015-16.
* Used in the Oxford AIMS CDT, 2014-15.

[^lattice]: A two-dimensional *lattice* is a discrete grid embedded in $R^2$, similar for example to a checkerboard.
