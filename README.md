
# A Branching and Merging Convolutional Network with Homogeneous Filter Capsules

![license Apache 2](https://img.shields.io/static/v1?label=license&message=Apache%202&color=blue "license Apache 2")
![version v1.0.1](https://img.shields.io/static/v1?label=version&message=v1.0.1&color=orange "version v1.0.1")
![codefactor A](https://www.codefactor.io/Content/badges/A.svg "codefactor A")
[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](http://arxiv.org/abs/2001.09136)
[![DOI](https://zenodo.org//badge/DOI/10.5281/zenodo.3596980.svg)](https://doi.org/10.5281/zenodo.3596980)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-branching-and-merging-convolutional-network/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=a-branching-and-merging-convolutional-network)
[![Tweeting](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=A%20Branching%20and%20Merging%20Convolutional%20Network%20with%20Homogeneous%20Filter%20Capsules%0AGitHub:&amp;url=https://github.com/AdamByerly/BMCNNwHFCs&amp;hashtags=NeuralNetwork,ConvolutionalNeuralNetwork,MNIST,StateOfTheArt,ImageClassification)

This repository contains the code used for the experiments detailed in a forthcoming paper. The paper is available pre-published at arXiv: http://arxiv.org/abs/2001.09136

## Required Libraries
To train models ([python/training](python/training#training-models)):
-   TensorFlow (see  [http://www.tensorflow.org](http://www.tensorflow.org))
-   NumPy (see  [http://www.numpy.org](http://www.numpy.org))
-   OpenCV (see [http://opencv.org](http://opencv.org))
-   At least one GPU

To produce GIFs of the MNIST evaluation digits ([python/etc/produce_MNIST_eval_digits.py](python/etc#produce_mnist_eval_digitspy)):
- Pillow (see [https://python-pillow.org/](https://python-pillow.org/))

To extract the scalars from the tensorflow events.out.tfevents file created during training into CSV formatted data ([python/etc/extract_scalars_from_logs.py](python/etc#extract_scalars_from_logspy)): 
- Pandas (see [https://pandas.pydata.org](https://pandas.pydata.org))

To evaluate all possible combinations of ensembles of models ([C++](C%2B%2B#evaluating-ensemble-model-combinations)):
- The NVIDIA CUDA Toolkit (see [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit))

## Reproducing the Results

### Single Models
Training a single model is relatively simple and can be accomplished with the execution of a single script.  See [python/training](python/training#training-models) for more information.

### Ensembles

Evaluating ensembles of multiple models is a bit more complicated and involves several manual steps.

First, you need to train the individual models that you will be evaluating for potential inclusion in an ensemble.

Note: _For our experiments, we trained 32 single models, resulting in ~2^32 potential ensembles.  As the number of models appears in the exponent in this expression, adding to the number of models to be evaluated for inclusion in an ensemble quite literally expands the search space exponentially._

Once you have done this, you should have an output directory for each individual model, within which are the saved weights for that model.
 
What follows is a truncated list of directories from one of our experiments:
````
...
20191229094854
20191229135730
20191229180414
...
````

Next, you need to choose the weights to use from each individual model you trained.  Our training procedure saves multiple copies of weights during the training process:
- the weights from the 2 best top-1 prediction accuracies (_as evaluated at the end of each epoch_)
- the weights from the 2 lowest losses (_as evaluated at the end of each epoch_)
- the weights at the end of each of the latest 2 epochs

Within each output directory that was created you will have files that correspond to these (_noting that Tensorflow creates 3 files for each set of weights_).  See the following example:

````
weights-43-best_top1-10750.data-00000-of-00001
weights-43-best_top1-10750.index
weights-43-best_top1-10750.meta
weights-46-best_top1-11500.data-00000-of-00001
weights-46-best_top1-11500.index
weights-46-best_top1-11500.meta
weights-119-best_loss-29750.data-00000-of-00001
weights-119-best_loss-29750.index
weights-119-best_loss-29750.meta
weights-120-best_loss-30000.data-00000-of-00001
weights-120-best_loss-30000.index
weights-120-best_loss-30000.meta
weights-299-latest-74749.data-00000-of-00001
weights-299-latest-74749.index
weights-299-latest-74749.meta
weights-300-latest-74999.data-00000-of-00001
weights-300-latest-74999.index
weights-300-latest-74999.meta
````

For our experiments, we used the best top-1 prediction accuaracies found during the training of the models.  As a new set of weights for top-1 predictions is saved only when a new, higher top-1 accuracy is achieved, the saved top-1 weights from the highest numbered epoch represent the best top-1 prediction weights found during the duration of the model's training. 

As such, to follow our process in the example provided above, you'd choose the three files that are named beginning with ``weights-46-best_top1``. (_The number after_ ``weights-`` _in these files is the epoch number that the weights were saved after._)

You must now create a copy of each model's output directory with  your chosen weights files (only) in them.  What follows is a truncated list of directories and the list of all files copied into them from one of our experiments:
````
...
|-- 20191229094854
|   |-- weights-128-best_top1-32000.data-00000-of-00001
|   |-- weights-128-best_top1-32000.index
|   |-- weights-128-best_top1-32000.meta
|
|-- 20191229135730
|   |-- weights-113-best_top1-28250.data-00000-of-00001
|   |-- weights-113-best_top1-28250.index
|   |-- weights-113-best_top1-28250.meta
|
|-- 20191229180414
    |-- weights-50-best_top1-12500.data-00000-of-00001
    |-- weights-50-best_top1-12500.index
    |-- weights-50-best_top1-12500.meta
...
````

After you have made these copies, you will need to execute ``python/etc/ensemble_evaluations.py`` from this repository, pointing that script at the weights files you have just copied.
The output of this script is a file named ``ensemble_data.txt``.
(_See [here](python/etc#ensemble_evaluationspy) for more information._)

You will then use ``ensemble_data.txt`` as an input into the binary compiled from the C++/CUDA code in this repository.
(_See [here](C%2B%2B#evaluating-ensemble-model-combinations) for more information._)

&nbsp;
&nbsp;

Maintained by Adam Byerly (abyerly@fsmail.bradley.edu)
