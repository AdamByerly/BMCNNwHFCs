
# No Routing Needed Between Capsules

![license Apache 2](https://img.shields.io/static/v1?label=license&message=Apache%202&color=blue "license Apache 2")
![version v1.0.1](https://img.shields.io/static/v1?label=version&message=v2.0.1&color=orange "version v1.0.1")
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

##
...more coming soon...
