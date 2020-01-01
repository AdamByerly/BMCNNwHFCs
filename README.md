
# A Merged Multi-Level Convolutional Network with Homogeneous Filter Capsules

This repository contains the code used for the experiments detailed in a forthcoming paper. The paper will be available pre-published at arXiv very soon.

## Required Libraries
To train models ([python/training](python/training)):
-   TensorFlow (see  [http://www.tensorflow.org](http://www.tensorflow.org))
-   NumPy (see  [http://www.numpy.org](http://www.numpy.org))
-   OpenCV (see [http://opencv.org](http://opencv.org))
-   At least one GPU

To produce GIFs of the MNIST evaluation digits ([python/etc/produce_MNIST_eval_digits.py](python/etc/produce_MNIST_eval_digits.py)):
- Pillow (see [https://python-pillow.org/](https://python-pillow.org/))

To extract the scalars from the tensorflow events.out.tfevents file created during training into CSV formatted data ([python/etc/extract_scalars_from_logs.py](python/etc/extract_scalars_from_logs.py)): 
- Pandas (see [https://pandas.pydata.org](https://pandas.pydata.org))

To evaluate all possible combinations of ensembles of models ([C++](C%2B%2B)):
- The NVIDIA CUDA Toolkit (see [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit))

## Reproducing the Results

Coming soon...

&nbsp;
&nbsp;

Maintained by Adam Byerly (abyerly@fsmail.bradley.edu)