The python code in this directory includes:
 - One script (``ensemble_evaluations.py``) used in creating ensembles from individual models.
 - Three scripts (``extract_scalars_from_logs.py``, ``produce_MNIST_eval_digits.py``, and ``count_vars.py``) that produce useful data for analysis, but are otherwise unrelated to the process of conducting the experiments. 

## ensemble_evaluations.py
This script is used to iterate over the "best" weights for trials in an experiment and make predictions for the MNIST validation data based on those weights.
For our experiments we used the highest top-1 prediction accuracy achieved at any point during training as the "best", but, concieveably, any set of weights could be chosen.
This script produces up 3 outputs:
1. **ensemble_data.txt** - This file is used by the binary compiled from the C++/CUDA code in this repository to search for ensembles.  (_See [here](../../C%2B%2B#evaluating-ensemble-model-combinations) for more information regarding the correct interpretation of this file's contents._)
2. Assuming the ``--output_images`` parameter is set to ``True``, renderings of the MNIST evaluation digits that no model in the experiment predicted correctly.  Note: _Files are named with the following pattern:_ ``XXXX(Y).gif`` _, where_``XXXX`` _is the position of the image in the MNIST evluation data and_ ``Y`` _is the label for that image._
3. Assuming the ``--output_images`` parameter is set to ``True``, renderings of the MNIST evaluation digits that were predicted correctly by at least one model in the experiment and incorrectly by at least one model.  _See the note in_ (2) _above regarding the interpretation of the files' names in this directory._

This script is executed thusly:
````
python ensemble_evaluations.py --batch_size=120 --gpus=1
    --output_all_logits=False --output_images=True
    --data_dir=../../data/mnist_data
    --images_dir=../../data/images/all
    --log_dir=../../data/learnable_ones_init/ensemble_weights
    --output_dir=../../data/learnable_ones_init
````

#### Parameters
```
--batch_size
``` 
 **Optional**.
Use this to specify the number of images that are processed at a time and before each pass of back-propagation.
If not specified, this value will default to 120.

```
--gpus
``` 
 **Optional**.
Use this to specify the number of GPUs available on the local machine.
If not specified, this value will default to 1.
Note that at least one GPU is required.

```
--output_all_logits
``` 
 **Optional**.
Use this to indicate that, for every validation image, all logits are to be output, not just the class of the highest probability.
This parameter was set to ``False`` for all experiments we performed.
This is because the ensembling method we used (majority vote) for our experiments only uses the most likely prediction.
The ensemble finding code provided in this repository is capable of ensembling via sum or product of the individual logits, and if you wanted to use one of those ensembling methods, you would need to output all of the logits (i.e. set this parameter to ``True``).
If not specified, this value will default to ``False``.

```
--output_images
``` 
 **Optional**.
Use this to indicate that you would like this script to produce renderings of the MNIST evaluation digits that either (a) no model in the experiment predicted correctly, or (b) were predicted correctly by at least one model in the experiment and incorrectly by at least one model.
If not specified, this value will default to ``True``.

```
--data_dir
``` 
 **Optional**.
Use this to specify the location of the MNIST data files in their original format.
If not specified, the files will be looked for in the relative directory location ``../../data/mnist_data``. 

```
--images_dir
``` 
 **Optional**.
If you have specified ``True`` for ``--output_images``, then provide the location of the already rendered MNIST evaluation digits.
If not specified, the files will be placed in the relative directory location ``../../data/images/all``.

```
--log_dir
``` 
 **Optional**.
Use this to specify the directory into which you copied the weights to use from each model you trained.
These weights will be used to make predictions for the MNIST validation data.
Note that this directory will only exist after a manual process that is described [here](../../README.md#ensembles).
If not specified, the files will be looked for in the relative directory location ``../../data/learnable_ones_init/ensemble_weights``.

```
--output_dir
``` 
 **Optional**.
Use this to specify the directory into which the ``ensemble_data.txt`` file this process generates will be placed.
If not specified, the file will be placed in the relative directory location ``../../data/learnable_ones_init``.

## extract_scalars_from_logs.py
This script can be used to convert the scalar data recorded during training and read by tensorboard to the more readily readable CSV data.

This script can be executed thusly:
````
python extract_scalars_from_logs.py
    --event_data_dir=../../data/learnable_ones_init/weights
    --output_dir=../../data/learnable_ones_init
````

#### Parameters
```
--event_data_dir
``` 
 **Optional**.
Use this to specify the location of the scalar event data output during training.  This will be the same directory as the directory passed to the ``--log_dir`` parameter when training models with ``python/training/train.py``.
If not specified, the files will be looked for in the relative directory location ``../../data/learnable_ones_init/weights``. 

````
--output_dir
````
 **Optional**.
Use this to specify the location in which the files generated by this script are to be placed.
If not specified, the files will be looked for in the relative directory location ``../../data/learnable_ones_init``. 

## produce_MNIST_eval_digits.py
This script can be used to convert the MNIST evaluation data into visually assessable GIF files.

The script is executed thusly:
````
python produce_MNIST_eval_digits.py --data_dir=../../data/mnist_data
    --output_dir=../../data/images/all --batch_size=120
````

#### Parameters
```
--data_dir
``` 
 **Optional**.
Use this to specify the location of the MNIST data files in their original format.
Specifically, ``t10k-images-idx3-ubyte`` and ``t10k-labels-idx1-ubyte`` are required by this script.
If not specified, the files will be looked for in the relative directory location ``../../data/mnist_data``. 

````
--output_dir
````
 **Optional**.
Use this to specify the location in which the files generated by this script are to be placed.
If not specified, the files will be placed in the relative directory location ``../../data/images/all``.

````
--batch_size
````
 **Optional**.
Use this to specify the minmum number of images that are to be loaded into the input stream and processed at a time.
If not specified, this value will default to 120.

## count_vars.py
This script can be used to count and report the number of trainable parametres (i.e. weights) in a model architecture.

The script is executed thusly:
````
python count_vars.py --merge_strategy=2
````

Valid values for ``merge_strategy`` are:
 - 0 - Use this to specify that the branches are to be merged with equal weight.
 - 1 - Use this to specify that the branches are to be merged with learned weights (such that those weights are intialized randomly--which is irrelevant for the purpose of counting trainable variables).
 - 2 - Use this to specify that the branches are to be merged with learned weights (such that those weights are intialized to 1--which is irrelevant for the purpose of counting trainable variables).
