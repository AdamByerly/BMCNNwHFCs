## Training Models
The only script you can execute directly in this directory is ``train.py``.
All other scripts in this directory contain code called by ``train.py`` and can not be executed directly.

Here is an example execution:
````
python train.py --merge_strategy=2 --batch_size=120 --gpus=1 --trials=32
	--log_dir=../../data/learnable_ones_init/weights
	--data_dir=../../data/mnist_data
````

#### Parameters
```
--merge_strategy
``` 
 **Optional**.
Use this to specify the manner in which the branches in the architecture are merged.
If not specified, this value will default to 2.

Valid values for ``merge_strategy`` are:
 - 0 - Use this to specify that the branches are to be merged with equal weight.
 - 1 - Use this to specify that the branches are to be merged with learned weights, such that those weights were intialized randomly.
 - 2 - Use this to specify that the branches are to be merged with learned weights, such that those weights were intialized to 1.

```
--ema_decay_rate
``` 
 **Optional**.
Test accuracy is measured using the exponential moving average of prior weights.
Use this to specity the decay rate used for the exponential moving average.
If not specified, this value will default to 0.999.

```
--start_epoch
```
**Optional**.
You would only want to override this value if you were resuming training a model that was stopped for some reason.  In that case, you would set this value to the number of the next epoch you want to train relative to the epoch after which the weights you are starting with were saved. (see `--weights_file` below)
If not specified, this value will default to 1.

```
--end_epoch
```
**Optional**.
Model training will continue until this many epochs have run, or something else stops it preemptively.
If not specified, this value will default to 300.

```
--run_name
```
**Optional**.
A unique name to associate with this training of the model.
When specified explicity, the name is appended with an underscore and the trial number (see ``--trials`` below).
If not specified, this value will default to an amalgamation of the digits taken from the current date and time.
For example:  20191229094854.
This run was created on December 29, 2019 at 9:48:54 AM local time.  When resuming a previously halted experiment, you will want to provide the run name that was used for that experiment in this parameter.
```
--weights_file
```
**Optional**.
In the event that you want to restart a previously interrupted training session, you'll need to provide the last saved weights as a starting point.  Note that when you provide a weights file, you'll also want to set the --start_epoch parameter to the next epoch following the epoch for which the weights were saved in the specified file.
If not specified, this value will default to None.

```
--profile_compute_time_steps
``` 
 **Optional**.
If you are using different GPUs or a different number of GPUs or are otherwise curious, you can set this flag to some positive integer.  100 is my suggestion.  This will cause tensorflow to profile the compute time used by the devices (CPUs and GPUs) on your machine each time this number of training steps has passed.  You can then look in TensorBoard, under the Graphs tab, and see a list of Session runs to choose from (one for each time the compute time was profiled).  You can then examine how compute time is being used during training for each of your devices.
If not specified, this value will default to None.

```
--save_summary_info_steps
``` 
 **Optional**.
This paramter can be used to track a histogram of values as they change during trainng.
If you are diving deep into this network, set this to a positive integer that represents how often you want to save values to be included in the histograms.
No values are specified to be stored in the network design as it currently exists; you'd have to add them yourself if you are curious.
Note that saving these values both takes disk space and slows down the training a little, so you would likely only want to use this exploratorily.  See [https://www.tensorflow.org/guide/tensorboard_histograms](https://www.tensorflow.org/guide/tensorboard_histograms) for more on adding values to be saved for displaying in histograms.
If not specified, this value will default to None.

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
Use this to specify the number of GPUs available on the local machine to be trained on.
If not specified, this value will default to 1.
Note that at least one GPU is required.

```
--trials
``` 
 **Optional**.
Use this to specify how many times to train an independent model.
You can think of this as the number of iterations to execute a _for_ loop around the entire training process.
When not specifying a value for ``--run_name``, a new name is generated for each round of training based on the time when that specific round started.
When specifying an explicit value for ``--run_name``, for each round of training, that name is appended with an underscore and the trial number.
If not specified, this value will default to 32.

```
--log_dir
``` 
 **Optional**.
Use this to indicate where to place both (a) the scalar event data output during training and (b) the weights saved during trainng.
The training procedure saves multiple copies of weights during the training process:
- the weights from the 2 best top-1 prediction accuracies (_as evaluated at the end of each epoch_)
- the weights from the 2 lowest losses (_as evaluated at the end of each epoch_)
- the weights at the end of each of the latest 2 epochs

If not specified, the files will be placed in the relative directory location ``../../data/learnable_ones_init/weights``. 

```
--data_dir
``` 
 **Optional**.
Use this to specify the location of the MNIST data files in their original format.
If not specified, the files will be looked for in the relative directory location ``../../data/mnist_data``. 

