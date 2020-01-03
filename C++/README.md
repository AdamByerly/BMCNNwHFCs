## Evaluating Ensemble Model Combinations
For each of our experiments, we generated 32 models.
We then needed to search the combination space of those models for ensembles.

Given that we wanted to evaluate every possible ensemble of those models, we had ~2^32 evaluations to perform for each experiment.

After executing our python implementaiton for accomplishing this for ~24 hours, we extrapolated and determined that it would need 40 days to complete.

So we decided to implement in C++/CUDA so that we could take advantage of GPU accelerated execution.
This implementation executes in roughly ~1 hour on an NVIDIA GeForce 940MX for the same data that was on track to take ~40 days.

Example usage:
````
EnsembleFind ../data/learnable_ones_init/ensemble_data.txt majority_vote 9982
````

#### Parameters
All parameters are required.
1. The first parameter is a the location of an ``ensemble_data.txt`` file that was created by executing ``ensemble_evaluations.py`` (See [here](python/etc#ensemble_evaluationspy).)
2. The second parameter is the ensembling method to use.  See _Ensembling Methods_ below for more information.
3. The third parameter is the accuracy reporting threshold.  Ensembles that achieve this level of accuracy or higher will be output.

#### Ensembling Methods
Although our experiments only used the majority vote ensembling method, the process supports 3 types of ensembling:
1. ``majority_vote`` - this method is the simplest and most common form of ensembling.  It simply involves counting a prediction as correct if the majority of models in the ensemble predicted it correctly.
2. ``product`` - this method involves multiplying together all of the logits for each validatoin sample from each of the models in the ensemble, then the largest value is counted as the prediction for the sample.
3. ``sum`` - this method is the same as the ``product`` method except that the values are added together, not multiplied.

In order to use the ``majority_vote`` method, the execution of ``ensemble_evaluations.py`` that created the ``ensemble_data.txt`` must have been called with ``--output_all_logits`` set to ``False`` or omitted.

In order to use either the ``product`` or ``sum`` methods, the execution of ``ensemble_evaluations.py`` that created the ``ensemble_data.txt`` must have been called with ``--output_all_logits`` set to ``True``.

#### Example Output

What follows is a truncated output from the process.
Every line in the ouput begins with either "Evaluating", "Complete", or a number.
Those lines that begin with "Evaluating" or "Complete" are process status lines.

Lines that begin with numbers are ensembles that have been found that are as accurate or more accurate than the accuracy reporting threshold the process was called with. 
The number is the accuracy.
Following the number and a colon are the list of models that achieved that accuracy.

````
Evaluating all 4960 ensemble combinations of 3 models...
Complete after 0.960248 seconds.
Evaluating all 35960 ensemble combinations of 4 models...
9982: 20191227130920; 20191228212334; 20191230021151;
Complete after 0.883085 seconds.
Evaluating all 201376 ensemble combinations of 5 models...
Complete after 0.905336 seconds.
Evaluating all 906192 ensemble combinations of 6 models...
9982: 20191220225200; 20191222021508; 20191226094610; 20191227130920; 20191230061935;
9982: 20191221024640; 20191222021508; 20191226172316; 20191227051218; 20191228011057;
.
.
.
````

#### ensemble_data.txt File Format

The contents of the ensemble_data.txt file are formatted as follows:

1. The first line has four numbers on it:
   1. The first number is the total number of validation samples.
   2. The second number is the count of validation samples that all models predicted correctly.
   3. The third number is the count of validation samples that none of the models predicted correctly.
   4. The fourth number is the count of models that are in the file.
2. If you subtract the second number on the first line from the first number on the first line, you get the number of validation samples that at least one model predicted correctly and at least one model predicted incorrectly.  The second line in the file will and must have this many numbers.  These numbers are the positions of the images in the MNIST evaluation data that correspond to the validation samples that at least one model predicted correctly and at least one model predicted incorrectly.
3. The third line in the file has the same amount of numbers as the second line, but the numbers in this line are the correct labels for the validation samples that at least one model predicted correctly and at least one model predicted incorrectly.
4. Each subsequent line in the file (one for each model in the experiment), begins with the model's run_name (as was specified or defaulted to in the call to ``train.py``).  After that is a series of numbers, the count of which depends on how the file was generated:
    1. If the file was generated with ``--output_all_logits`` set to ``False`` or omitted, there will be the same amount of numbers (after the model's run_name) as there were on the second and third lines.  These numbers are the predictions made by the model on this line for those validation samples that at least one model predicted correctly and at least one model predicted incorrectly.
    2. If the file was generated with ``--output_all_logits`` set to ``True``, there will be 10x as many numbers (after the model's run_name) as there were on the second and third lines.  These numbers are the logits the model on this line produced for those validation samples that at least one model predicted correctly and at least one model predicted incorrectly.  There, are 10x as many numbers on the line in this case because there are 10 output classes in the MNIST data.
