## Evaluating Ensemble Model Combinations
For each of our experiments, we generated 32 models and then needed to search the combination space of those models for ensembles.

Given that we wanted to evaluate every possible combination of including or excluding the 32 models, we have ~2^32 evaluations to perform.

After executing our python implementaiton for accomplishing this for ~24 hours, we extrapolated and determined that it would need 40 days to complete.

So we decided to implement in C++/CUDA so that we could take advantage of GPU accelerated execution.
This implementation executes in roughly ~1 hour (on an NVIDIA GeForce 940MX) for the same data that was on track to take ~40 days.
