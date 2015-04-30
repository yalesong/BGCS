# BGCS

This software package provides tools to perform multi-label classification/regression (i.e., given an input signal, predict multi-dimensional output labels). It includes MATLAB implementations of Bayesian Compressed Sensing (BCS) [1] and Bayesian Group-sparse Compressed Sensing (BGCS) [2], which extends BCS by considering group structures in the output label space. 

[1] Ashish Kapoor, Raajay Viswanathan, and Prateek Jain. "Multilabel classification using bayesian compressed sensing." NIPS 2012.

[2] Yale Song, Daniel McDuff, Deepak Vasisht, and Ashish Kapoor. "Exploiting sparsity and co-occurrence structure for action unit recognition." IEEE FG 2015.

Copyright (c) 2015 Yale Song (yalesong@csail.mit.edu). <br>
Permissions are granted under the MIT License (MIT).


# Contents
To start experimenting with the package, please see <code>./run_demo.m</code>
```
./license.txt : copy of the MIT License
./run_demo.m  : MATLAB demo script 
./lib
./lib/run_bcs.m   : BCS model script
./lib/run_bgcs.m  : BGCS model script
./lib/train_bcs.m : BCS/BGCS model training
./lib/test_bcs.m  : BCS model test
./lib/test_bgcs.m : BGCS model test
./helper
./helper/get_best_results.m : obtain best results based on performance on validation set
./helper/split_data.m       : leave-one-subject-out data split
./helper/eval_bc.m          : evaluation code for binary classification 

```

# Reference
Please cite the following paper if you end up using the code:

```
@inproceedings{song2015exploiting,
  title={Exploiting Sparsity and Co-occurrence Structure for Action Unit Recognition},
  author={Song, Yale and McDuff, Daniel and Vasisht, Deepak and Kapoor, Ashish},
  booktitle={Automatic Face \& Gesture Recognition and Workshops (FG 2015), 2015 IEEE International Conference on},
  year={2015},
  organization={IEEE}
}
```
