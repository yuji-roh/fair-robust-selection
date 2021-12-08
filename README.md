# Sample Selection for Fair and Robust Training

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS), 2021
----------------------------------------------------------------------

This directory is for simulating fair and robust sample selection on the 
synthetic dataset. The program needs PyTorch, Jupyter Notebook, and CUDA.

The directory contains a total of 5 files and 1 child directory: 
1 README, 2 python files, 2 jupyter notebooks, 
and the child directory containing 11 numpy files for synthetic data.
The synthetic data contains training set, validation set, and test set.
Note that the validation set is for another method in the paper (i.e., FR-Train), 
so it is not used in this simulation.

----------------------------------------------------------------------
#### To simulate the algorithm, please use the jupyter notebook in the directory.
----------------------------------------------------------------------

The jupyter notebook will load the data and train the models with two 
different fairness metrics: equalized odds and demographic parity.

Each training utilizes the FairRobust sampler, which defines in FairRobustSampler.py.
The pytorch dataloader serves the batches to the model via the FairRobust sampler. 
Experiments are repeated 5 times each.
After the training, the test accuracy and fairness will be shown.

The two python files are models.py and FairRobustSampler.py.
The models.py contains a logistic regression architecture and a test function.
The FairRobustSampler.py contains two classes: CustomDataset and FairRobust. 
CustomDataset class defines the dataset, and FairRobust class implements 
the proposed algorithm for fair and robust training as described in the paper.

The detailed explanations about each component have been written 
in the codes as comments.
Thanks!
