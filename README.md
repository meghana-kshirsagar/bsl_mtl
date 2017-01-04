# bsl_mtl
Code from JCB 2016 paper


Compiling:
----------
To compile the code, you will need to install Shogun toolbox (http://shogun-toolbox.org/) which is used for matrix datastructures.
Upon installing shogun, you need to know where the libraries are. Here is how you compile my code (please open parcompile.sh to change the path pointing to the libraries):

sh parcompile.sh

Usage:
------
To see usage of the code, do:
./bsl_mtl

Usage: ./bsl_mtl \<config-file\> \<trainMatrix\> \<lambda\> \<num_factors_K\> \<model_prefix\> \<costFile\> \<cost\> \<l1/l2\> \<testFile\> \<iter-count\>

The last parameter (iter-count) is optional and is to be ignored.

Example run:
-------------
./bsl_mtl params.cfg train.txt 0.01 5 modelname train_costs.txt 100 l1 test.txt 1>out 2>err

To see the optimization output and labels on test, open the files "out" and "err". I am currently only printing average least-squared loss on test data. You can get RMSE by taking a square root of this value. 


Number of tasks in our setting: 3

You can change the following parameters (some are on the command prompt and the rest are in the config-file: params.cfg).
Open and check the file params.cfg for what data is used and other parameters.

trainMatrix: file containing training graph with edges from all tasks (format of each line of file: \<node1\> \<node2\> \<edge-weight\>)

taskIndices = "0 9305 103370"; (parameter from params.cfg, specifies the start line-number in training file for each task's training data)

K=5 (num_factors_K: low-rank factorization dimension)

lambda=0.01 (regularization to control frobenius norm of U and V)

sigmas = "0.0005 0.000001 0.000001"; (parameter in params.cfg, specifies task-specific regularization on matrix S_t)

betas = "0.5 0.3 0.5"; (controls how much importance to give to task-specific part)

cost=5 (This causes the loss on the training data to be multiplied by 5. I use this in cases where the loss value is too small which sometimes results in numerical issues)


Output files:
-------------
Model files: 

\<model-prefix\>.U.mod, 
\<model-prefix\>.V.mod, 
\<model-prefix\>.S_taskT.mod store these matrices for every few iterations

Matrix dimensions: 
U (dimension dxk)
V (dimension dxk)
S (dimension dxd)
