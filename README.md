# UDFS
Unsupervised Discriminative Feature Selection
The Unsupervised Discriminative Feature Selection (UDFS) algorithm originally proposed by Yang et al. [Click here](https://www.ijcai.org/Proceedings/11/Papers/267.pdf) aims to select the most discriminative features for data representation. The algorithm optimizes the features and provides an output with feature ranking and weights. It uses inout training file with features and class values. Certain other parameters such as gamma, lambda, k and nclasses are required for computatio of optimal features. The algorithm is based on L-2,1 regularization approach for minimization of the objective function and generating feature coefficient for each value of lambda.

The UDFS application can be executed by the following command: 
```
usage: python2.7 UDFS.py -i input.csv -k 1 -g 0.00001 -l 0.00001 -n 2 -o output.csv

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input training file containing features and classes
  -k K, --k K           input k value ( default: k = 1 )
  -g GAMMA, --gamma GAMMA
                        input gamma value ( default: g = 0.00001 )
  -l LAMDA, --lamda LAMDA
                        input lamda value ( default: l = 0.00001 )
  -n NCLASSES, --nclasses NCLASSES
                        input number of classes in the training set
  -o OUTPUT, --output OUTPUT
                        Output filename containing feature ranking and weights
```
