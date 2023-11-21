## Caption images with RNN

### Usage
`usage: main.py [-h] -m MODE [-l LOAD]`

#### Training
`python main.py -m training [-l x]`

When `x` is not specified, training starts from zero. When it is specified, program 
will load the according `encoder-x.pth` and `decoder-x.pth` files to continue on with previous training.

#### Evaluation
`python main.py -m evaluating [-l x]`

When `x` is not specified, evaluation will be done with `encoder-1.pth` and `decoder-1.pth`. When it is specified, program will load the according `encoder-x.pth` and `decoder-x.pth` files to evaluate the results.