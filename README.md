## Caption images with RNN

### Training
`python main.py training x`

When `x` is not specified, training starts from zero. When it is specified, program 
will load the according `encoder-x.pth` and `decoder-x.pth` files to continue from previous training.

### Evaluation
`python main.py evaluating x`

When `x` is not specified, evaluation will be done with `encoder-1.pth` and `decoder-1.pth`. When it is specified, program 
will load the according `encoder-x.pth` and `decoder-x.pth` files to evaluate the results.