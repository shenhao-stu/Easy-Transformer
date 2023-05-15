# Easy-Transformer
This repository contains an implementation of the Transformer model with forward and backward propagation using the numpy and cupy libraries.

## Overview
The Transformer model is a popular architecture for natural language processing tasks such as machine translation and language modeling. This implementation demonstrates how to use numpy and cupy libraries to build a complete Transformer model with multiple layers, including positional encodings, multi-head attention, and point-wise feed-forward networks.

## Requirements
To run this implementation, you will mainly need:

- Python 3.x
- Numpy
- Cupy
- Wandb (optional)

## Usage
To use this implementation, clone the repository and run the train.py script with the desired hyperparameters. The script will train the Transformer model on the provided dataset and save the trained model to a file.

```bash
python train.py
```

## Credits
This implementation is based on the Transformer model described in the paper "Attention Is All You Need" by Ashish Vaswani et al. and the implementation in the HuggingFace and Pytorch official repository.

## License
This implementation is released under the MIT License. See LICENSE for details.

## To Do List

- [ ] Optimize Softmax Compution
- [ ] Implement [Llama](https://github.com/facebookresearch/llama)