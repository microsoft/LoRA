#!/bin/bash

echo "downloading pretrained model checkpoints..."
mkdir pretrained_checkpoints
cd pretrained_checkpoints
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin
cd ..

echo "script complete!"
