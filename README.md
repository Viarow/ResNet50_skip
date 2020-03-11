# ResNet50_skip
Implemented layer skipping and channel skipping in ResNet_50_v1, specifically at the branch2b conv layer in every residual block.

The additional item in the loss is the sum of flops at each branch2b conv layer after a forward pass, at the layer the flops is computed by: (2*in_channels*kernel_size*kernel_size-1)*h*w*out_channels. In this project out_channels is number of channels that are not skipped.

[Architecture diagram of ResNet_50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)

The training dataset is cifar10

If you want to train on specific GPUs, please use 'CUDA_VISIBLE_DEVICES='.
