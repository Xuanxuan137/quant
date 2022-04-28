#!/bin/bash
# mnist_conv3
./quant \
--model_dir ../files/mnist_conv3_output \
--calib_set ../files/mnist_calib_set.txt \
--method per_tensor \
--output_dir ../files/mnist_quanted_output \
--val_set ../files/mnist_val_set.txt

# mnist_bn
./quant \
--model_dir ../files/mnist_bn_output \
--calib_set ../files/mnist_calib_set.txt \
--method per_tensor \
--output_dir ../files/mnist_quanted_output \
--val_set ../files/mnist_val_set.txt

# mnist_conv_deep
./quant \
--model_dir ../files/mnist_conv_deep_output \
--calib_set ../files/mnist_calib_set.txt \
--method per_tensor \
--output_dir ../files/mnist_quanted_output \
--val_set ../files/mnist_val_set.txt

# mnist_conv_add
./quant \
--model_dir ../files/mnist_conv_add_output \
--calib_set ../files/mnist_calib_set.txt \
--method per_tensor \
--output_dir ../files/mnist_quanted_output \
--val_set ../files/mnist_val_set.txt

# cifar10
./quant \
--model_dir ../files/cifar_output \
--calib_set ../files/cifar_calib_set.txt \
--method per_tensor \
--output_dir ../files/cifar_quanted_output \
--val_set ../files/cifar_val_set.txt

# vgg11
./quant \
--model_dir ../files/vgg11_output \
--calib_set ../files/imgnet_calib_set.txt \
--method per_tensor \
--output_dir ../files/imgnet_quanted_output \
--val_set ../files/imgnet_val_set.txt

# resnet18
./quant \
--model_dir ../files/resnet18_output \
--calib_set ../files/imgnet_calib_set.txt \
--method per_tensor \
--output_dir ../files/imgnet_quanted_output \
--val_set ../files/imgnet_val_set.txt