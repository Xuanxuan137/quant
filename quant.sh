#!/bin/bash
# mnist_conv3
./quant \
--model_dir ../mnist_conv3_output \
--calib_set ../mnist_calib_set.txt \
--method per_tensor \
--output_dir ../mnist_quanted_output \
--val_set ../mnist_val_set.txt

# mnist_bn
./quant \
--model_dir ../mnist_bn_output \
--calib_set ../mnist_calib_set.txt \
--method per_tensor \
--output_dir ../mnist_quanted_output \
--val_set ../mnist_val_set.txt

# mnist_conv_deep
./quant \
--model_dir ../mnist_conv_deep_output \
--calib_set ../mnist_calib_set.txt \
--method per_tensor \
--output_dir ../mnist_quanted_output \
--val_set ../mnist_val_set.txt

# cifar10
./quant \
--model_dir ../cifar_output \
--calib_set ../cifar_calib_set.txt \
--method per_tensor \
--output_dir ../cifar_quanted_output \
--val_set ../cifar_val_set.txt

# resnet18
./quant \
--model_dir ../resnet18_output \
--calib_set ../imgnet_calib_set.txt \
--method per_tensor \
--output_dir ../imgnet_quanted_output \
--val_set ../imgnet_val_set.txt