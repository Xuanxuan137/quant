# mnist
./quant --graph ../graph.txt --calib_set ../mnist_calib_set.txt --calib_size 1,1,28,28 --output_dir ../mnist_quanted_output

# cifar10
./quant --graph ../graph.txt --calib_set ../cifar_calib_set.txt --calib_size 1,3,32,32 --output_dir ../cifar_quanted_output

# resnet18
./quant --graph ../graph.txt --calib_set ../imgnet_calib_set.txt --calib_size 1,3,224,224 --output_dir ../imgnet_quanted_output