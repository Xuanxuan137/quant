参数说明:</br>
--model_dir ../mnist_conv3_output               模型文件路径</br>
--calib_set ../mnist_calib_set.txt              calib_set路径</br>
--method per_tensor                             量化方案per_tensor或者per_channel(暂不支持)</br>
--output_dir ../mnist_quanted_output            输出路径</br>
--val_set ../mnist_val_set.txt                  测试文件路径(不是必须)</br>
<!-- --activation_dtype int8                         activation量化数据类型</br>
--activation_symmetry asymmetric                activation对称性</br>
--weight_dtype int8                             weight量化数据类型</br>
--weight_symmetry symmetric                     weight对称性</br> -->


<!-- bias=None时，将bias设为全为0</br> -->
<!-- --calc_running_img_list     突然发现running mean和running var是能够直接从模型中提取出来的，所以不需要计算了</br> -->
graph.txt中的权重路径均使用相对于graph.txt的相对路径