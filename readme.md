参数说明:
--graph             前一步提取的计算图所在位置。绝对路径或相对路径
--calib_set         量化所需数据集。将所有图片路径存在一个文本文件中，一行一个。
                    并将该文本文件路径传递给此参数。
--calib_size        量化图片的尺寸。应为4个由逗号分隔的整数(batch_size, C, H, W)，如"1,3,224,224"
                    其中batch_size是指量化时一次输入的图片的数量，不是总数量
--calc_running      是否现场计算running mean和running var。(仅在网络中存在bn层时需要此参数)
                    该参数可选值为true或false
--calc_running_img_list     现场计算running mean和running var所需的图片列表。(仅--calc_running为true时需要此参数)
                            将要用到的所有图片的路径存在一个文本文件中，每行一个图片路径，
                            并将该文本文件路径传递给此参数。默认将所有图片用于计算running mean和running var
--running_size              running mean和running var的数据量。(仅--calc_running为false时需要此参数)
                            在提供running mean var时，需要给定数据量以便读取。例如某层卷积channel为16，
                            则其后的bn的running mean和running var应各为16个数，则此参数应为16
--running_mean_var_binary   running mean和running var二进制数值的路径(仅--calc_running为false时需要此参数)
                            (与--running_mean_var_txt二选一，若都出现，以此参数优先)
                            直接以二进制形式提供计算好的running mean和runnin var。
                            将running mean和running var以二进制形式按顺序存在一个文件中
                            按顺序：各层running_mean按需排放，然后各层running_var按需排放
--running_mean_var_txt      running mean和running var文本数值的路径(仅--calc_running为false时需要此参数)
                            (与--running_mean_var_binary二选一，若都出现，以--running_mean_var_binary优先)
                            以文本数字形式提供计算好的running mean和running var。
                            将running mean和running var以文本形式按顺序存在一个文件中，一行一个数



Graph从导出模型读取，本身就是静态图，无法修改

以后可以再新建一个Module类，作为动态图，允许用户调用接口。用户可以继承Module，然后重载构造函数和forward