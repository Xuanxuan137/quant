参数说明:
--graph             前一步提取的计算图所在位置。绝对路径或相对路径
--calib_set         量化所需数据集。将所有图片路径存在一个文本文件中，一行一个。
                    并将该文本文件路径传递给此参数。
--calib_size        量化图片的尺寸。应为4个由逗号分隔的整数(batch_size, C, H, W)，如"1,3,224,224"
                    其中batch_size是指量化时一次输入的图片的数量，不是总数量
--output_dir        量化后计算图保存的位置


bias=None时，将bias设为全为0
--calc_running_img_list     突然发现running mean和running var是能够直接从模型中提取出来的，所以不需要计算了