1. 安装环境
2. cd health_cube_ai/model/MedicalSeg
3. python deploy/python/infer.py \
--config saved_model/deploy.yaml \
--image_path ../images \
--save_dir ../results
--image_path D:\PyProject\health_cube_ai\model\images\3a93b3d5-b8be-4b6f-bd00-922d6080d9fb.nii.gz
--save_dir 为推理结果保存路径，文件格式为.nii.npy


首先需要再model文件夹下创建一个images文件夹与results文件夹
其次将权重文件放入MedicalSeg/saved_model文件夹下
[文件下载地址](https://drive.google.com/drive/folders/1huuxj7CgfgV-vk1BAcqwVn2fgGlfyvky?usp=drive_link)

