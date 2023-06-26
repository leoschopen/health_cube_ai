1. 安装环境
2. cd health_cube_ai/model/MedicalSeg
3. python deploy/python/infer.py \
--config saved_model/deploy.yaml \
--image_path ../images \
--save_dir ../results
4. --image_path 为待推理图片路径，可以为文件夹、可以为文件
--save_dir 为推理结果保存路径，文件格式为.nii.npy