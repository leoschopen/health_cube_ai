from tools.preprocess_utils import verify_dataset_integrity
from tools.preprocess_utils._convert_to_decathlon import convert_to_decathlon
import numpy as np
import os

decathlon_dir = "/workspace/code/LQ/data/decathlon"
raw_data_dir = "/workspace/code/medical-seg/PaddleSeg/contrib/MedicalSeg/data/raw_data"
num_threads = 4

# AssertionError: There needs to be a dataset.json file in folder /workspace/code/LQ/data/decathlon, but not found.
# 要有dataset.json
# Traceback (most recent call last):
#   File "/workspace/code/medical-seg/PaddleSeg/contrib/MedicalSeg/lqpreprocess.py", line 17, in <module>
#     verify_dataset_integrity(
#   File "/workspace/code/medical-seg/PaddleSeg/contrib/MedicalSeg/tools/preprocess_utils/integrity_checks.py", line 202, in verify_dataset_integrity
#     assert os.path.isfile(

if not os.path.exists(decathlon_dir):
    print("{} not found, convert data to decathlon.".format(
        decathlon_dir))
    convert_to_decathlon(
        input_folder=raw_data_dir,
        output_folder=decathlon_dir,
        num_processes=num_threads)
    # verify_dataset_integrity(
        # decathlon_dir, default_num_threads=num_threads)
else:
    print(
        "Found existed {}, please ensure your dataset is preprocessed correctly!!!".
        format(decathlon_dir))