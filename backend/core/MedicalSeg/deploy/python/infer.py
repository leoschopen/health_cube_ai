# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import codecs
import warnings
import argparse

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import functools
import numpy as np

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
import paddle

import medicalseg.transforms as T
from medicalseg.cvlibs import manager
from medicalseg.utils import get_sys_env, logger, get_image_list
from medicalseg.utils.visualize import get_pseudo_color_map
from medicalseg.core.infer import sliding_window_inference
from tools import HUnorm, resample
from tools import Prep



parser = argparse.ArgumentParser(description='Test')
parser.add_argument(
    "--config",
    dest="cfg",
    help="The config file.",
    default=None,
    type=str,
    required=True)
parser.add_argument(
    '--image_path',
    dest='image_path',
    help='The directory or path or file list of the images to be predicted.',
    type=str,
    default=None,
    required=True)
parser.add_argument(
    '--batch_size',
    dest='batch_size',
    help='Mini batch size of one gpu or cpu.',
    type=int,
    default=1)
parser.add_argument(
    '--save_dir',
    dest='save_dir',
    help='The directory for saving the predict result.',
    type=str,
    default='./output')
parser.add_argument(
    '--device',
    choices=['cpu', 'gpu'],
    default="gpu",
    help="Select which device to inference, defaults to gpu.")

parser.add_argument(
    '--use_trt',
    default=False,
    type=eval,
    choices=[True, False],
    help='Whether to use Nvidia TensorRT to accelerate prediction.')
parser.add_argument(
    "--precision",
    default="fp32",
    type=str,
    choices=["fp32", "fp16", "int8"],
    help='The tensorrt precision.')
parser.add_argument(
    '--enable_auto_tune',
    default=False,
    type=eval,
    choices=[True, False],
    help='Whether to enable tuned dynamic shape. We uses some images to collect '
    'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
)
parser.add_argument(
    '--auto_tuned_shape_file',
    type=str,
    default="auto_tune_tmp.pbtxt",
    help='The temp file to save tuned dynamic shape.')

parser.add_argument(
    '--cpu_threads',
    default=10,
    type=int,
    help='Number of threads to predict when using cpu.')
parser.add_argument(
    '--enable_mkldnn',
    default=False,
    type=eval,
    choices=[True, False],
    help='Enable to use mkldnn to speed up when using cpu.')

parser.add_argument(
    "--benchmark",
    type=eval,
    default=False,
    help="Whether to log some information about environment, model, configuration and performance."
)
parser.add_argument(
    "--model_name",
    default="",
    type=str,
    help='When `--benchmark` is True, the specified model name is displayed.'
)

parser.add_argument(
    '--with_argmax',
    dest='with_argmax',
    help='Perform argmax operation on the predict result.',
    action='store_true')
parser.add_argument(
    '--print_detail',
    default=True,
    type=eval,
    choices=[True, False],
    help='Print GLOG information of Paddle Inference.')

parser.add_argument(
    '--use_swl',
    default=False,
    type=eval,
    help='use sliding_window_inference')

parser.add_argument('--use_warmup', default=True, type=eval, help='warmup')

parser.add_argument(
    '--img_shape',
    default=[128],
    nargs='+',
    help='"A single value or three values to specify the size in each dimension."'
)

parser.add_argument('--is_nhwd', default=True, type=eval, help='is_nhwd')



def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
           and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
           and args.device == "gpu" and args.use_trt and args.enable_auto_tune


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
            'transforms'])
        if self.dic['Deploy']['inference_helper'] is not None:
            self._inference_helper = self.load_inference_helper(self.dic[
                'Deploy']['inference_helper'])
        else:
            self._inference_helper = None
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def inference_helper(self):
        return self._inference_helper

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type', None)
            if ctype is not None:
                transforms.append(com[ctype](**t))

        return T.Compose(transforms)

    @staticmethod
    def load_inference_helper(t):
        com = manager.INFERENCE_HELPERS
        inference_helper = None
        ctype = t.pop('type', None)
        if ctype is not None:
            inference_helper = com[ctype](**t)

        return inference_helper


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.
    Args:
        args(dict): input args.
        imgs(str, list[str]): the path for images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args)

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        data = np.array([cfg.transforms(imgs[i])[0]])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except:
            logger.info(
                "Auto tune fail. Usually, the error is out of GPU memory, "
                "because the model and image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


class ModelLikeInfer:
    def __init__(self, input_handle, output_handle, predictor):
        self.input_handle = input_handle
        self.output_handle = output_handle
        self.predictor = predictor

    def infer_likemodel(self, input_handle, output_handle, predictor, data):
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data.numpy())
        predictor.run()
        return paddle.to_tensor(output_handle.copy_to_cpu())

    def infer_model(self, data):
        return (self.infer_likemodel(self.input_handle, self.output_handle,
                                     self.predictor, data), )


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        self.predictor = create_predictor(self.pred_cfg)

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        "初始化基础配置"
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=300,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                    os.path.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i in range(0, len(imgs_path), args.batch_size):

            if args.use_warmup:
                # warm up
                if i == 0 and args.benchmark:
                    for j in range(5):
                        if self.cfg.inference_helper is not None:
                            data = self.cfg.inference_helper.preprocess(
                                self.cfg, imgs_path, args.batch_size, 0)
                        else:
                            data = np.array([
                                self._preprocess(img)  # load from original
                                for img in imgs_path[0:args.batch_size]
                            ])
                        input_handle.reshape(data.shape)
                        input_handle.copy_from_cpu(data)
                        self.predictor.run()
                        results = output_handle.copy_to_cpu()
                        results = self._postprocess(results)

            # inference
            if args.benchmark:
                self.autolog.times.start()
            if self.cfg.inference_helper is not None:
                data = self.cfg.inference_helper.preprocess(self.cfg, imgs_path,
                                                            args.batch_size, i)
            else:
                data = np.array([
                    self._preprocess(p)
                    for p in imgs_path[i:i + args.batch_size]
                ])

            if args.benchmark:
                self.autolog.times.stamp()

            if args.use_swl:

                infer_like_model = ModelLikeInfer(input_handle, output_handle,
                                                  self.predictor)
                data = paddle.to_tensor(data)
                if args.is_nhwd:
                    data = paddle.squeeze(data, axis=1)

                if len(args.img_shape) == 1:
                    results = sliding_window_inference(
                        data, (int(args.img_shape[0]), int(args.img_shape[0]),
                               int(args.img_shape[0])), 1,
                        infer_like_model.infer_model)
                else:
                    results = sliding_window_inference(
                        data, (int(args.img_shape[0]), int(args.img_shape[1]),
                               int(args.img_shape[2])), 1,
                        infer_like_model.infer_model, "NCDHW")

                results = results[0]

            else:
                input_handle.reshape(data.shape)
                input_handle.copy_from_cpu(data)

                self.predictor.run()

                results = output_handle.copy_to_cpu()

            if args.benchmark:
                self.autolog.times.stamp()
            if self.cfg.inference_helper is not None:
                results = self.cfg.inference_helper.postprocess(results)
            else:
                results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)
            self._save_npy(results, imgs_path[i:i + args.batch_size])
        logger.info("Finish")

    def _preprocess(self, img):
        """load img and transform it
        Args:
        Img(str): A batch of image path
        """
        if not "npy" in img:
            image_files = get_image_list(img, None, None)
            warnings.warn(
                "The image path is {}, please make sure this is the images you want to infer".
                format(image_files))
            savepath = os.path.dirname(img)
            pre = [
                HUnorm,
                functools.partial(
                    resample,  # TODO: config preprocess in deply.yaml(export) to set params
                    new_shape=[128, 128, 128],
                    order=1)
            ]

            for f in image_files:
                f_nps = Prep.load_medical_data(f)
                for f_np in f_nps:
                    if pre is not None:
                        for op in pre:
                            f_np = op(f_np)

                    # Set image to a uniform format before save.
                    if isinstance(f_np, tuple):
                        f_np = f_np[0]
                    f_np = f_np.astype("float32")

                    np.save(
                        os.path.join(
                            savepath,
                            f.split("/")[-1].split(
                                ".", maxsplit=1)[0]),
                        f_np)

            img = img.split(".", maxsplit=1)[0] + ".npy"
        return self.cfg.transforms(img)[0]

    def _postprocess(self, results):
        "results is numpy array, optionally postprocess with argmax"
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def _save_npy(self, results, imgs_path):
        for i in range(results.shape[0]):
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.npy'
            np.save(os.path.join(self.args.save_dir, basename), results)

import shutil

# 输入一个nii.gz，推理后将结果保存在predict文件夹
def inference(nii_path, config_path = "./core/MedicalSeg/tools/preprocess_globals.yml", save_dir = "./tmp/results"):
    args = argparse.Namespace()
    args.config = config_path
    args.image_path = nii_path
    args.batch_size = 1
    args.save_dir = save_dir
    args.device = 'gpu'
    args.use_trt = False
    args.precision = 'fp32'
    args.enable_auto_tune = False
    args.auto_tuned_shape_file = 'auto_tune_tmp.pbtxt'
    args.cpu_threads = 10
    args.enable_mkldnn = False
    args.benchmark = False
    args.model_name = ''
    args.with_argmax = False
    args.print_detail = True
    args.use_swl = False
    args.use_warmup = True
    args.img_shape = [128]
    args.is_nhwd = True

    src_nii_name = nii_path.split('\\')[-1]

    # 模拟处理后并保存
    shutil.copy(nii_path, save_dir)
    predict_nii_path = os.path.join(save_dir, src_nii_name)

    # imgs_list = get_image_list(img_path)
    # if use_auto_tune(args):
    #     tune_img_nums = 10
    #     auto_tune(args, imgs_list, tune_img_nums)
        
    # predictor = Predictor(args)
    # predictor.run(imgs_list)

    # if use_auto_tune(args) and \
    #         os.path.exists(args.auto_tuned_shape_file):
    #     os.remove(args.auto_tuned_shape_file)

    # # test the speed.
    # if args.benchmark:
    #     predictor.autolog.report()

    return predict_nii_path

def get_slices(nib_nii):
    data = nib_nii.get_fdata()
    slices = []
    for i in range(data.shape[2]):
        slice_data = np.rot90(data[:, :, i])
        slices.append(slice_data.tolist())
    return slices


import os
import nibabel as nib
import numpy as np
import cv2
import SimpleITK as sitk

def read_nii_info(filepath):
    sitkImage = sitk.ReadImage(filepath)
    info = {
        "origin": sitkImage.GetOrigin(),
        "size": sitkImage.GetSize(),
        "spacing": sitkImage.GetSpacing(),
        "direction": sitkImage.GetDirection(),
        "dimension": sitkImage.GetDimension(),
        "width": sitkImage.GetWidth(),
        "height": sitkImage.GetHeight(),
        "depth": sitkImage.GetDepth(),
        "pixel_type": sitkImage.GetPixelIDTypeAsString()
    }
    return info

import numpy as np
import json

def calculate_properties(segmentation, spacing):
    # 计算目标数量
    num_targets = len(np.unique(segmentation)) - 1
    if num_targets < 1:
        raise ValueError("分割结果中没有目标")
    
    # 计算每个目标的属性
    targets = []
    for i in range(1, num_targets + 1):
        # 计算目标表面积
        surface_area = calculate_surface_area(segmentation == i, spacing)
        
        # 计算目标体积
        volume = calculate_volume(segmentation == i, spacing)
        
        # 计算目标在 x、y、z 切面上的直径
        diameters = calculate_diameters(segmentation == i, spacing)
        
        # 将目标属性保存为字典
        target = {
            "label": f"target {i}",
            "surface_area": round(surface_area, 2),
            "volume": round(volume, 2),
            "diameters": {
                "x": round(diameters[0], 2),
                "y": round(diameters[1], 2),
                "z": round(diameters[2], 2)
            }
        }
        targets.append(target)
    
    return targets

def calculate_surface_area(segmentation, spacing):
    # 计算表面积
    dx, dy, dz = spacing
    x_gradient = np.gradient(np.logical_xor(segmentation, False).astype(int), dx, axis=0)
    y_gradient = np.gradient(np.logical_xor(segmentation, False).astype(int), dy, axis=1)
    z_gradient = np.gradient(np.logical_xor(segmentation, False).astype(int), dz, axis=2)
    surface_area = np.sum(np.sqrt(x_gradient**2 + y_gradient**2 + z_gradient**2))
    return surface_area * dx * dy * dz

def calculate_volume(segmentation, spacing):
    # 计算体积
    volume = np.sum(segmentation)
    dx, dy, dz = spacing
    return volume * dx * dy * dz

def calculate_diameters(segmentation, spacing):
    # 计算直径
    dx, dy, dz = spacing
    x_diameter = np.sqrt(np.max(np.sum(segmentation, axis=(1,2))) * dx)
    y_diameter = np.sqrt(np.max(np.sum(segmentation, axis=(0,2))) * dy)
    z_diameter = np.sqrt(np.max(np.sum(segmentation, axis=(0,1))) * dz)
    return x_diameter, y_diameter, z_diameter

import SimpleITK as sitk
import numpy as np
import pandas as pd

def get_seg_info_one_slice(output_nii,index):
    output_img = sitk.ReadImage(output_nii)
    output_array = sitk.GetArrayFromImage(output_img)
    pixel_spacing = output_img.GetSpacing()

    # 计算不同类别数据的数量并保存到Pandas DataFrame,类别数需要根据实际情况修改
    num_classes = len(np.unique(output_array[index]))
    label = [f"label {i}" for i in range(1, num_classes)]
    count = [np.sum(output_array[index] == i) for i in range(1, num_classes)]
    pixel_size = 1.0 # 像素尺寸为1mm x 1mm x 1mm
    area = [round(np.sum(output_array[index] == i) * pixel_spacing[0] * pixel_spacing[1] * pixel_size, 2) for i in range(1, num_classes)]

    result = pd.DataFrame({
        "label": label,
        "count": count,
        "area": area,
    })
    json_data = result.to_json(orient='records', force_ascii=False)
    json_str = json.loads(json_data)
    
    return json_str

def get_seg_info_all_slice(output_nii):
    output_img = sitk.ReadImage(output_nii)
    output_array = sitk.GetArrayFromImage(output_img)

    pixel_spacing = output_img.GetSpacing()
    seg_info  = calculate_properties(output_array, pixel_spacing)
    result = pd.DataFrame({
        'label' : [i['label'] for i in seg_info],
        'surface_area' : [i['surface_area'] for i in seg_info],
        'volume' : [i['volume'] for i in seg_info],
        'diameters_x' : [i['diameters']['x'] for i in seg_info],
        'diameters_y' : [i['diameters']['y'] for i in seg_info],
        'diameters_z' : [i['diameters']['z'] for i in seg_info],
    })
    json_data = result.to_json(orient='records', force_ascii=False)
    json_str = json.loads(json_data)
    
    return json_str

def predict(src_nii_path):
    # print("=====src_nii_path",src_nii_path)
    src_nii_info = read_nii_info(src_nii_path)
    predict_nii_path = inference(src_nii_path)

    src_nii = nib.load(src_nii_path)
    predict_nii = nib.load(predict_nii_path)

    data = src_nii.get_fdata()
    src_slices = get_slices(src_nii)
    predict_slices = get_slices(predict_nii)

    src_img = np.array(src_slices[0])
    predict_img = np.array(predict_slices[0])

    # 保存为图片进行展示
    src_path = './tmp/src_display/' + src_nii_path.split('\\')[-1].split('.')[0] + '.png'
    predict_path = './tmp/predict_display/' + predict_nii_path.split('\\')[-1].split('.')[0] + '.png'
    cv2.imwrite(src_path, src_img)
    cv2.imwrite(predict_path, predict_img)

    output_img = sitk.ReadImage(predict_nii_path)
    output_array = sitk.GetArrayFromImage(output_img)

    pixel_spacing = output_img.GetSpacing()
    seg_info  = calculate_properties(output_array, pixel_spacing)

    # 保存为json
    src_nii_path = src_nii_path.split('\\')[-1].split('.')[0] + '_src.json'
    predict_nii_path = predict_nii_path.split('\\')[-1].split('.')[0] + '_predict.json'

    with open('./tmp/json/' + src_nii_path, 'w') as f:
        json.dump(src_nii_info, f)
    with open('./tmp/json/' + predict_nii_path, 'w') as f:
        json.dump(seg_info, f)


    return src_path, predict_path, src_slices, predict_slices, src_nii_path, predict_nii_path




def main(args):
    imgs_list = get_image_list(
        args.image_path)  # get image list from image path

    # support autotune to collect dynamic shape, works only with trt on.
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # infer with paddle inference.
    predictor = Predictor(args)
    predictor.run(imgs_list)

    if use_auto_tune(args) and \
            os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    # test the speed.
    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
