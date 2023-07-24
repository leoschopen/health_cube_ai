import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
import segmentation_models_pytorch as smp

import torch
from flask import *

import core.main
import core.net.unet as net
from core.MedicalSeg.deploy.python.infer import *
import SimpleITK as sitk

UPLOAD_FOLDER = r'.\tmp\uploads'

ALLOWED_EXTENSIONS = set(['dcm','jpg','png','nii','gz'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        file_type = file.filename.rsplit('.', 1)[1]
        if file_type == 'png' or file_type == 'jpg':
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            tar_path = os.path.join('./tmp/ct', file.filename)
            file.save(src_path)
            shutil.copy(src_path, './tmp/images')
            image_path = './tmp/images/' + file.filename
            # pid, image_info = core.main.c_main(image_path, current_app.model)
            pid, image_info = core.main.png_main(image_path)
            return jsonify({'status': 1,
                            'image_url': 'http://127.0.0.1:5003/' + pid,
                            'draw_url': 'http://127.0.0.1:5003/' + tar_path,
                        'image_info': image_info
                        })
        elif file_type == 'gz' or file_type == 'nii':
            # 保存上传图像至uploads与nii文件夹
            src_path = os.path.join('./tmp/uploads', file.filename)
            file.save(src_path)
            shutil.copy(src_path, './tmp/nii')

            src_path, predict_path, src_slices, predict_slices = predict(src_path)
            image_info = ""
            # 展示原图与推理后的图
            return jsonify({'status': 1,
                            'image_url': 'http://127.0.0.1:5003/' + src_path,
                            'draw_url': 'http://127.0.0.1:5003/' + predict_path,
                        'image_info': image_info
                        })


    return jsonify({'status': 0})


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)

import nibabel as nib
import numpy as np

# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    print("=======show",file)
    if request.method == 'GET':
        if file is None:
            pass
        else:
            file_type = file.rsplit('.', 1)[1]
            if file_type == 'png' or file_type == 'jpg':
                image_data = open(f'./tmp/{file}', "rb").read()
                response = make_response(image_data)
                response.headers['Content-Type'] = 'image/png'
                return response
            elif file_type == 'gz' or file_type == 'nii':
                img = nib.load(f'./tmp/{file}')
                data = img.get_fdata()
                slices = []
                for i in range(data.shape[2]):
                    slice_data = np.rot90(data[:, :, i])
                    slices.append(slice_data.tolist())
                return jsonify(slices)
    else:
        pass


# predict and get result
@app.route('/inference', methods=['POST'])
def do_inference():
    config_path = './core/MedicalSeg/saved_model/deploy.yaml'
    img_path = request.json['img_path'] # image_dir 文件夹
    save_dir = request.json['save_dir'] # save_dir 文件夹
    inference(config_path, img_path, save_dir)
    return jsonify({'message': 'Inference completed successfully.'})


if __name__ == '__main__':
    # with app.app_context():
    #     current_app.model = init_model()
    app.debug = True
    app.run(host='127.0.0.1', port=5003, debug=True)
