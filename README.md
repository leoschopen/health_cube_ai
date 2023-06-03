# 🏥  HealthCubeAI

HealthCubeAI是一款基于WEB的3D医疗数据解析平台，提供医疗数据的导入、分割、可视化和数据分析功能，以及飞桨模型在本地或云端的部署和推理。该平台还提供了更多相关场景的附加功能，通过稳定的软件功能和优秀的人机交互，为非AI专业人员提供良好的用户体验。

## 主要功能

- 医疗数据导入：支持多种医疗数据格式的导入，包括DICOM、NRRD等。
- 数据分割：使用飞桨深度学习模型对医疗数据进行分割，提取目标区域，如肿瘤、血管等。
- 可视化：将医疗数据和分割结果以3D模型的形式进行可视化展示，支持多种视图、交互和操作。
- 数据分析：提供多种数据分析和统计功能，如xxx等。
- 模型部署：支持飞桨模型在本地或云端的部署和推理，可根据实际需求选择合适的部署方式。
- 附加功能：包括用户管理、数据备份、数据分享等多种附加功能，提高平台的灵活性和可用性。



## 技术栈

- 前端
  - Vue.js
  - WebGL
- 后端
  - Django
  - MySQL
  - Redis
- 深度学习框架
  - PaddlePaddle

## 系统架构

系统采用前后端分离的架构，前端使用Vue.js框架进行开发，后端使用Django框架进行开发。前端通过WebGL技术实现了3D模型的可视化，后端采用MySQL和Redis数据库存储数据，并使用PaddlePaddle深度学习框架进行数据分割和模型部署。

## 运行预览+环境部署
```
Python >= 3.8.0 (推荐3.9+版本)
nodejs >= 14.0 (推荐最新)
Mysql >= 5.7.0 (可选，默认数据库sqlite3，推荐8.0版本)
Redis(可选，最新版)
```
### 后端运行

1. 进入后端项目目录:cd backend

2. 安装依赖环境: pip install -r requirements.txt

3. 执行迁移命令: python manage.py makemigrations python manage.py migrate

4. 初始化数据: python3 manage.py init

5. 启动项目: python3 manage.py runserver 0.0.0.0:8000




### 前端运行

1. 进入前端项目目录 cd web
2. 安装依赖 npm install --registry=https://registry.npm.taobao.org
3. 启动服务 npm run dev



### 访问项目

​    访问地址：http://localhost:8080 (opens new window)(默认为此地址，如有修改请按照配置文件)
​    账号：superadmin 密码：admin123456






## 贡献者

- SMY、SWH（前端开发）
- WXY（后端开发）
- LQ（深度学习模型开发）




# 目录管理：  
| 目录      | 功能                   |
| --------- | ---------------------- |
| uploads   | 直接上传目录           |
| tmp/ct    | dcm文件副本目录        |
| tmp/image | dcm读取转换为png目录   |
| tmp/mask  | 预测结果肿瘤掩膜目录   |
| tmp/draw  | 勾画肿瘤后处理结果目录 |


