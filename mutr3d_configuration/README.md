# 目录
1. [简介](#简介)
2. [环境配置](#环境配置)
3. [数据集的处理](#数据集的处理)
4. [启动训练](#启动训练)

## 简介
本文档介绍了mutr3d项目在本地的部署方法，参考了网站
https://zhuanlan.zhihu.com/p/671956617 和
https://blog.csdn.net/newbie_dqt/article/details/136740751的内容

## 环境配置
下载MUTR3D项目<br> 
在配置环境前，先下载MUTR3D项目到本地<br> 
```
git clone https://github.com/a1600012888/MUTR3D.git
```


FUTR3D需要安装的包的版本
FUTR3D README关于依赖的版本描述如下：

1. mmcv==1.3.14
2. mmdetection==2.12.0
3. motmetrics==1.1.3
4. mmdetection3d==0.13.0
5. nuscenes-devkit
本文选用的各种依赖的版本为：

具体配置流程如下：
- 创建并进入环境
```
conda create -n mutr3d python=3.8 -y
conda activate futr3d
```
- 安装torch==1.13.0、cuda==11.6
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```
- 安装mmcv-full==1.3.14
```
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.3.14
```
- 安装mmdet==2.12.0（使用pip安装，作者使用mim安装遇到“循环”问题，实测使用pip安装可行）
```
pip install mmdet==2.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装motmetrics==1.1.3（使用pip安装，作者使用mim安装遇到“循环”问题，实测使用pip安装可行）
```
pip install motmetrics==1.1.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装nuscenes-devkit
```
pip install nuscenes-devkit -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装mmdet3d 0.13.0
在futr3d目录下下载mmdet的工程，but replace its mmdet3d/api/ from mmdetection3d by mmdet3d/api/ in this repo.
```
cd MUTR3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.13.0
# cp -r ../mmdet3d/api mmdet3d/
# cp ../mmdet3d/models/builder.py mmdet3d/models/
# cp ../mmdet3d/models/detectors/mvx_two_stage.py mmdet3d/models/detectors/mvx_two_stage.py

# replace the mmdetection3d/mmdet3d with the mmdet3d_full
cp -r ../mmdet3d_full ./mmdet3d

cp -r ../plugin ./ 
cp -r ../tools ./ 
# then install mmdetection3d following its instruction. 
# and mmdetection3d becomes your new working directories. 
pip install -v -e .
```

## 数据集的处理
- 数据集下载
使用nuscenes v1.0-mini数据集，下载并复制到本地文件夹下data/nuscenes文件夹下，解压缩
```
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
mkdir /home/baojiali/Downloads/public_code/futr3d/data/nuscenes
cp v1.0-mini.tgz /home/baojiali/Downloads/public_code/futr3d/data/nuscenes
cd /home/baojiali/Downloads/public_code/futr3d/data/nuscenes
tar -xvzf v1.0-mini.tgz
```

futr3d/data/nuscenes/
├── LICENSE
├── maps
├── samples
├── sweeps
├── v1.0-mini
└── v1.0-mini.tgz

- 修改路径问题
把mmdet3d和tools里的create_data.py和merge_infos.py
总共四个文件中data/nusc_new/改成data/nuscenes/

- 数据适配代码
运行如下代码得到mmdet3d能够处理的形式
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

futr3d/data/nuscenes/
├── LICENSE
├── maps
├── nuscenes_dbinfos_train.pkl
├── nuscenes_gt_database
├── nuscenes_infos_train_mono3d.coco.json
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val_mono3d.coco.json
├── nuscenes_infos_val.pkl
├── samples
├── sweeps
├── v1.0-mini
└── v1.0-mini.tgz

## 启动训练
```
python tools/train.py plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py --gpu-id 0 --autoscale-lr
```

如果按照github代码上的训练方法
```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py 8
```
注意，要把参数中运行程序的文件名改成lidar_0075v_900q.py
如果出现字符串问题，可能是dist_train.sh文件里的字符由于windows/unix间换行符的问题，需要转换一下
sudo apt install dos2unix
dos2unix tools/dist_train.sh

## 启动运行
```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py ../lidar_cam.pth 8 --eval bbox
```
注意，要把运行参数中的程序的文件名改成lidar_0075v_cam_res101.py