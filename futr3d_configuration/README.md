# 目录
1. [简介](#简介)
2. [环境配置](#环境配置)
3. [数据集的处理](#数据集的处理)
4. [启动训练](#启动训练)

## 简介
本文档介绍了futr3d项目在本地的部署方法，参考了网站
https://zhuanlan.zhihu.com/p/671956617 和
https://blog.csdn.net/newbie_dqt/article/details/136740751的内容

## 环境配置
FUTR环境配置<br>
下载FUTR3D项目<br> 
在配置环境前，先下载FUTR3D项目到本地<br> 
```
git clone https://github.com/Tsinghua-MARS-Lab/futr3d.git
```


FUTR3D需要安装的包的版本
FUTR3D README关于依赖的版本描述如下：

1. mmcv-full>=1.5.2, <=1.7.0
2. mmdet>=2.24.0, <=3.0.0
3. mmseg>=0.20.0, <=1.0.0
4. nuscenes-devkit
本文选用的各种依赖的版本为：

torch==1.13.0、cuda==11.6
mmcv-full==1.7.0
mmdet==2.27.0
mmseg==0.30.0
nuscenes-devkit==默认

具体配置流程如下：
- 创建并进入环境
```
conda create -n futr3d python=3.8 -y
conda activate futr3d
```
- 安装torch==1.13.0、cuda==11.6
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
- 安装mmcv-full==1.7.0
```
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.0
```
- 安装mmdet==2.27.0（使用pip安装，作者使用mim安装遇到“循环”问题，实测使用pip安装可行）
```
pip install mmdet==2.27.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装mmseg==0.30.0（使用pip安装，作者使用mim安装遇到“循环”问题，实测使用pip安装可行）
```
pip install mmsegmentation==0.30.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装nuscenes-devkit
```
pip install nuscenes-devkit -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 安装mmdet3d
```
cd futr3d/
pip install -v -e . 
```
- 降级numpy（mmdet3d限制了numba版本，发生报错"SystemError: initialization of _internal failed without raising an exception"，原因是安装torch那一步安装的numpy版本过高，产生了冲突）
```
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 降级yapf（不降级会在训练环节报错"FormatCode() got an unexpected keyword argument 'verify'"）
```
pip install yapf==0.40.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
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
- 修改路径问题
把mmdet3d和tools里的
create_data.py
merge_infos.py
总共四个文件中data/nusc_new/改成data/nuscenes/

- 数据适配代码
运行如下代码得到mmdet3d能够处理的形式
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```
## 启动训练
```
python tools/train.py plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py --gpu-id 0 --autoscale-lr
```

如果按照github代码上的训练方法
```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py 8
```
注意，要把github的文件名改成lidar_0075v_900q.py
如果出现字符串问题，可能是dist_train.sh文件里的字符由于windows/unix间换行符的问题，需要转换一下
sudo apt install dos2unix
dos2unix tools/dist_train.sh

## 启动运行
```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py ../lidar_cam.pth 8 --eval bbox
```
注意，要把github的文件名改成lidar_0075v_cam_res101.py