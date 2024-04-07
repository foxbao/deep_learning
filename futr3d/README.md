https://zhuanlan.zhihu.com/p/671956617
FUTR环境配置
下载FUTR3D项目
在配置环境前，先下载FUTR3D项目到本地
git clone https://github.com/Tsinghua-MARS-Lab/futr3d.git

FUTR3D需要安装的包的版本
FUTR3D README关于依赖的版本描述如下：

mmcv-full>=1.5.2, <=1.7.0
mmdet>=2.24.0, <=3.0.0
mmseg>=0.20.0, <=1.0.0
nuscenes-devkit
本文选用的各种依赖的版本为：

torch==1.13.0、cuda==11.6
mmcv-full==1.7.0
mmdet==2.27.0
mmseg==0.30.0
nuscenes-devkit==默认
具体配置流程如下：

创建并进入环境
conda create -n futr3d python=3.8 -y
conda activate futr3d
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
