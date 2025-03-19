# jiashuo
3D shape match notebook

#一.环境配置
#windows
cuda安装：
nvidia-smi 查看当前显卡支持的最高cuda版本，虚拟环境中安装的cuda版本≤这个版本即可
cudatoolkit 支持不同虚拟环境中安装不同版本的cuda

pytorch安装：
wheel下载速度一般比较快，注意与cuda和python版本对应，使用GPU版本。

pytorch3d：
推荐 vs2019
管理员身份进入 x64 Native Tools Command Prompt for VS 2019终端
cd pytorch3d文件夹，激活虚拟环境
输入以下指令
set DISTUTILS_USE_SDK=1
set PYTORCH3D_NO_NINJA=1
执行：
python setup.py install


