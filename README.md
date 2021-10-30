```shell
conda create -n mmhid python=3.8 -y
conda activate mmhid
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=10.2 -c pytorch -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html  
pip install mmdet
pip install -e .[full]
```