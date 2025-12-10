# OWl-GPS

## Setup
### Dependencies
1. `conda create -n <environment-name> python==3.9`
2. `conda activate <environment-name>`
3. Install torch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
4. `pip install -U openmim`
5. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html`

6. Follow the required packages as mentioned in requirements.txt 

#To train our proposed framework, execute:

python OWL-GPS.py 

