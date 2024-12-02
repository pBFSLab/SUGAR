# SUGAR: Spherical Ultrafast Graph Attention Framework for Cortical Surface Registration

**SUGAR Official Repository**

# Introduction

## Related datasets
[SALD dataset](http://fcon_1000.projects.nitrc.org/indi/retro/sald.html)

[ADRC dataset](https://sites.google.com/view/yeolab/software)

[HCP dataset](https://db.humanconnectome.org/data/projects/HCP_1200)

[MSC dataset](https://openneuro.org/datasets/ds000224/versions/1.0.4)

[CoRR-HNU dataset](http://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html)

[UKB dataset](https://www.ukbiobank.ac.uk/)

## Implemented based on the following PyTorch libraries

[PyTorch3D](https://github.com/facebookresearch/pytorch3d) A library for deep learning with 3D data

[PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) PyTorch Extension Library of Optimized Scatter Operations

[PyG](https://github.com/pyg-team/pytorch_geometric) Graph Neural Network Library for PyTorch

## State-of-the-art models for comparison

[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)

[Spherical Demons](https://sites.google.com/view/yeolab/software/sphericaldemonsrelease)

[MSM Pair](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MSM)

[MSM Strain](https://github.com/ecr05/MSM_HOCR)

[S3Reg](https://github.com/zhaofenqiang/SphericalUNetPackage)


# Usage
## Test on a sample dataset
Download an example data from [GoogleDrive](https://drive.google.com/drive/folders/11ZR-trRGzRhWjxxhPL63Rgw4IkZzmxOC?usp=share_link) or [Baidu NetDisk](https://pan.baidu.com/s/1OCLCpPa53_yACL-N3RNxCA?pwd=d6mz), the surface and morphometrics for registration are preprocessed using FreeSurfer recon-all.

## Test on new datatsets
If new datasets are welling to be used, please make sure you have all the pre-request files.


## Docker 
run docker container is highly recommended, as you only need to pull docker.


### Pull image

```shell
sudo docker pull ninganme/sugar:latest
```

### Registration
```shell
sudo docker run -it --rm --gpus all \
-v test_dataset_path:/data \
-v output_path:/out sugar \
--fsd /usr/local/freesurfer \
--sd /data \
--out /out \
--sid sub01 \
--hemi lh \
--device cuda
```

## Installation

download model [Google Drive](https://drive.google.com/drive/folders/1WmhJqQSxZnAIqwnV4-4fVfSCiyBxrhzy?usp=share_link) or [Baidu NetDisk](https://pan.baidu.com/s/1-3bg8-XDy7dQr1HnFRG-YQ?pwd=ji9t). 

**Note:** please have FreeSurfer properly installed, as the registration atlas is from FreeSurfer ***(FreeSurfer/subjects/fsaverage)***


### install libraries

```shell
# install pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 

# install pytorch3d
pip install fvcore==0.1.5.post20221221
pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html 

# install torch_geometric
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch_geometric==2.2.0

# install nibabel 
pip install nibabel==3.2.2
```

### run SUGAR

```shell
cd SUGAR

python3 predict.py \ 
--sd test_dataset_path \
--out output_path \
--fsd freesurfer_path \
--sid sub01 \
--model-path model_path \
--hemi lh \
--device cuda
```

# Training

[Training Code](https://github.com/pBFSLab/SUGAR/tree/main/train)
