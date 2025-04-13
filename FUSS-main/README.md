## [FUSS: A Universal and Flexible Framework for Unsupervised Statistical Shape Model Learning](https://nafieamrani.github.io/assets/pdf/nafie2024miccai.pdf)
![img](figures/miccai24_pipeline.png)

Official repository for the MICCAI 2024 paper: A Universal and Flexible Framework for Unsupervised Statistical Shape Model Learning by Nafie El Amrani, Dongliang Cao and Florian Bernard (University of Bonn).

## 🧑‍💻️‍ Installation
```bash 
conda create -n fuss python=3.9 # create new viertual environment
conda activate fuss
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia # install pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d # install pytorch3d
conda install pyg -c pyg # install torch_geometric
pip install -r requirements.txt # install other necessary libraries via pip
```

## 📝 Dataset
The pancreas dataset from Med Decathalon Dataset used in this paper was processed using mesh grooming tools. Please download the datasets from the this [link](https://www.shapeworks-cloud.org/#/) and put all datasets under ../data/. 
Other datasets can be processed in a similar fashion given CT or MRI images and their corresponding segmentations. 
```Shell
├── data
    ├── pancreas
        ├── off 
```
We thank the original dataset providers for their contributions, and that all credits should go to the original authors.

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for pancreas.
```python
python preprocess.py --data_root ../data/pancreas/ --n_eig 200

python preprocess.py --data_root data/vertebrae/ --n_eig 200
```

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/pancreas.yaml 

python train.py --opt options/train/vertebrae.yaml 
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/pancreas.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## 🙏 Acknowledgement
The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).
The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching).
This repository is adapted from [Spectral Meets Spatial: Harmonising 3D Shape Matching and Interpolation](https://github.com/dongliangcao/Spectral-Meets-Spatial).

### Med Decathalon Dataset: Pancreas 

From the website: http://medicaldecathlon.com/
All data will be made available online with a permissive copyright-license (CC-BY-SA 4.0), allowing for data to be shared, distributed and improved upon. All data has been labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use. 

### Citation for Pancreas Dataset
To cite this data, please refer to this [page](https://arxiv.org/abs/1902.09063). This dataset was pre-processed using [ShapeWorks](https://sciinstitute.github.io/ShapeWorks/latest/) mesh grooming tools. The data pre-processing implementation is based on [Mesh2SSM](https://github.com/iyerkrithika21/mesh2SSM_2023). 

## 🎓 Attribution
```bibtex
@inproceedings{elamrani2024fuss,
  author = {El Amrani, Nafie and Cao, Dongliang and Bernard, Florian},
  title = {A Universal and Flexible Framework for Unsupervised Statistical Shape Model Learning},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  month = oct,
  year = {2024}
}
```

## License 🚀
This repo is licensed under MIT licence.