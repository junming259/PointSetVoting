__Point Set Voting for Partial Point Clouds Analysis__
===

![](figures/architectures.png)
This paper proposes a general model for partial point clouds analysis wherein the
latent feature encoding a complete point clouds is inferred by applying a local
point set voting strategy. In particular, each local point set constructs a vote
that corresponds to a distribution in the latent space, and the optimal latent
feature is the one with the highest probability. We illustrates that this proposed
method achieves state-of-the-art performance on shape classification, part segmentation
and point cloud completion.


## Requirements
- Python 3.5
- Pytorch:1.4.0
- [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- CUDA 10.1
- open3D (optinoal for visualization of points clouds completion)


## Directory Structure

```
.
├── data_root
│   ├── ModelNet40 (dataset)
│   ├── ShapeNet_normal (dataset)
│   └── completion3D (dataset)
│
├── modelnet
│   ├── train_modelnet.sh
│   ├── evaluate_modelnet.sh
│   └── tensorboard.sh
│
├── shapenet_seg
│   ├── train_shapelnet.sh
│   ├── evaluate_shapelnet.sh
│   └── tensorboard.sh
│
├── completion3D
│   ├── train_completion3D.sh
│   ├── evaluate_completion3D.sh
│   └── tensorboard.sh
│
├── utils
│   ├── class_completion3D.py
│   ├── main.py
│   ├── model_utils.py
│   └── models.py
│
├── visulaization
│   ├── visualize_results_pro.py
│   └── visualize_results.py
│
├── demo
│   ├── point_cloud_completion_demo.sh
│   ├── partial_point_cloud_demo.sh
│   └── visualize.py
│
├── Dockerfile
├── build.sh
└── readme.md
```


## Preparation
The code is containterized. Build docker image:
```
$ bash build.sh
```


## Shape Classification on ModelNet40
The [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) (415M) dataset
is used to perform shape classification task. To do point clouds classification,
first download the ModelNet40 dataset and save it to `data_root/`.

+ Train the model. Specify which GPU devices to be used, and change `--gpus `
option in `train_modelnet.sh` to support multi-GPU training.
```shell
cd modelnet/
bash train_modelnet.sh
```

+ Visualize the training process by running Tensorboard.
```shell
cd modelnet/
bash tensorboard.sh
```

+ Evaluate the trained model. Make sure parameters in `evaluate_modelnet.sh`
is consistent with those in `train_modelnet.sh`.
```shell
cd modelnet/
bash evaluate_modelnet.sh
```


## Part Segmentation on ShapeNet

![](figures/part_segmentation.png)

The [ShapeNet](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
(674M) dataset is used to perform part segmentation task. Download dataset and
save it to `data_root/`. You can set the `--categories` in ` train_shapenet.sh` to
specify which category or categories of object will be trained.

+ Train the model. Specify which GPU devices to be used, and change `--gpus` option
in ` train_shapenet.sh` to support multi-GPU training.
```shell
cd shapnet_seg/
bash train_shapenet.sh
```

+ Visualize the training process by running Tensorboard.
```shell
cd shapnet_seg/
bash tensorboard.sh
```

+ Evaluate your trained model. Make sure the parameters in `evaluate_shapenet.sh`
is consistent with those in `train_shapenet.sh`. Sample predicted part segmentation
results are saved into `shapenet_seg/checkpoint/{model_name}/eval_sample_results/`.
```shell
cd shapnet_seg/
bash evaluate_shapenet.sh
```

+ Visualize sample part segmentation results. After evaluation, three `.npy` files are saved
for each sample: `pos_{idx}.npy` contains the input point clouds; `pred_{idx}.npy` contains
the predicted part labels; `label_{idx}.npy` contains the ground-truth labels. Sample output 
can be found [here](figures/visualization_part_segmentation.png).
```shell
cd visulaization/
python3 visualize_part_segmentation.py --model_name {model_name} --idx {idx}
``` 


## Point Cloud Completion on Completion3D

![](figures/completion.png)

The [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip)
(1.5GB) dataset is used to evaluate 3D object point cloud completion methods.
Specifically, partial point clouds are taken as inputs and the goal is to infer
complete point clouds . Download dataset and save it to `data_root/`. You can
set the `--categories` in ` train_completion3D.sh` to specify which category or
categories of object will be trained.

+ Train the model. Specify which GPU devices to be used, and change `--gpus` option
in ` train_completion3D.sh` to support multi-GPU training.
```shell
cd completion3D/
bash train_completion3D.sh
```

+ Evaluate your trained model. Make sure the parameters in `evaluate_completion3D.sh`
is consistent with those in `train_completion3D.sh`. Sample predicted part segmentation
results are saved into `completion3D/checkpoint/{model_name}/eval_sample_results/`.
```shell
cd completion3D/
bash evaluate_completion3D.sh
```

+ Visualize sample point clouds completionn results. After evaluation, four
`.npy` files are saved for each sample: `pos_{idx}.npy` contains the complete
point clouds; `pred_{idx}.npy` contains the predicted complete point clouds;
`pos_observed_{idx}.npy` contains the observed partial point clouds;
`pred_diverse_{idx}.npy` contains a diverse predicted completion point clouds.
Sample output can be found
[here](figures/visualization_point_clouds_completion.png).
```shell
cd visulaization/
python3 visualize_point_clouds_completion.py --model_name {model name} --idx {idx}
``` 


## Demo 
Here we provide a quick demo for point cloud completion. Specically, the
[pretraiend model](https://drive.google.com/drive/folders/1P96RiD1ODsOTum6A0VLwKpcp4P94-tUM?usp=sharing)
(pretrained only on cars from ShapeNet) is used to do point cloud completion on
partial point clouds of vehicles generated from KITTI. The partial point cloud
generation process can be found in
[here](https://github.com/junming259/Partial_Point_Clouds_generatioin). Note
that input point clouds should be in `.npy` format and in the shape of `(N, 3)`.
For example, your input point clouds are in the `demo/demo_inputs/*.npy`

```python
cd demo/
python3 point_cloud_completion_demo.py \
--data_path ${Path of partial point clouds} \
--checkpoint ${Path of pretrained model}
```

After running, predicted compeltion results will be saved in the
`demo/demo_results/`. Then visualize the results by running:

```python
python3 visualize.py \
--data_path ${Input partial point clouds}
```

`--data_path` can be either set to a certain point cloud, such as
`demo/demo_inputs/000000_car_point_1.npy`, or a directory containing input
parital point clouds, such as `demo/demo_inputs`. In the later case, a random
sample from `demo/demo_inputs/` will be selected to visualize. 


## Citation
If you find this project useful in your research, please consider cite:
```
@article{pointsetvoting,
    title={Point Set Voting for Partial Point Cloud Analysis},
    author={Zhang, Junming and Chen, Weijia and Wang, Yuping and Vasudevan, Ram and Johnson-Roberson, Matthew},
    journal={arXiv preprint arXiv:2007.04537},
    year={2020}
}
```
