__Incomplete Point Clouds Analysis__
===

![](figures/sample_results.jpg)
+ Green points are ground truth; blue points are observed points; red points are predicted points based on all observed points; pink points are predicted points based on one set of contribution points.

### Previous related works:
- [PointNet](https://arxiv.org/pdf/1612.00593.pdf)
- [PointNet++](https://arxiv.org/pdf/1706.02413.pdf)
- [FoldingNet](https://arxiv.org/pdf/1712.07262.pdf)
- [PCN](https://arxiv.org/pdf/1808.00671.pdf)
- [TopNet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)

### Requirements
- Python 3.5
- Pytorch:1.4.0
- [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- CUDA 10.1
- open3D (optional for visulaization of point clouds)

### Directory Structure

```
.
├── data_root
│   ├── ModelNet40 (dataset)
│   ├── ShapeNet_normal (dataset)
│   └── completion3D (dataset, to do)
│
├── modelnet
│   ├── train_modelnet.sh
│   ├── evaluate_modelnet.sh
│   └── tensorboard.sh
│
├── shapenet
│   ├── train_shapelnet.sh
│   ├── evaluate_shapelnet.sh
│   └── tensorboard.sh
│
├── completion3D (to do)
│   ├── train_completion3D.sh
│   ├── evaluate_completion3D.sh
│   └── tensorboard.sh
│
├── utils
│   ├── main.py
│   ├── model_utils.py
│   └── models.py
│
├── visulaization
│   ├── visualize_results_pro.py
│   └── visualize_results.py
│
├── Dockerfile
├── build.sh
└── readme.md
```


### Preparation
The code is containterized. Build docker image:
```
$ bash build.sh
```

### ShapeNet
Currently only point clouds completion is supported on [ShapeNet](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) (674M) dataset. Manually download dataset and save it to `data_root/`. You can set the ```--categories``` in ``` train_shapenet.sh``` to specify which category or categories of object will be used.

+ Train the model. Specify which GPU devices to be used, and change ```--gpus ``` option in ``` train_shapenet.sh``` to support multi-GPU training.
```
$ cd shapnet/
$ bash train_shapenet.sh
```

+ Visualize the training process by running Tensorboard.
```
$ cd shapnet/
$ bash tensorboard.sh
```

+ Evaluate your trained model. Make sure the ```--checkpoint``` in ```evaluate_shapenet.sh``` is consistent with the one in ``` train_shapenet.sh```.
```
$ cd shapnet/
$ bash evaluate_shapenet.sh
```

+ Visualize sample completion results. After evaluation sample completion results are saved in ```shapnet/checkpoint/{model_name}/eval_sample_results/```. The results can be visualized by running:
```
$ cd visulaization/
$ python3 visualize_results_pro.py
```


### ModelNet40
Both point clouds completion and point clouds classification are supported on [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) (415M) dataset. Training model to do point clouds completion is similar to that on ShapeNet. To do point clouds classification, first download the ModelNet40 dataset and save it to `data_root/`.

+ Train the model. Specify which GPU devices to be used, and change ```--gpus ``` option in ``` train_modelnet.sh``` to support multi-GPU training.
```
$ cd modelnet/
$ bash train_modelnet.sh
```

+ Visualize the training process by running Tensorboard.
```
$ cd modelnet/
$ bash tensorboard.sh
```

+ Evaluate your trained model. Make sure the ```--checkpoint``` in ```evaluate_modelnet.sh``` is consistent with the one in ``` train_modelnet.sh```.
```
$ cd modelnet/
$ bash evaluate_modelnet.sh
```


### Completion3D (Weijia)
To do: training model on [Completion3D benchmark](https://completion3d.stanford.edu/).
A customized dataset would be preferred. You can find an [example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html#ShapeNet) and [tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html).

The finished dataset class you implement should be used similarly to [ModelNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ModelNet) and [ShapeNet](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ShapeNet). Please check the function ```load_dataset()``` in ```utils/main.py``` to see how the dataset object is defined. Please check the function ```train_one_epoch()``` in ```utils/main.py``` to see how the dataset is loading the data and what the output looks like.

Your job is to train and evaluate our proposed method on Completion3D dataset. Specifically, create a similar table as the Table 1. in the [TopNet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf) and add a column showing our results.  



### To do
- [x] Point clouds completion on ShapeNet
- [x] Multi GPUs implementation
- [x] Point clouds completion & classification on ModelNet40
- [ ] Point clouds completion on Completion3D (Weijia)
- [ ] Create a dataset featuring on occlusion
