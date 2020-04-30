__Point Clouds Completion__
===

![](figures/sample_results.jpg)
+ Green points are ground truth; blue points are observed points; red points are predicted points based on all observed points; pink points are predicted points based on one set of contribution points.

### Previous related works:
- [PointNet](https://arxiv.org/pdf/1612.00593.pdf)
- [PointNet++](https://arxiv.org/pdf/1706.02413.pdf)
- [FoldingNet](https://arxiv.org/pdf/1712.07262.pdf)
- [PCN](https://arxiv.org/pdf/1808.00671.pdf)

### Requirements
- Python 3.5
- Pytorch:1.4.0
- [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- CUDA 10.1
- open3D (optional for visulaization of point clouds)


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
Both point clouds completion and point clouds classification are supported on [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip) (415M) dataset. Manually download dataset and save it to `data_root/`.




### Completion3D (Weijia)
To do: training model on [Completion3D benchmark](https://completion3d.stanford.edu/).
A customized dataset would be preferred. You can find an [example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html#ShapeNet) and [tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html).


### Visualization
Visulize sample results:
```
$ cd visulaization/
$ python3 visualize_results_pro.py
```

### To do
- [x] Point clouds completion on ShapeNet
- [x] Multi GPUs implementation
- [x] Point clouds completion & classification on ModelNet40
- [ ] Point clouds completion on Completion3D (Weijia)
- [ ] Create a dataset featuring on occlusion
