import os
import os.path as osp
import shutil
import json
import h5py
import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array


class completion3D_class(InMemoryDataset):
    """The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('http://download.cs.stanford.edu/downloads/completion3d/'
            'dataset2019.zip')

    category_ids = {
        'plane': '02691156',
        'bench': '02828884',
        'cabinet': '02933112',
        'car': '02958343',
        'chair': '03001627',
        'monitor': '03211117',
        'lamp': '03636649',
        'speaker': '03691459',
        'firearm': '04090263',
        'couch': '04256520',
        'table': '04379243',
        'cellphone': '04401088',
        'watercraft': '04530566',
    }


    # seg_classes = {
    #     'Airplane': [0, 1, 2, 3],
    #     'Bag': [4, 5],
    #     'Cap': [6, 7],
    #     'Car': [8, 9, 10, 11],
    #     'Chair': [12, 13, 14, 15],
    #     'Earphone': [16, 17, 18],
    #     'Guitar': [19, 20, 21],
    #     'Knife': [22, 23],
    #     'Lamp': [24, 25, 26, 27],
    #     'Laptop': [28, 29],
    #     'Motorbike': [30, 31, 32, 33, 34, 35],
    #     'Mug': [36, 37],
    #     'Pistol': [38, 39, 40],
    #     'Rocket': [41, 42, 43],
    #     'Skateboard': [44, 45, 46],
    #     'Table': [47, 48, 49],
    # }

    def __init__(self, root, categories=None, include_normals=True,
                 split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)

        self.categories = categories
        self.split = split
        super(completion3D_class, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        if split == 'test':
            path = self.processed_paths[0]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)
        self.data.x = self.data.x if include_normals else None

        # self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50),
        #                           dtype=torch.bool)
        # for i, labels in enumerate(self.seg_classes.values()):
        #     self.y_mask[i, labels] = 1

    @property
    # all the folder names except xxx.txt
    def raw_file_names(self):
        # return list(self.category_ids.values()) + ['train_test_split']
        return list(['train', 'test', 'val', 'train.list', 'test.list', 'val.list'])

    @property
    # naming the pt files, eg : cha_air_car_test.pt, cha_air_car_train.pt
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, split))
            for split in ['test']
        ]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        # name = self.url.split('/')[-1].split('.')[0]
        name = 'shapenet'
        os.rename(osp.join(self.root, name), self.raw_dir)
        # print(osp.join(self.root, name))
        # print(self.raw_dir)
        print('end of download')

    def process_filenames(self, filenames, split_in_loop):
        data_list = []
        # categories_ids :
        # ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', 
        # '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        # i : 0 -> num of classes; 
        # cat_idx : {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4,
        # '03211117': 5, '03636649': 6, '03691459': 7, '04090263': 8, '04256520': 9, '04379243'
        # : 10, '04401088': 11, '04530566': 12}
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}
        #name : 04530566/786f18c5f99f7006b1d1509c24a9f631
        #name.split(osp.sep) : ['04530566', '786f18c5f99f7006b1d1509c24a9f631']
        for name in filenames:
            cat = name.split(osp.sep)[0]

            fpos = None
            pos = None

            if split_in_loop == 'test':
                fpos = h5py.File(osp.join(osp.join(self.raw_dir, f'{split_in_loop}/partial'), name), 'r')
                pos = torch.tensor(fpos['data'], dtype=torch.float32)

            data = Data(pos=pos)
  
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        for i, split in enumerate(['test']):
            print('in the loop')
            path = None
            # if split == 'train' or split == 'val':
            #     path = osp.join(self.raw_dir, f'{split}.list')
            if split == 'test':
                path = osp.join(self.raw_dir, 'test.list')
            with open(path, 'r') as f:
                tmp = ".h5"
                filenames = [                    
                    (name[0: -1] + tmp)
                    for name in f
                ]
            data_list = self.process_filenames(filenames, split)
            torch.save(self.collate(data_list), self.processed_paths[0])
        print('end of process()')

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)
