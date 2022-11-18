#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download_modelnet():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_modelnet(partition):
    download_modelnet()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def download_scanobjectnn():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'ScanObjectNN')):
        www = 'https://download.cs.stanford.edu/orion/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s/%s' % ('h5_files', DATA_DIR, 'ScanObjectNN'))
        os.system('rm %s' % (zipfile))


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) 
    return pointcloud


def normalize_pointcloud(pointcloud):
    pointcloud -= pointcloud.mean(0)
    d = ((pointcloud**2).sum(-1)**(1./2)).max()
    pointcloud /= d
    return pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
   
   
class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='train', ver = "easy"):
        self.num_points = num_points
        self.partition = partition 
        self.ver = ver
        download_scanobjectnn()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        
        if ver=="easy":
            if self.partition == 'train':
                file = os.path.join(DATA_DIR, "ScanObjectNN/main_split_nobg/training_objectdataset.h5")
            else:
                file = os.path.join(DATA_DIR, "ScanObjectNN/main_split_nobg/test_objectdataset.h5")
                
        elif ver =="hard":
            if self.partition == 'train':
                file = os.path.join(DATA_DIR, "ScanObjectNN/main_split/training_objectdataset_augmentedrot_scale75.h5")
            else:
                file = os.path.join(DATA_DIR, "ScanObjectNN/main_split/test_objectdataset_augmentedrot_scale75.h5")
                
        else:
            raise NotImplementedError

        f = h5py.File(file, 'r')
        self.data = f['data'][:].astype('float32')
        self.label = f['label'][:].astype('int64')
        f.close()

    def __getitem__(self, item):
        idx = np.arange(2048)
        np.random.shuffle(idx)
        idx = idx[:self.num_points]
        pointcloud = self.data[item][idx]
        label = self.label[item]

        pointcloud = normalize_pointcloud(pointcloud)
        
        if self.partition == 'train':
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]