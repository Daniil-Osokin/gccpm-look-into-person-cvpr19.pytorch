import math
import os
import random

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset


class LipTrainDataset(Dataset):
    num_keypoints = 16

    def __init__(self, dataset_folder, stride, sigma, transform=None):
        super().__init__()
        self._dataset_folder = dataset_folder
        self._stride = stride
        self._sigma = sigma
        self._transform = transform
        self._labels = [line.rstrip('\n') for line in
                        open(os.path.join(self._dataset_folder, 'TrainVal_pose_annotations', 'lip_train_set.csv'), 'r')]

    def __getitem__(self, idx):
        tokens = self._labels[idx].split(',')
        image = cv2.imread(os.path.join(self._dataset_folder, 'TrainVal_images', 'train_images', tokens[0]), cv2.IMREAD_COLOR)
        keypoints = np.ones(LipTrainDataset.num_keypoints*3, dtype=np.float32) * -1
        for id in range(keypoints.shape[0]//3):
            if tokens[1 + id*3] != 'nan':
                keypoints[id * 3] = int(tokens[1 + id*3])          # x
                keypoints[id * 3 + 1] = int(tokens[1 + id*3 + 1])  # y
                keypoints[id * 3 + 2] = 1                          # visible == 1, not visible == 0
                if int(tokens[1 + id*3 + 2]) == 1:
                    keypoints[id * 3 + 2] = 0

        sample = {
            'keypoints': keypoints,
            'image': image,
        }
        if self._transform:
            sample = self._transform(sample)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        return sample

    def __len__(self):
        return len(self._labels)

    def _generate_keypoint_maps(self, sample):
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(LipTrainDataset.num_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        keypoints = sample['keypoints']
        for id in range(len(keypoints) // 3):
            if keypoints[id * 3] == -1:
                continue
            self._add_gaussian(keypoint_maps[id], keypoints[id * 3], keypoints[id * 3 + 1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)

        return keypoint_maps

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                     (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1


class LipValDataset(Dataset):
    def __init__(self, dataset_folder, num_images=-1):
        super().__init__()
        self._dataset_folder = dataset_folder
        self.labels_file_path = os.path.join(self._dataset_folder, 'TrainVal_pose_annotations', 'lip_val_set.csv')
        self._labels = [line.rstrip('\n') for line in open(self.labels_file_path, 'r')]
        if num_images > 0:
            self._labels = self._labels[:num_images]

    def __getitem__(self, id):
        tokens = self._labels[id].split(',')
        image = cv2.imread(os.path.join(self._dataset_folder, 'TrainVal_images', 'val_images', tokens[0]), cv2.IMREAD_COLOR)
        sample = {
            'image': image,
            'file_name': tokens[0]
        }
        return sample

    def __len__(self):
        return len(self._labels)


class LipTestDataset(Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self._dataset_folder = dataset_folder
        self._names = [line.rstrip('\n') for line in open(os.path.join(self._dataset_folder, 'Testing_images', 'test_id.txt'), 'r')]

    def __getitem__(self, id):
        name = '{}.jpg'.format(self._names[id])
        img = cv2.imread(os.path.join(self._dataset_folder, 'Testing_images', 'testing_images', name))
        sample = {
            'image': img,
            'file_name': name
        }
        return sample

    def __len__(self):
        return len(self._names)
