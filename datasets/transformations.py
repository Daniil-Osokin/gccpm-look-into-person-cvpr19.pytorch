import random

import cv2
import numpy as np


class SinglePersonBodyMasking(object):
    def __init__(self, prob=0.5, percentage=0.3, mask_color=(128, 128, 128)):
        super().__init__()
        self._prob = prob
        self._percentage = percentage
        self._mask_color = mask_color

    def __call__(self, sample):
        image = sample['image']
        h, w, c = image.shape
        if random.random() > self._prob:
            center_x = random.randint(w // 3, w - 1 - w // 3)
            center_y = random.randint(h // 3, h - 1 - h // 3)
            kpt = [
                [center_x - random.randint(1, int(w * self._percentage)), center_y - random.randint(1, int(h * self._percentage))],
                [center_x + random.randint(1, int(w * self._percentage)), center_y - random.randint(1, int(h * self._percentage))],
                [center_x + random.randint(1, int(w * self._percentage)), center_y + random.randint(1, int(h * self._percentage))],
                [center_x - random.randint(1, int(w * self._percentage)), center_y + random.randint(1, int(h * self._percentage))]
            ]
            cv2.fillConvexPoly(image, np.array(kpt, dtype=np.int32), (128, 128, 128))
        return sample


class SinglePersonFlip(object):
    def __init__(self, prob=0.5):
        super().__init__()
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        do_flip = prob <= self._prob
        if not do_flip:
            return sample

        sample['image'] = cv2.flip(sample['image'], 1)

        w, h = sample['image'].shape[1], sample['image'].shape[0]
        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            sample['keypoints'][id * 3] = w - 1 - sample['keypoints'][id * 3]
        sample['keypoints'] = self._swap_left_right(sample['keypoints'])

        return sample

    def _swap_left_right(self, keypoints):
        right = [0, 1, 2, 3, 4, 5, 6, 7, 8, 30, 31, 32, 33, 34, 35, 36, 37, 38]
        left = [15, 16, 17, 12, 13, 14, 9, 10, 11, 45, 46, 47, 42, 43, 44, 39, 40, 41]
        for r, l in zip(right, left):
            keypoints[r], keypoints[l] = keypoints[l], keypoints[r]
        return keypoints


class ChannelPermutation(object):
    def __init__(self, prob=0.5):
        super().__init__()
        self._prob = prob

    def __call__(self, sample):
        prob = random.random()
        if prob > 0.5:
            new_order = np.random.permutation(3)
            image = sample['image']
            image[:, :, 0], image[:, :, 1], image[:, :, 2] =\
                image[:, :, new_order[0]], image[:, :, new_order[1]], image[:, :, new_order[2]]
            sample['image'] = image
        return sample


class SinglePersonRotate(object):
    def __init__(self, pad=(128, 128, 128), max_rotate_degree=40):
        self._pad = pad
        self._max_rotate_degree = max_rotate_degree

    def __call__(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self._max_rotate_degree
        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]
        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self._pad)

        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            point = (sample['keypoints'][id * 3], sample['keypoints'][id * 3 + 1])
            point = self._rotate(point, R)
            sample['keypoints'][id * 3], sample['keypoints'][id * 3 + 1] = point
        return sample

    def _rotate(self, point, R):
        return (R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2])


class SinglePersonCropPad(object):
    def __init__(self, pad, crop_x=256, crop_y=256):
        self._pad = pad
        self._crop_x = crop_x
        self._crop_y = crop_y

    def __call__(self, sample):
        img = sample['image']
        rnd_scale = 1
        rnd_offset_x = 0
        rnd_offset_y = 0

        if random.random() > 0.5:
            rnd_scale = random.random() * 0.7 + 0.8
            h, w, _ = img.shape
            scaled_img = cv2.resize(img, dsize=None, fx=rnd_scale, fy=rnd_scale, interpolation=cv2.INTER_CUBIC)
            sh, sw, _ = scaled_img.shape
            if rnd_scale >= 1:  # random crop from upsampled image
                rnd_offset_x = (sw - w) // 2
                rnd_offset_y = (sh - h) // 2
                img = scaled_img[rnd_offset_y:rnd_offset_y + h, rnd_offset_x:rnd_offset_x + w]
                rnd_offset_x *= -1
                rnd_offset_y *= -1
            else:  # pad to original size
                rnd_offset_x = (w - sw) // 2
                rnd_offset_y = (h - sh) // 2
                b_border = h - sh - rnd_offset_y
                r_border = w - sw - rnd_offset_x
                img = cv2.copyMakeBorder(scaled_img, rnd_offset_y, b_border, rnd_offset_x, r_border,
                                         borderType=cv2.BORDER_CONSTANT, value=self._pad)

        scale = self._crop_x / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        offset_x = (self._crop_x - img.shape[1]) // 2
        offset_y = (self._crop_y - img.shape[0]) // 2

        padded_img = np.ones((self._crop_y, self._crop_x, 3), dtype=np.uint8) * self._pad
        padded_img[offset_y:offset_y+img.shape[0], offset_x:offset_x+img.shape[1], :] = img
        sample['image'] = padded_img

        for id in range(len(sample['keypoints']) // 3):
            if sample['keypoints'][id * 3] == -1:
                continue
            sample['keypoints'][id * 3] = (sample['keypoints'][id * 3] * rnd_scale + rnd_offset_x) * scale + offset_x
            sample['keypoints'][id * 3 + 1] = (sample['keypoints'][id * 3 + 1] * rnd_scale + rnd_offset_y) * scale + offset_y

        return sample
