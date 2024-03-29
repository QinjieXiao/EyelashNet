import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms

from   utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
        # alpha256 = cv2.resize(alpha, None, None, fx=0.5, fy=0.5)
        # alpha128 = cv2.resize(alpha256, None, None, fx=0.5, fy=0.5)
        # alpha64 = cv2.resize(alpha128, None, None, fx=0.5, fy=0.5)
        # alpha32 = cv2.resize(alpha64, None, None, fx=0.5, fy=0.5)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)

        # alpha256 = np.expand_dims(alpha256.astype(np.float32), axis=0)
        # alpha128 = np.expand_dims(alpha128.astype(np.float32), axis=0)
        # alpha64 = np.expand_dims(alpha64.astype(np.float32), axis=0)
        # alpha32 = np.expand_dims(alpha32.astype(np.float32), axis=0)


        if CONFIG.prior == "PR":
            trimap = trimap.astype(np.float32)/128.
        else:
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1
        # xqj normalize trimap
        #trimap /=255.
        # normalize image
        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            # del sample['image_name']

        if CONFIG.prior == "PR":
            sample['trimap'] = torch.from_numpy(trimap)
        else:
            sample['trimap'] = torch.from_numpy(trimap).to(torch.long)

        sample['image'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(alpha) 
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        if CONFIG.model.trimap_channel == 3:
            if CONFIG.prior == "PR":
                sample['trimap'] = sample['trimap'].permute(2,0,1).float()
            else:
                sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha, trimap = sample['fg'], sample['alpha'], sample['trimap']
        #print("fg shape = {}, alpha shape = {}".format(fg.shape,alpha.shape))

        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        trimap = cv2.warpAffine(trimap, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'], sample['trimap'] = fg, alpha, trimap
        #print("fg shape = {}, alpha shape = {}".format(fg.shape,alpha.shape))
        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix
class RandomSwapRGB(object):
    def __call__(self, sample):
        fg = sample['fg']
        bgr = cv2.split(fg)
        np.random.shuffle(bgr)
        sample['fg'] = cv2.merge(bgr)
        return sample
class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample
class BgRandomCrop(object):
    def __call__(self, sample):
        alpha,bg = sample['alpha'],sample['bg']
        #print("alpha={}, bg={}".format(alpha.shape,bg.shape))
        h,w = alpha.shape
        bgh,bgw = bg.shape[0:2]
        ub = min([h,w,bgh,bgw])
        sz = random.randint(int(0.9*ub),ub-1)
        hoffset = random.randint(0,bgh - 1 - sz)
        woffset = random.randint(0,bgw - 1 - sz)
        new_bg = bg[hoffset:hoffset + sz,woffset:woffset+sz,:]
        new_bg = cv2.resize(new_bg,(w,h),interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        #print("alpha={}, bg={},new_bg={}".format(alpha.shape,bg.shape,new_bg.shape))
        sample['bg'] = new_bg
        return sample
class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            # cv2.imwrite('../tmp/tmp.jpg', fg.astype(np.uint8))
            # cv2.imwrite('../tmp/tmp.png', (alpha*255).astype(np.uint8))
            # cv2.imwrite('../tmp/tmp2.png', trimap.astype(np.uint8))
            # raise ValueError("{} does    not have enough unknown area for crop.".format(name))

        sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop
        sample['bg'] = bg_crop

        '''cv2.imwrite('zhangzr/fg.png',sample['fg'])
        cv2.imwrite('zhangzr/alpha.png',sample['alpha']*255)
        cv2.imwrite('zhangzr/trimap.png',sample['trimap']*255)
        cv2.imwrite('zhangzr/bg.png',sample['bg'])'''
        return sample



# def trimap_statistics(trimap):
#     bg = np.full_like(trimap,0)
#     bg[trimap == 0] = 1
#
#     unknown = np.full_like(trimap,0)
#     unknown[trimap==128] = 1
#
#     fg = np.full_like(trimap, 0)
#     fg[trimap == 255] = 1
#
#     bgnum = bg.sum()
#     fgsum = fg.sum()
#     unknownsum = unknown.sum()
#
#     total = bgnum + fgsum + unknownsum
#
#     bgrate = bgnum/total
#     fgrate = fgsum/total
#     ukrate = unknownsum/total
#
#     if bgrate > 0.9:
#         return False
#     if fgrate > 0.9:
#         return  False
#     return True

def trimap_statistics(trimap):



    unknown = np.full_like(trimap,0)
    unknown[trimap==128] = 1


    unknownsum = unknown.sum()


    ukrate = unknownsum/(trimap.shape[0]*trimap.shape[1])

    if ukrate > 0.01:
        return True
    else:
        return False

class RandomCropV2(object):
    """

    Crop randomly the auxiliary image in a sample

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        # self.logger = logging.getLogger("Logger")

    def __call__(self, sample):

        fg, alpha, trimap, bg, name = sample['fg'], sample['alpha'], sample['trimap'], sample['bg'], sample['image_name']
        h, w = trimap.shape

        bgsz = min(bg.shape[0], bg.shape[1], self.output_size[0], self.output_size[1])
        if bg.shape[0] > bgsz:
            bgtop = np.random.randint(bg.shape[0] - bgsz)
        else:
            bgtop = 0
        if bg.shape[1] > bgsz:
            bgleft = np.random.randint(bg.shape[1] - bgsz)
        else:
            bgleft = 0

        bg_crop = bg[bgtop: bgtop + bgsz, bgleft: bgleft + bgsz]
        bg_crop = cv2.resize(bg_crop, self.output_size, self.output_size, interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        sample['bg'] = bg_crop

        # if h != self.output_size[0] or w !=self.output_size[1]:
        #     sz = min(h, w, self.output_size[0], self.output_size[1])
        #     rangeh = h - sz
        #     rangew = w - sz
        #     if rangeh == 0:
        #         top = 0
        #     else:
        #         top = np.random.randint(rangeh)
        #     if rangew == 0:
        #         left = 0
        #     else:
        #         left = np.random.randint(rangew)
        #
        #     trimap_crop = trimap[top:top + sz, left: left + sz]
        #     while not trimap_statistics(trimap_crop):
        #         if rangeh == 0:
        #             top = 0
        #         else:
        #             top = np.random.randint(rangeh)
        #         if rangew == 0:
        #             left = 0
        #         else:
        #             left = np.random.randint(rangew)
        #         trimap_crop = trimap[top:top + sz, left: left + sz]
        #
        #     fg_crop    = fg[top: top + sz, left: left + sz, :]
        #     alpha_crop = alpha[top: top + sz, left: left + sz]
        #
        #     if sz < self.output_size[0]:
        #         intp = maybe_random_interp(cv2.INTER_CUBIC)
        #         fg_crop = cv2.resize(fg_crop, self.output_size, self.output_size, interpolation=intp)
        #         alpha_crop = cv2.resize(alpha_crop, self.output_size, self.output_size, interpolation=intp)
        #         trimap_crop = cv2.resize(trimap_crop, self.output_size, self.output_size, interpolation=intp)
        #     sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop

        return sample



class CompositeV2(object):
    def __call__(self, sample):
        rnd = random.random()
        img_name = sample['image_name']
        if img_name.find('AMD') or img_name.find('HEAD_PC') < 0:
            if rnd <= 0.5:
                sample['image'] = sample['fg']
            elif rnd > 0.9:
                fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']

                alpha[alpha < 0] = 0
                alpha[alpha > 1] = 1

                fg[fg < 0] = 0
                fg[fg > 255] = 255
                bg[bg < 0] = 0
                bg[bg > 255] = 255

                image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])
                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)
                sample['image'] = image

            else:
                fg, alpha = sample['fg'], sample['alpha']
                h, w = fg.shape[0:2]
                bg = cv2.imread(sample['pupil_fn'])
                bg = cv2.resize(bg, (w, h))
                alpha[alpha < 0] = 0
                alpha[alpha > 1] = 1

                fg[fg < 0] = 0
                fg[fg > 255] = 255
                bg[bg < 0] = 0
                bg[bg > 255] = 255

                # print(bg.shape)
                image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])

                # cv2.imshow("fg", fg)
                # cv2.imshow("matte fg", (fg * alpha[:, :, None]).astype(np.uint8))
                # cv2.imshow("image", image.astype(np.uint8))
                # cv2.waitKey()

                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)

                sample['image'] = image
                # cv2.imshow("", image)
                # cv2.waitKey()
                # cv2.imwrite("/home/xqj/zhangzr/GCA-Matting-master/data/pupil_bg/composite_test/" + str(sample['idx']) + '0.jpg', image.astype(np.uint8))


        else:
            if rnd<=0.5:
                fg, alpha = sample['fg'], sample['alpha']
                h, w = fg.shape[0:2]
                bg = cv2.imread(sample['pupil_fn'])
                bg = cv2.resize(bg, (w, h))
                alpha[alpha < 0] = 0
                alpha[alpha > 1] = 1

                fg[fg < 0] = 0
                fg[fg > 255] = 255
                bg[bg < 0] = 0
                bg[bg > 255] = 255

                image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])


                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)

                sample['image'] = image
            else:
                fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']

                alpha[alpha < 0] = 0
                alpha[alpha > 1] = 1

                fg[fg < 0] = 0
                fg[fg > 255] = 255
                bg[bg < 0] = 0
                bg[bg > 255] = 255

                image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])
                image[image > 255] = 255
                image[image < 0] = 0
                image = image.astype(np.uint8)
                sample['image'] = image
        # print('{}-{}-{}-{}'.format(sample['fg'].shape, sample['bg'].shape, sample['alpha'].shape, sample['image'].shape))
        return sample




class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        sample['image'], sample['alpha'], sample['trimap'] = image, alpha, trimap

        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]
        # sample['origin_trimap'] = sample['trimap']
        # # if h % 32 == 0 and w % 32 == 0:
        # #     return sample
        # # target_h = h - h % 32
        # # target_w = w - w % 32
        # target_h = 32 * ((h - 1) // 32 + 1)
        # target_w = 32 * ((w - 1) // 32 + 1)
        # sample['image'] = cv2.resize(sample['image'], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        # sample['trimap'] = cv2.resize(sample['trimap'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if h % 32 == 0 and w % 32 == 0:
            return sample
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

        return sample

class Blur(object):
    def __call__(self,sample):
        if random.random() < 0.8:
            return sample 
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        k = random.randint(5, 20)
        if k % 2 == 0:
            k = k + 1
        sample['fg'] = cv2.GaussianBlur(fg, (k, k),0)
        sample['bg'] = cv2.GaussianBlur(bg, (k, k),0)
        alpha = (alpha*255).astype(np.uint8)
        blur_alpha = cv2.GaussianBlur(alpha, (k, k),0)
        sample['alpha'] = blur_alpha.astype(np.float32)/255.
        #print("fg shape = {}, alpha shape = {}".format(sample['fg'].shape,sample['alpha'].shape))


        return sample
        

class GenGrayTrimap(object):
    def __call__(self,sample):
        alpha = sample['alpha']
        sample['trimap'] = np.full_like(sample['alpha'],128)
        return sample



class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        if CONFIG.prior == "trimap":

            image_name = sample['image_name']
            alpha = sample['alpha']
            if image_name.find('HEAD_PC') >=0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                trimap = alpha
                trimap = cv2.erode(trimap, kernel)
                trimap = cv2.dilate(trimap, kernel)
                trimap[trimap > 0] = 128
            else:
                # Adobe 1K
                fg_width = np.random.randint(1, 7)
                bg_width = np.random.randint(5, 10)
                fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
                bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

                trimap = np.ones_like(alpha) * 128
                trimap[fg_mask == 1] = 255
                trimap[bg_mask == 1] = 0





        elif CONFIG.prior == "PR":
            trimap = cv2.imread(sample['pr_fn'])
        else:
            alpha = sample['alpha']
            trimap = np.full_like(alpha,128)

        #print("fg shape = {}, alpha shape = {}".format(sample['fg'].shape,sample['alpha'].shape))

        sample['trimap'] = trimap
        
        #cv2.imwrite('zhangzr/trimap.png',sample['trimap'])
        return sample
class GammaTransform(object):
    def __init__(self, low=0.5, up=1.4):
        self.low = low
        self.up = up
    def __call__(self, sample):

        # if random.random() < 0.6:
        #     return sample
        #print("fg shape = {}, alpha shape = {}".format(sample['fg'].shape,sample['alpha'].shape))


        fg = sample['fg']
        gamma = self.low + random.random()*(self.up - self.low)

        hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum = np.power(illum, gamma)
        v = illum * 255.
        v[v > 255] = 255
        v[v < 0] = 0
        hsv[..., 2] = v.astype(np.uint8)
        sample['fg'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return sample
    def get_illum_mean_and_std(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        return np.mean(illum), np.std(illum)

class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image

        return sample



class BgComposite(object):
    def __call__(self, sample):
        rnd = random.random()
        if rnd <= 0.5:
            sample['image'] = sample['fg']
            return sample
        elif rnd > 0.9:
            fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']

            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1

            fg[fg < 0] = 0
            fg[fg > 255] = 255
            bg[bg < 0] = 0
            bg[bg > 255] = 255

            image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])
            image[image > 255] = 255
            image[image < 0] = 0
            image = image.astype(np.uint8)
            sample['image'] = image
            return sample

        else:
            fg, alpha = sample['fg'], sample['alpha']
            bg = cv2.imread(sample['pupil_fn'], 0)

            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1

            fg[fg < 0] = 0
            fg[fg > 255] = 255
            bg[bg < 0] = 0
            bg[bg > 255] = 255

            image = fg.astype(np.float) * alpha[:, :, None] + bg.astype(np.float) * (1 - alpha[:, :, None])

            # cv2.imshow("fg", fg)
            # cv2.imshow("matte fg", (fg * alpha[:, :, None]).astype(np.uint8))
            # cv2.imshow("image", image.astype(np.uint8))
            # cv2.waitKey()

            image[image > 255] = 255
            image[image < 0] = 0
            image = image.astype(np.uint8)

            sample['image'] = image
            # cv2.imshow("", image)
            # cv2.waitKey()
            # cv2.imwrite("/home/xqj/zhangzr/GCA-Matting-master/data/pupil_bg/composite_test/" + str(sample['idx']) + '0.jpg', image.astype(np.uint8))
            return sample



class FakeComposite(object):
    def __call__(self, sample):
        sample['image'] = sample['fg']

        '''cv2.imwrite('zhangzr/fg.png',sample['fg'])
        cv2.imwrite('zhangzr/alpha.png',sample['alpha']*255)
        cv2.imwrite('zhangzr/trimap.png',sample['trimap'])
        cv2.imwrite('zhangzr/composite_image.png',sample['image'])'''
        return sample

class DataGenerator(Dataset):
    def __init__(self, data, phase="train", test_scale="resize"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.alpha = data.alpha
        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.pupil_bg = data.pupil_bg
            self.pupil_bg_num = len(self.pupil_bg)
            self.merged = []
            self.trimap = []
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        if CONFIG.data.augmentation:
            train_trans = [ GammaTransform(),
                            Blur(),
                            GenTrimap(),
                            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                            #GenTrimap(),
                            # BgRandomCrop(),
                            RandomCropV2((self.crop_size, self.crop_size)),
                            RandomJitter(),
                            # CompositeV2(),
                            BgComposite(),
                            # FakeComposite(),
                            ToTensor(phase="train")]
        else:
            train_trans = [
                            GenTrimap(),
                            #GenGrayTrimap(),
                            # BgRandomCrop(),
                            RandomCrop((self.crop_size, self.crop_size)),
                            FakeComposite(),
                            ToTensor(phase="train") ]

        if test_scale.lower() == "origin":
            test_trans = [OriginScale(), ToTensor()]
        elif test_scale.lower() == "resize":
            test_trans = [Rescale((self.crop_size, self.crop_size)), ToTensor()]
        elif test_scale.lower() == "crop":
            test_trans = [RandomCrop((self.crop_size, self.crop_size)), ToTensor()]
        else:
            raise NotImplementedError("test_scale {} not implemented".format(test_scale))

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,20)]

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])

            alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)
            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name, 'pr_fn': self.fg[idx % self.fg_num].replace("image","trimap"), 'pupil_fn': self.pupil_bg[(idx + random.randint(0,100)) % self.pupil_bg_num], 'idx':idx }



        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            #trimap = cv2.imread(self.trimap[idx], 0)
            if CONFIG.prior == "trimap":
                trimap = cv2.imread(self.trimap[idx], 0)
            elif CONFIG.prior == "PR":
                trimap = cv2.imread(self.trimap[idx])
            else:
                trimap = np.full_like(alpha,128)

            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'image_name': image_name, 'alpha_shape': alpha.shape}


        sample = self.transform(sample)
        # print("fg shape = {}, alpha shape = {}, image = {}, bg = {}".format(sample['fg'].size,sample['alpha'].size), sample['image'].size, sample['bg'].size)
        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


if __name__ == '__main__':

    from dataloader.image_file import ImageFileTrain, ImageFileTest
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    CONFIG.data.augmentation = True
    matting = ImageFileTrain(alpha_dir="/home/liyaoyi/dataset/Adobe/train/alpha",
                             fg_dir="/home/liyaoyi/dataset/Adobe/train/fg",
                             bg_dir="/home/Data/coco/images2017")
    matting_test = ImageFileTrain(alpha_dir="/home/liyaoyi/dataset/Adobe/train/alpha",
                             fg_dir="/home/liyaoyi/dataset/Adobe/train/fg",
                             bg_dir="/home/Data/coco/images2017")
    data_dataset = DataGenerator(matting, phase='train')
    batch_size = 16
    num_workers = 8
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    import time

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = time.time()
    from tqdm import tqdm

    # for i, batch in enumerate(tqdm(data_loader)):
    #     image, bg, alpha = batch['image'], batch['bg'], batch['alpha']
    b = next(iter(data_loader))
    for i in range(b['image'].shape[0]):
        image = (b['image'][i]*std+mean).data.numpy()*255
        image = image.transpose(1,2,0)[:,:,::-1]
        trimap = b['trimap'][i].argmax(dim=0).data.numpy()*127
        cv2.imwrite('../tmp/'+str(i)+'.jpg', image.astype(np.uint8))
        cv2.imwrite('../tmp/'+str(i)+'.png', trimap.astype(np.uint8))
        # if i > 10:
        #     break
        # print(b['image_name'][i])
    # print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)
