import os
import glob
import logging
import functools
import numpy as np

EXT = '.jpg'
BG_EXT = '.jpg'
class ImageFile(object):
    def __init__(self, phase='train'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            self.logger.error('No image valid')
        else:
            self.logger.info('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 pupil_bg_dir="pupil_bg",
                 alpha_ext=EXT,
                 fg_ext=EXT,
                 bg_ext=BG_EXT,
                 pupil_bg_ext=BG_EXT):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext
        self.pupil_bg_ext = pupil_bg_ext

        self.pupil_bg_dir = pupil_bg_dir

        self.logger.debug('Load Training Images From Folders')

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]
        self.valid_pupil_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.pupil_bg_dir)]



        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)
        self.pupil_bg = self._list_abspath(self.pupil_bg_dir, self.pupil_bg_ext, self.valid_pupil_bg_list)

        #shuffle
        randomize = np.arange(len(self.fg))
        np.random.shuffle(randomize)
        self.fg = np.array(self.fg)[randomize]
        self.fg = self.fg.tolist()

        self.alpha = np.array(self.alpha)[randomize]
        self.alpha = self.alpha.tolist()
    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=EXT,
                 merged_ext=EXT,
                 trimap_ext=EXT):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.logger.debug('Load Testing Images From Folders')

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)
        #print(self.valid_image_list)
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    train_data = ImageFileTrain(alpha_dir="/home/liyaoyi/dataset/Adobe/train/alpha",
                                fg_dir="/home/liyaoyi/dataset/Adobe/train/fg",
                                bg_dir="/home/Data/coco/images2017",
                                alpha_ext=EXT,
                                fg_ext=EXT,
                                bg_ext=EXT)
    test_data = ImageFileTest(alpha_dir="/home/liyaoyi/dataset/Adobe/test/alpha",
                              merged_dir="/home/liyaoyi/dataset/Adobe/test/merged",
                              trimap_dir="/home/liyaoyi/dataset/Adobe/test/trimap",
                              alpha_ext=EXT,
                              merged_ext=EXT,
                              trimap_ext=EXT)

    print(train_data.alpha[0], train_data.fg[0], train_data.bg[0])
    print(len(train_data.alpha), len(train_data.fg), len(train_data.bg))
    print(test_data.alpha[0], test_data.merged[0], test_data.trimap[0])
    print(len(test_data.alpha), len(test_data.merged), len(test_data.trimap))
