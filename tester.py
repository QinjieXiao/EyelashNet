import os
from os.path import join, exists,splitext, split
import cv2
import logging
import numpy as np

import torch

import utils
from   utils import CONFIG
import networks
from   utils import comput_sad_loss, compute_connectivity_error, compute_gradient_loss, compute_mse_loss


import toml
import argparse
from   pprint import pprint

from datetime import datetime
from glob import glob
import random

from   torch.utils.data import DataLoader
from   dataloader.image_file import ImageFileTrain, ImageFileTest
from   dataloader.data_generator import DataGenerator
from   dataloader.prefetcher import Prefetcher

import imageio

class ModelProxy(object):
    def __init__(self):
        self.model_path = ""
        self.version = ""
        self.SAD = 0.0
        self.MSE = 0.0
        self.Grad = 0.0
        self.Conn = 0.0
        self.use_trimap = False

    def __str__(self):
        return ("version: {}; SAD: {}; MSE: {}; Grad: {}; Conn: {}; use_trimap: {}".format(self.version, self.SAD, self.MSE, self.Grad, self.Conn, self.use_trimap))


class Tester(object):

    def __init__(self, test_dataloader):

        self.test_dataloader = test_dataloader
        self.logger = logging.getLogger("Logger")

        self.model_config = CONFIG.model
        self.test_config = CONFIG.test
        self.log_config = CONFIG.log
        self.data_config = CONFIG.data

        self.build_model()
        self.resume_step = None

        utils.print_network(self.G, CONFIG.version)

        if self.test_config.checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.test_config.checkpoint))
            self.restore_model(self.test_config.checkpoint)

    def build_model(self):
        self.G = networks.get_generator(encoder=self.model_config.arch.encoder, decoder=self.model_config.arch.decoder)
        if not self.test_config.cpu:
            self.G.cuda()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path)
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)


    def test_v2(self):
        self.G = self.G.eval()
        mse_loss = 0
        sad_loss = 0
        conn_loss = 0
        grad_loss = 0

        test_num = 0

        with torch.no_grad():
            for image_dict in self.test_dataloader:
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape, name = image_dict['alpha_shape'], image_dict['image_name']
                if not self.test_config.cpu:
                    image = image.cuda()
                    alpha = alpha.cuda()
                    trimap = trimap.cuda()
                alpha_pred, _ = self.G(image, trimap)

                if self.model_config.trimap_channel == 3:
                    trimap = trimap.argmax(dim=1, keepdim=True)

                alpha_pred[trimap == 2] = 1
                alpha_pred[trimap == 0] = 0

                trimap[trimap==2] = 255
                trimap[trimap==1] = 128

                for cnt in range(image.shape[0]):

                    h, w = alpha_shape
                    test_alpha = alpha[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = alpha_pred[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = test_pred.astype(np.uint8)
                    test_trimap = trimap[cnt, 0, ...].data.cpu().numpy()

                    test_pred = test_pred[:h, :w]
                    test_trimap = test_trimap[:h, :w]


                    mse_loss += compute_mse_loss(test_pred, test_alpha, test_trimap)
                    sad_loss += comput_sad_loss(test_pred, test_alpha, test_trimap)[0]
                    if not self.test_config.fast_eval:
                        conn_loss += compute_connectivity_error(test_pred, test_alpha, test_trimap, 0.1)
                        grad_loss += compute_gradient_loss(test_pred, test_alpha, test_trimap)

                    if self.test_config.out_path is not None:
                        cv2.imwrite(os.path.join(self.test_config.out_path, os.path.splitext(name[cnt])[0] + ".png"), test_pred)
                    if self.test_config.viz_dir is not None:
                        print(test_num, name, comput_sad_loss(test_pred, test_alpha, test_trimap)[0])
                        out_fn = os.path.join(self.test_config.viz_dir, os.path.splitext(name[cnt])[0] + ".gif")
                        gt = cv2.merge((test_alpha.astype(np.uint8),test_alpha.astype(np.uint8),test_alpha.astype(np.uint8)))
                        pred = cv2.merge((test_pred,test_pred,test_pred))

                        bgr = cv2.imread(join(self.test_config.merged, name[cnt]))

                        # rgb = (image[cnt, ...].data.cpu().numpy() * 255).astype(np.uint8)
                        # rgb = rgb.transpose((1, 2, 0))
                        # print("gt={}, pred={}, bgr={}".format(gt.shape, pred.shape, bgr.shape))
                        masked = cv2.add(bgr, pred)
                        cv2.putText(masked, "pred", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

                        gtmasked = cv2.add(bgr, gt)
                        cv2.putText(gtmasked, "GT", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        frames = []
                        frames.append(rgb)
                        frames.append(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
                        frames.append(rgb)
                        frames.append(cv2.cvtColor(gtmasked, cv2.COLOR_BGR2RGB))

                        imageio.mimsave(out_fn, frames, 'GIF', duration=0.3)


                    test_num += 1

        self.logger.info("TEST NUM: \t\t {}".format(test_num))
        self.logger.info("MSE: \t\t {}".format(mse_loss / test_num))
        self.logger.info("SAD: \t\t {}".format(sad_loss / test_num))
        if not self.test_config.fast_eval:
            self.logger.info("GRAD: \t\t {}".format(grad_loss / test_num))
            self.logger.info("CONN: \t\t {}".format(conn_loss / test_num))

    def test(self):
        self.G = self.G.eval()
        mse_loss = 0
        sad_loss = 0
        conn_loss = 0
        grad_loss = 0

        test_num = 0

        with torch.no_grad():
            for image_dict in self.test_dataloader:
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape, name = image_dict['alpha_shape'], image_dict['image_name']
                if not self.test_config.cpu:
                    image = image.cuda()
                    alpha = alpha.cuda()
                    trimap = trimap.cuda()
                alpha_pred, _ = self.G(image, trimap)

                if self.model_config.trimap_channel == 3:
                    trimap = trimap.argmax(dim=1, keepdim=True)

                alpha_pred[trimap == 2] = 1
                alpha_pred[trimap == 0] = 0

                trimap[trimap==2] = 255
                trimap[trimap==1] = 128

                for cnt in range(image.shape[0]):

                    h, w = alpha_shape
                    test_alpha = alpha[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = alpha_pred[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = test_pred.astype(np.uint8)
                    test_trimap = trimap[cnt, 0, ...].data.cpu().numpy()

                    test_pred = test_pred[:h, :w]
                    test_trimap = test_trimap[:h, :w]

                    if self.test_config.alpha_path is not None:
                        cv2.imwrite(os.path.join(self.test_config.alpha_path, os.path.splitext(name[cnt])[0] + ".png"),
                                    test_pred)

                    mse_loss += compute_mse_loss(test_pred, test_alpha, test_trimap)
                    print(name, comput_sad_loss(test_pred, test_alpha, test_trimap)[0])
                    sad_loss += comput_sad_loss(test_pred, test_alpha, test_trimap)[0]
                    if not self.test_config.fast_eval:
                        conn_loss += compute_connectivity_error(test_pred, test_alpha, test_trimap, 0.1)
                        grad_loss += compute_gradient_loss(test_pred, test_alpha, test_trimap)

                    test_num += 1

        self.logger.info("TEST NUM: \t\t {}".format(test_num))
        self.logger.info("MSE: \t\t {}".format(mse_loss / test_num))
        self.logger.info("SAD: \t\t {}".format(sad_loss / test_num))
        if not self.test_config.fast_eval:
            self.logger.info("GRAD: \t\t {}".format(grad_loss / test_num))
            self.logger.info("CONN: \t\t {}".format(conn_loss / test_num))




def main():

    # Train or Test
    if CONFIG.phase.lower() == "test":
        CONFIG.log.logging_path += "_test"
        if CONFIG.test.alpha_path is not None:
            utils.make_dir(CONFIG.test.alpha_path)
        utils.make_dir(CONFIG.log.logging_path)

        # Create a logger
        logger = utils.get_logger(CONFIG.log.logging_path,
                                  logging_level=CONFIG.log.logging_level)

        test_image_file = ImageFileTest(alpha_dir=CONFIG.test.alpha,
                                        merged_dir=CONFIG.test.merged,
                                        trimap_dir=CONFIG.test.trimap)
        test_dataset = DataGenerator(test_image_file, phase='test', test_scale=CONFIG.test.scale)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=CONFIG.test.batch_size,
                                     shuffle=False,
                                     num_workers=CONFIG.data.workers,
                                     drop_last=False)

        tester = Tester(test_dataloader=test_dataloader)
        return tester.test_v2()




if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--config', type=str, default='E:/matting/GCA-Matting-master/config/eyelash_render_dataset.toml')
    parser.add_argument('-t', '--test_dir', type=str, required=False)
    parser.add_argument('-v', '--viz', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")




    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    if CONFIG.test.alpha_path is not None:
        CONFIG.test.alpha_path = os.path.join(CONFIG.test.alpha_path, CONFIG.version)
    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
    CONFIG.local_rank = args.local_rank

    CONFIG.test.merged = join(args.test_dir, 'image')
    CONFIG.test.alpha = join(args.test_dir, 'mask')
    CONFIG.test.trimap = join(args.test_dir, 'trimap')
    CONFIG.test.fast_eval = False
    CONFIG.test.out_path = CONFIG.test.merged + "." + CONFIG.version


    if not exists(CONFIG.test.out_path):
        os.makedirs(CONFIG.test.out_path)


    if args.viz:
        CONFIG.test.viz_dir = CONFIG.test.merged + "." + CONFIG.version + ".viz"
        if not exists(CONFIG.test.viz_dir):
            os.makedirs(CONFIG.test.viz_dir)
    else:
        CONFIG.test.viz_dir = None




    # cv2.imwrite(os.path.join(args.output, image_name), pred)


    rlt = main()


    # mp.MSE = rlt[0]
    # mp.SAD = rlt[1]
    # mp.Grad = rlt[2]
    # mp.Conn = rlt[3]
    # mp.test_num = rlt[4]
    # # (mp.MSE, mp.SAD, mp.GRAD, mp.CONN, mp.test_num) = main()
    # evaluations.append(mp)
    #
    #
    # with open(root_ckpt_dir + "/evaluations.txt","a") as f:
    #     f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    #     for eval in evaluations:
    #         f.write(eval.__str__() + '\n')
    #     f.write('\n')







    # Train or Test
    # main()
