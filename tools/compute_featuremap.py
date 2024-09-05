# encoding: utf-8

import argparse
import os
import sys
from functools import partial
from os import mkdir
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
import warnings
from utils.logger import setup_logger
import os.path as osp
import pickle
from collections import OrderedDict
from data.datasets import init_dataset


def main():

    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)




    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    gt_label = dataset._get_image_label(cfg.GRADCAM.IMAGE_PATH)
    print("The model is " + str(cfg.MODEL.NAME))
    print("The ground truth label is " + str(gt_label))



    model = build_model(cfg, num_classes)
    model.load_param_withfc(cfg.GRADCAM.MODEL_WEIGHT_PATH)

    if cfg.MODEL.DEVICE == "cuda":
        use_cuda = True
    else:
        use_cuda = False





    ## call grad_cam
    grad_cam = GradCam(model=model, feature_module=model.base.non_local_4, target_layer_names=[cfg.GRADCAM.TARGET_LAYER_NAMES], use_cuda=use_cuda)






    img = cv2.imread(cfg.GRADCAM.IMAGE_PATH, 1)
    path = cfg.GRADCAM.OUT_PATH
    img = cv2.resize(img, cfg.GRADCAM.IMAGE_SIZE)
    cv2.imwrite(os.path.join(path, 'original_img.jpg'), img)



    img = np.float32(img) / 255
    # print("img.shape")
    # print(img.shape)

    input = preprocess_image(img)
    # print("input.shape")
    # print(input.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.


    if cfg.GRADCAM.USE_GT_LABEL:
        target_index = gt_label
    else:
        if cfg.GRADCAM.SET_TARGET_INDEX:
            target_index = cfg.GRADCAM.TARGET_CLASS_INDEX
        else:
            target_index = None


    feature_map = grad_cam(input, target_index)

    N, C, H, W = feature_map.shape
    feature = feature_map.view(C, H*W)
    feature_np = feature.data.cpu().numpy()

    for row in range(C):
        feature_np[row, :] = feature_np[row, :] - np.min(feature_np[row, :])
        feature_np[row, :] = feature_np[row, :] / np.max(feature_np[row, :])

    # feature_np = feature_np - np.min(feature_np)
    # feature_np = feature_np / np.max(feature_np)

    heatmap = cv2.applyColorMap(np.uint8(255 * feature_np), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap / np.max(heatmap)



    # model.base.non_local_3.ch_startpos_int
    #
    # print("feature_map.shape --------------------- ")
    # print(feature_map.shape)
    # print(model.base.non_local_3.ch_startpos_int)
    # print(model.base.non_local_3.ch_startpos_dec)
    # print(model.base.non_local_3.ch_length_int)
    # print(model.base.non_local_3.ch_length_dec)
    # start_int = model.base.non_local_3.ch_startpos_int.data.cpu().numpy()
    # length_int = model.base.non_local_3.ch_length_int.data.cpu().numpy()
    # print("feature_map.shape --------------------- ")

    # index = [0, 15]
    # length = [14, 17]
    # for i in range(len(index)):
    #     heatmap[index[i], :] = 1
    #     heatmap[index[i]+length[i], :] = 1



    # heatmap[start_int, :] = 1
    # heatmap[np.mod(start_int + length_int, C), :] = 1
    cv2.imwrite(os.path.join(path, 'feature_map.jpg'), np.uint8(255 * heatmap))




class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():

            print(name)
            # print(x.size)

            if "base" in name:
                for name_base, module_base in module._modules.items():
                    if module_base == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                        intermediate_feature_map = x
                    else:
                        x = module_base(x)
            elif "gap" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "bottleneck" in name:
                x = x
            else:
                x = module(x)
        print("The size of target_activations is " + str(len(target_activations)))
        return intermediate_feature_map, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(os.path.join(path, 'show_cam_on_img.jpg'), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        feature_map = features

        return feature_map


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--result-path', type=str, default='./results/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

if __name__ == '__main__':
    main()

