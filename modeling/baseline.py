# encoding: utf-8


import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.mobilenetv2 import mobilenetv2_x1_0, mobilenetv2_x1_4, mobilenetv2_x2_0, mobilenetv2_x1_0_lastS_1
from .backbones.mobilenetv3 import mobilenetv3_large_100
from .backbones.mobilenetv2_large import mobilenetv2_x1_0_large, mobilenetv2_x1_4_large, mobilenetv2_x2_0_large
from .backbones.hamobile import hamobilenet
from .backbones.mobilenetv2_modified import mobilenetv2_modified
from .backbones.attn_mobile import attn_mobilenet
from .backbones.mobilenetv2_edited import mobilenetv2_x1_0_edited
from .backbones.mobilenetv2_deeper import mobilenetv2_107, mobilenetv2_161, mobilenetv2_200, mobilenetv2_53, mobilenetv2_300
from .backbones.hacnn_original import HACNN, hacnn_256_24_28, hacnn_256_40_56, hacnn_256_80_56
from .backbones.ha_mobile import ha_mobilenet_global_local, ha_mobilenet_global_only
from .backbones.ha_mobile_modified import ha_mobilenet_modified_global_local, ha_mobilenet_modified_global_only, ha_mobilenet_modified_local
from .backbones.searched_mobilenetv2.mobilenet_supernet import autonl_l, mobilenetv2_nl, mobilenetv2_baseline, mobilenetv2_stage_nl
from .backbones.mobilenetv2_RGA.mobilenetv2_RGA import mobilenetv2_rga, mobilenetv2_rga_attend_3
from .backbones.searched_mobilenetv2.lightNL import mobilenetv2_53_lightNL_c025, mobilenetv2_53_lightNL_c100
from .backbones.searched_mobilenetv2.stage_lightNL import mobilenetv2_53_stage_lightNL, mobilenetv2_200_stage_lightNL, mobilenetv2_300_stage_lightNL
from .backbones.searched_mobilenetv2.stage_bottle_lightNL import mobilenetv2_53_stage_bottle_lightNL
from .backbones.pyramid_models.pyramid_reconfig import mobilenetv2_pyramidRC
from .backbones.super_mobilenetv2 import super_mobilenetv2_53
from .backbones.mobilenetv2_slimmed import mobilenetv2_53_slimmed
from .backbones.multiW_mobilenetv2_slimmed import multiW_mobilenetv2_53
from .backbones.multiW_compressed import multiW_compressed_mobilenetv2_53
from .backbones.multiW_compressed_all import multiW_compressed_all_mobilenetv2_53, mobilenetv2_300_compress_S3
from .backbones.mobilenetv2_bottle import mobilenetv2_300_bottle
from .backbones.compress_300_w1_baseline import mobilenetv2_300_w1S309_baseline
from .backbones.mobilenetv2_dnl.mobilenetv2_dnl import mobilenetv2_53_stage_dNL, mobilenetv2_300_stage_dNL
from .backbones.mobilenetv2_dnl.mobilenetv2_dnl_new import mobilenetv2_53_stage_dNL_both
from .backbones.mobilenetv2_dnl.mobilenetv2_dnl_new_v2 import mobilenetv2_53_stage_dNL_bothv2, mobilenetv2_200_stage_dNL_bothv2, mobilenetv2_300_stage_dNL_bothv2
from .backbones.searched_mobilenetv2.double_stage_lightNL import mobilenetv2_53_stage_double_lightNL
from .backbones.osnet import osnet_x1_0
from .backbones.searched_mobilenetv2.double_stage_osnet import double_baseline_osnet_ibn_x1_0, double_baseline_osnet_x1_0
from .backbones.searched_mobilenetv2.stage_lightNL_my import mobilenetv2_53_stage_lightNL_My
from .backbones.searched_mobilenetv2.stage_lightNL_linear import mobilenetv2_53_stage_lightNL_linear
from .backbones.final_version.mobilenetv2_DSP_VCH import mobilenetv2_53_DSP_VCH, mobilenetv2_200_DSP_VCH
from .backbones.final_version.finetune_mobile import mobilenetv2_53_two_step_finetune, mobilenetv2_200_two_step_finetune
from .backbones.searched_mobilenetv2.Stage_ResNet import StageResNet, Stage_ResNet50
from .backbones.FBNetV2_dnl.FBNetV2_dnl_3x_supernet import FBNetV2_3x_dnl_supernet
from .backbones.FBNetV2_dnl.FBNetV2_dnl_3x import FBNetV2_3x_dnl, FBNetV2_3x_dnl_searched
from .backbones.FBNetV2_dnl.FBNetV2_dnl_6x import FBNetV2_6x_dnl, FBNetV2_6x_dnl_searched
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, cfg):
        super(Baseline, self).__init__()

        # last_stride, model_path, neck, neck_feat, model_name, pretrain_choice
        # cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        neck = cfg.MODEL.NECK
        neck_feat = cfg.TEST.NECK_FEAT
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        feat_dim = cfg.MODEL.FEAT_DIM
        self.backbone_pretrain = cfg.MODEL.BACKBONE_PRETRAIN
        self.loss_branch = cfg.MODEL.LOSS_BRANCH
        self.SEARCH_FBNETV2 = cfg.SOLVER.SEARCH_FBNETV2

        self.model_name = model_name
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512

            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':

            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == 'mobilenetv2':
            self.in_planes = 1280
            self.base = mobilenetv2_x1_0(num_classes, pretrain_choice)
        elif model_name =='mobilenetv2_lastS_1':
            self.in_planes = 1280
            self.base = mobilenetv2_x1_0_lastS_1(num_classes, pretrain_choice)
        elif model_name == 'mobilenetv2_edited':
            self.in_planes = 1280
            self.base = mobilenetv2_x1_0_edited(num_classes, pretrain_choice)
        elif model_name == 'mobilenetv2_1dot4':
            self.in_planes = 1792
            self.base = mobilenetv2_x1_4(num_classes, pretrain_choice)
        elif model_name == 'mobilenetv2_2dot0':
            self.in_planes = int(1280*2)
            self.base = mobilenetv2_x2_0(num_classes, pretrain_choice)
        elif model_name == 'mobilenetv3_large':
            self.in_planes = 1280
            self.base = mobilenetv3_large_100(pretrained=True, last_stride=last_stride)
        elif model_name == 'mobilenetv2_large':
            self.in_planes = 1280
            self.base = mobilenetv2_x1_0_large(num_classes, 'none')
        elif model_name == 'mobilenetv2_large_1dot4':
            self.in_planes = int(1280*1.4)
            self.base = mobilenetv2_x1_4_large(num_classes, 'none')
        elif model_name == 'mobilenetv2_large_2dot0':
            self.in_planes = int(1280*2)
            self.base = mobilenetv2_x2_0_large(num_classes, 'none')
        elif model_name == 'hamobile':
            self.in_planes = 2560
            self.base = hamobilenet(num_classes, pretrain_choice='imagenet')
        elif model_name == 'mobilenetv2_modified':
            self.in_planes = 1280
            self.base = mobilenetv2_modified(num_classes, pretrain_choice='imagenet')
        elif model_name == 'attn_mobilenet':
            self.in_planes = 2560
            self.base = attn_mobilenet(num_classes, pretrain_choice='imagenet')


        elif model_name == 'hacnn':
            # original hacnn is actually hacnn_160_24_28
            self.base = HACNN(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hacnn_256_40_56':
            self.base = hacnn_256_40_56(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hacnn_256_80_56':
            self.base = hacnn_256_80_56(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hacnn_256_24_28':
            self.base = hacnn_256_24_28(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hamobile_global_local':
            self.base = ha_mobilenet_global_local(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hamobile_global_only':
            self.base = ha_mobilenet_global_only(cfg, num_classes)
            self.loss_branch = 2
            # self.loss_branch change to 2 for the forward computing
        elif model_name == 'hamobile_modified_global_local':
            self.base = ha_mobilenet_modified_global_local(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hamobile_modified_global_only':
            self.base = ha_mobilenet_modified_global_only(cfg, num_classes)
            self.loss_branch = 2
        elif model_name == 'hamobile_modified_local':
            self.base = ha_mobilenet_modified_local(cfg, num_classes)
            self.loss_branch = 2





        elif model_name == 'mobilenetv2_53':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)
        elif model_name == 'mobilenetv2_107':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_107(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)
        elif model_name == 'mobilenetv2_161':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_161(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)
        elif model_name == 'mobilenetv2_200':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_200(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)
        elif model_name == 'mobilenetv2_300':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)
        elif model_name == 'mobilenetv2_300_bottle':
            self.in_planes = 1280
            self.base = mobilenetv2_300_bottle(num_classes, last_stride=last_stride, width_mult=cfg.MODEL.WIDTH_MULT)

        elif model_name == 'mobilenetv2_53_pyramidRC':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_pyramidRC(num_classes, last_stride=last_stride)

        elif model_name == 'mobilenetv2_53_lightNL_c100_BN':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_lightNL_c100(num_classes, last_stride=last_stride, nl_norm_method='batch_norm', pre_train=False)
        elif model_name == 'mobilenetv2_53_lightNL_c100_IN':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_lightNL_c100(num_classes, last_stride=last_stride, nl_norm_method='instance_norm', pre_train=False)
        elif model_name == 'mobilenetv2_53_lightNL_c025':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_lightNL_c025(num_classes, last_stride=last_stride, pre_train=False)

        elif model_name == 'mobilenetv2_53_stage_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_lightNL(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain, nl_c=cfg.MODEL.NL_C)
        elif model_name == 'mobilenetv2_200_stage_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_200_stage_lightNL(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain)
        elif model_name == 'mobilenetv2_300_stage_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300_stage_lightNL(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain)




        elif model_name == 'mobilenetv2_53_stage_bottle_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_bottle_lightNL(num_classes, last_stride=last_stride, pre_train=False)

        elif model_name == 'mobilenetv2_rga':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_rga(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain)
        elif model_name == 'mobilenetv2_rga_attend_3':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_rga_attend_3(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain)

        elif model_name == 'autonl_l':
            self.base = autonl_l(num_classes, last_stride=last_stride)
            self.loss_branch = 2
        elif model_name == 'mobilenetv2_nl':
            self.base = mobilenetv2_nl(num_classes, last_stride=last_stride, nl_c=0.25)
            self.loss_branch = 2
        elif model_name == 'mobilenetv2_nl_baseline':
            self.base = mobilenetv2_baseline(num_classes, last_stride=last_stride)
            self.loss_branch = 2
        elif model_name == 'mobilenetv2_stage_nl':
            self.base = mobilenetv2_stage_nl(num_classes, last_stride=last_stride)
            self.loss_branch = 2
        elif model_name == 'mobilenetv2_nl_c_ratio_1':
            self.base = mobilenetv2_nl(num_classes, last_stride=last_stride, nl_c=1)
            self.loss_branch = 2

        elif model_name == 'super_mobilenetv2_ratio_2':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = super_mobilenetv2_53(num_classes, last_stride=last_stride)
        elif model_name == 'mobilenetv2_53_slimmed':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_slimmed(num_classes, last_stride=last_stride)
        elif model_name == 'mobilenetv2_53_slimmed_baseline':
            self.base = mobilenetv2_53(num_classes, last_stride=last_stride, width_mult=0.5)

        elif model_name == 'multiW_mobilenetv2_53':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = multiW_mobilenetv2_53(num_classes, last_stride=last_stride)

        elif model_name == 'multiW_compressed_mobilenetv2_53':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = multiW_compressed_mobilenetv2_53(num_classes, last_stride=last_stride)

        elif model_name == 'multiW_compressed_all_mobilenetv2_53':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = multiW_compressed_all_mobilenetv2_53(num_classes, last_stride=last_stride)

        elif model_name == 'mobilenetv2_300_compress_S3':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300_compress_S3(num_classes, last_stride=last_stride)
        elif model_name == 'mobilenetv2_300_w1S309_baseline':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300_w1S309_baseline(num_classes, last_stride=last_stride)

        elif model_name == 'mobilnetv2_53_stage_dnl':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_dNL(num_classes, last_stride)

        elif model_name == 'mobilnetv2_300_stage_dnl':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300_stage_dNL(num_classes, last_stride)

        elif model_name == 'mobilnetv2_53_stage_dnl_both':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_dNL_both(num_classes, last_stride)

        elif model_name == 'mobilnetv2_53_stage_dnl_bothv2':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_dNL_bothv2(num_classes, last_stride)

        elif model_name == 'mobilnetv2_200_stage_dnl_bothv2':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_200_stage_dNL_bothv2(num_classes, last_stride)

        elif model_name == 'mobilnetv2_300_stage_dnl_bothv2':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_300_stage_dNL_bothv2(num_classes, last_stride)

        elif model_name == 'mobilenetv2_53_stage_double_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_double_lightNL(num_classes, last_stride, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH)

        elif model_name == 'osnet_x1_0':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = osnet_x1_0(num_classes, cfg.MODEL.BACKBONE_PRETRAIN)

        elif model_name == 'mobilenetv2_53_stage_double_lightNL':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_double_lightNL(num_classes, last_stride, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH)

        elif model_name == 'double_baseline_osnet_x1_0':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = double_baseline_osnet_x1_0(num_classes, cfg.MODEL.BACKBONE_PRETRAIN, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH)

        elif model_name == 'double_baseline_osnet_ibn_x1_0':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = double_baseline_osnet_ibn_x1_0(num_classes, cfg.MODEL.BACKBONE_PRETRAIN, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH)

        elif model_name == 'mobilenetv2_53_DSP_VCH':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_DSP_VCH(num_classes,
                                               pre_train=cfg.MODEL.BACKBONE_PRETRAIN,
                                               sp_NL_opt=cfg.MODEL.DOUBLE_SP,
                                               ch_NL_opt=cfg.MODEL.DOUBLE_CH,
                                               VCH_nls=cfg.DNL.VCH_nls
                                               )

        elif model_name == 'mobilenetv2_200_DSP_VCH':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_200_DSP_VCH(num_classes,
                                               pre_train=cfg.MODEL.BACKBONE_PRETRAIN,
                                               sp_NL_opt=cfg.MODEL.DOUBLE_SP,
                                               ch_NL_opt=cfg.MODEL.DOUBLE_CH,
                                               VCH_nls=cfg.DNL.VCH_nls
                                               )
        elif model_name == 'mobilenetv2_53_stage_lightNL_My':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_lightNL_My(num_classes, last_stride=last_stride, pre_train=self.backbone_pretrain, nl_c=cfg.MODEL.NL_C)

        elif model_name == 'mobilenetv2_53_stage_lightNL_linear':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_stage_lightNL_linear(num_classes, last_stride=last_stride,
                                                        pre_train=self.backbone_pretrain, nl_c=cfg.MODEL.NL_C)

        elif model_name == 'mobilenetv2_53_two_step_finetune':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_53_two_step_finetune(num_classes, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH, VCH_nls=cfg.DNL.VCH_nls)

        elif model_name == 'mobilenetv2_200_two_step_finetune':
            self.in_planes = cfg.MODEL.FEAT_DIM
            self.base = mobilenetv2_200_two_step_finetune(num_classes, sp_NL_opt=cfg.MODEL.DOUBLE_SP, ch_NL_opt=cfg.MODEL.DOUBLE_CH, VCH_nls=cfg.DNL.VCH_nls)

        elif model_name == 'stage_resnet50':
            self.in_planes = 2048
            self.base = Stage_ResNet50()

        elif model_name == 'dmasking_l3':
            ori_model = fbnet("dmasking_l3", pretrained=True)
            self.base = ori_model.backbone
            self.in_planes = 1984

        elif model_name == 'dmasking_f4':
            ori_model = fbnet("dmasking_f4", pretrained=True)
            self.base = ori_model.backbone
            self.in_planes = 1984

        elif model_name == 'dmasking_l2_hs':
            ori_model = fbnet("dmasking_l2_hs", pretrained=True)
            self.base = ori_model.backbone
            self.in_planes = 1984

        elif model_name == 'FBNetV2_3x_dnl_supernet':
            self.base = FBNetV2_3x_dnl_supernet()
            self.in_planes = 1984


        # if pretrain_choice == 'imagenet':
        #     pretrain_imgN = True
        #     if model_name == 'mobilenetv2' or 'mobilenetv2_1dot4':
        #         pretrain_imgN = False
        #     if model_name == 'mobilenetv3_large':
        #         pretrain_imgN = False
        #     if model_name == 'mobilenetv2_large':
        #         pretrain_imgN = False
        #     if model_name == 'hamobile':
        #         pretrain_imgN = False
        #     if model_name == 'mobilenetv2_modified':
        #         pretrain_imgN = False
        #     if model_name == 'attn_mobilenet':
        #         pretrain_imgN = False
        #     if pretrain_imgN:
        #         self.base.load_param(model_path)
        #         print('Loading pretrained ImageNet model......')

        if self.loss_branch == 1:
            self.gap = nn.AdaptiveAvgPool2d(1)
            # self.gap = nn.AdaptiveMaxPool2d(1)
            self.num_classes = num_classes
            self.neck = neck
            self.neck_feat = neck_feat

            if self.neck == 'no':
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
                # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                # self.classifier.apply(weights_init_classifier)  # new add by luo

            elif self.neck == 'bnneck':
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)  # no shift
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

                self.bottleneck.apply(weights_init_kaiming)
                self.classifier.apply(weights_init_classifier)

        if model_name == 'multiW_compressed_mobilenetv2_53':

            param_dict = torch.load('/home/ywan1053/reid-strong-baseline-master/log/market1501/mobilenetv2_deeper/mobilenetv2_53_lastS_1/mobilenetv2_53_model_680.pth')
            for k, v in param_dict.state_dict().items():
                if 'base' in k:
                    continue
                if 'classifier' or 'bottleneck' in k:
                    self.state_dict()[k].copy_(param_dict.state_dict()[k])

        if model_name == 'mobilenetv2_300_compress_S3':
            param_dict = torch.load(
                '/home/ywan1053/reid-strong-baseline-master/log/market1501/large_mobilenetv2/mobilenetv2_300_w1_longer/mobilenetv2_300_model_1000.pth')
            for k, v in param_dict.state_dict().items():
                if 'base' in k:
                    continue
                if 'classifier' or 'bottleneck' in k:
                    self.state_dict()[k].copy_(param_dict.state_dict()[k])

    def forward(self, x):
        if self.SEARCH_FBNETV2:
            global_feat = self.gap(self.base(x))

            global_feat = global_feat.view(global_feat.shape[0], -1)

            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)  # normalize for angular softmax

            if self.training:
                cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    return feat
                else:
                    # print("Test with feature before BN")
                    return global_feat

        if self.loss_branch == 1:

            global_feat = self.gap(self.base(x))

            global_feat = global_feat.view(global_feat.shape[0], -1)

            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)  # normalize for angular softmax

            if self.training:
                cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    return feat
                else:
                    # print("Test with feature before BN")
                    return global_feat
        else:
            return self.base(x)


        # global_feat = self.gap(self.base(x))
        # global_feat = global_feat.view(global_feat.shape[0], -1)
        #
        # if self.neck == 'no':
        #     feat = global_feat
        # elif self.neck == 'bnneck':
        #     feat = self.bottleneck(global_feat)  # normalize for angular softmax
        #
        # if self.training:
        #     cls_score = self.classifier(feat)
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            if 'classifier' in k:
                # print(i[0])
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])


    def load_param_withfc(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            self.state_dict()[k].copy_(param_dict.state_dict()[k])

    def load_trained(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            self.state_dict()[k].copy_(param_dict.state_dict()[k])


class Search_Base(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Search_Base, self).__init__()

        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        neck = cfg.MODEL.NECK
        neck_feat = cfg.TEST.NECK_FEAT
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        feat_dim = cfg.MODEL.FEAT_DIM
        self.backbone_pretrain = cfg.MODEL.BACKBONE_PRETRAIN
        self.loss_branch = cfg.MODEL.LOSS_BRANCH
        self.SEARCH_FBNETV2 = cfg.SOLVER.SEARCH_FBNETV2

        if model_name == 'FBNetV2_3x_dnl_supernet':
            self.base = FBNetV2_3x_dnl_supernet()
            self.in_planes = 1984

        if self.loss_branch == 1:
            self.gap = nn.AdaptiveAvgPool2d(1)
            # self.gap = nn.AdaptiveMaxPool2d(1)
            self.num_classes = num_classes
            self.neck = neck
            self.neck_feat = neck_feat

            if self.neck == 'no':
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
                # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                # self.classifier.apply(weights_init_classifier)  # new add by luo

            elif self.neck == 'bnneck':
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)  # no shift
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

                self.bottleneck.apply(weights_init_kaiming)
                self.classifier.apply(weights_init_classifier)

    def forward(self, x, temperature):
        x, cost_latency = self.base(x, temperature)
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, cost_latency  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat
