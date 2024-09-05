from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DNL = CN()
_C.DNL.START_SEARCH = 0
_C.DNL.VCH_nls = 2
_C.DNL.OPTIMIZER_TYPE = 'SGD'
_C.DNL.SEPARATE_SEARCH_SCHEDULER = False
_C.DNL.SEPARATE_SEARCH_STEPS = (30, 55)

_C.DNL.INIT_AFTER_SEARCH = False
_C.DNL.LEARN_LENGTH = False
_C.DNL.LEARN_STARTPOS = False
_C.DNL.LR_SCHEDULER = 'WarmupMultiStepLR'
_C.DNL.LENGTH_LR = 0.00003
_C.DNL.LENGTH_WEIGHT_DECAY = 0.0

_C.DNL.SEPARATE_TRAINING = False
_C.DNL.SEPARATE_TRAINING_ITER = False
_C.DNL.STARTPOS_LR = 0.00003
_C.DNL.STARTPOS_WEIGHT_DECAY = 0.0
_C.DNL.STEPS = (30, 55)
# warm up factor
_C.DNL.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.DNL.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.DNL.WARMUP_METHOD = "linear"
_C.DNL.GAMMA = 0.1
_C.DNL.WDB = False
_C.DNL.MAX_SEARCH_EPOCH = 100
_C.DNL.ITER_EPOCH_NUM = 10
_C.DNL.WEIGHT_FACTOR = 0.5






_C.MODEL = CN()

_C.MODEL.DNL = False
_C.MODEL.NL_C = 0.5
_C.MODEL.DOUBLE_CH = False
_C.MODEL.DOUBLE_SP = False
_C.MODEL.SUPER_MODEL_INIT = False
_C.MODEL.SUPER_MODEL_DIR = '/data/yyang409/yancheng/log/market1501/compress_300_w1/S3_bLR_00035_posLR_0001/mobilenetv2_300_compress_S3_model_100.pth'
_C.MODEL.SUPER_MODEL_CENTER = '/data/yyang409/yancheng/log/market1501/compress_300_w1/S3_bLR_00035_posLR_0001/mobilenetv2_300_compress_S3_center_param_100.pth'
_C.MODEL.CONTINUE = False
_C.MODEL.CONTINUE_PATH = ''
_C.MODEL.CONTINUE_CENTER_PATH = ''
_C.MODEL.LOSS_BRANCH = 1
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.BACKBONE_PRETRAIN = False
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
_C.MODEL.FEAT_DIM = 1280
_C.MODEL.COMPUTE_MODEL_COMPLEXITY = False
_C.MODEL.WIDTH_MULT = 1.0

_C.MODEL.WEIGHTS_OPEN = True
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.FREEZE_POS = 1000

# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.POSITION_LR_FACTOR = 1.0
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1


# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.
_C.SOLVER.WEIGHT_DECAY_POSITION = 0.

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

_C.SOLVER.TWO_STEP = False
_C.SOLVER.TWO_STEP_MAX = 20

_C.SOLVER.SEARCH_FBNETV2 = False


# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

_C.TEST.IF_WDB = False
_C.TEST.WDB_NAME = 'run_name'
_C.TEST.WDB_PROJECT = 'project_name'
_C.TEST.WDB_PRINT_POSITION = False
_C.TEST.WDB_PRINT_ITER = False
# ---------------------------------------------------------------------------- #

_C.GRADCAM = CN()
_C.GRADCAM.TARGET_LAYER_NAMES = "conv"
_C.GRADCAM.IMAGE_PATH = '/data/yyang409/yancheng/data/market1501/bounding_box_train/0098_c1s1_015651_03.jpg'
_C.GRADCAM.MODEL_WEIGHT_PATH = ''
_C.GRADCAM.IMAGE_SIZE = (256, 128)
_C.GRADCAM.SET_TARGET_INDEX = False
_C.GRADCAM.USE_GT_LABEL = True
_C.GRADCAM.TARGET_CLASS_INDEX = 0
_C.GRADCAM.OUT_PATH = '/home/ywan1053/reid-strong-baseline-master/grad_cam/img/'

# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
