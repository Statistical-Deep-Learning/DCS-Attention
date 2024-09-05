
import mobile_cv.common.misc.registry as registry

from . import modeldef_utils as mdu
from .modeldef_utils import _ex, e1, e3, e4, e6

MODEL_ARCH = registry.Registry("cls_arch_factory")

MODEL_ARCH_DEFAULT = {
    "default": {
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
            # stage 2
            [("ir_k3", 24, 2, 2, e6)],
            # stage 3
            [("ir_k3", 32, 2, 3, e6)],
            # stage 4
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
            # stage 5
            [("ir_k3", 160, 2, 3, e6), ("ir_k3", 320, 1, 1, e6)],
            # stage 6
            [("conv_k1", 1280, 1, 1)],
        ]
    },
    "mnv3": {
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [("conv_k3_hs", 16, 2, 1)],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [["ir_k3", 24, 2, 1, e4], ["ir_k3", 24, 1, 1, e3]],
            # stage 3
            [["ir_k5_sehsig", 40, 2, 3, e3]],
            # stage 4
            [
                ["ir_k3_hs", 80, 2, 1, e6],
                ["ir_k3_hs", 80, 1, 1, _ex(2.5)],
                ["ir_k3_hs", 80, 1, 2, _ex(2.3)],
                ["ir_k3_sehsig_hs", 112, 1, 2, e6],
            ],
            # stage 5
            [["ir_k5_sehsig_hs", 160, 2, 3, e6]],
            # stage 6
            [["ir_pool_hs", 1280, 1, 1, e6]],
        ]
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DEFAULT)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_DEFAULT))
