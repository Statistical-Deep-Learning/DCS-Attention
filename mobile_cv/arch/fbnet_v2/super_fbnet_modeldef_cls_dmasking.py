
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e6

BASIC_ARGS = {}

IRF_CFG = {"less_se_channels": False}

MODEL_ARCH_DMASKING_NET = {
    "da_fbnetv2_3x": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ### 1
            # stage 1
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],  ### 2
            # stage 2
            [
                ["ir_k3_hs", 24, 2, 1, _ex(6), IRF_CFG],  ### 3 4
                ["ir_k3_hs", 24, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k3_hs", 32, 2, 3, _ex(6), IRF_CFG],  ### 5 6
                ["ir_k3_hs", 32, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k3_hs", 112, 2, 3, _ex(6), IRF_CFG],  ### 7 8 9
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 112, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_hs", 216, 2, 3, _ex(6), IRF_CFG],  ### 10 11
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],  ### 12 13
        ],
    },
    "da_fbnetv2_3x_supernet": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ### 1
            # stage 1
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],  ### 2
            # stage 2
            [
                ["ir_k3_hs", 28, 2, 1, _ex(6), IRF_CFG],  ### 3 4
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k3_hs", 40, 2, 3, _ex(6), IRF_CFG],  ### 5 6
                ["ir_k3_hs", 40, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 40, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 40, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k3_hs", 128, 2, 3, _ex(6), IRF_CFG],  ### 7 8 9
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_hs", 216, 2, 3, _ex(6), IRF_CFG],  ### 10 11
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 3, _ex(6), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],  ### 12 13
        ],
    },
    "da_fbnetv2_6x": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ### 1
            # stage 1
            [["ir_k3", 16, 1, 1, e1, IRF_CFG]],  ### 2
            # stage 2
            [
                ["ir_k3_hs", 24, 2, 1, _ex(6), IRF_CFG],  ### 3 4
                ["ir_k3_hs", 24, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k3_hs", 32, 2, 6, _ex(6), IRF_CFG],  ### 5 6
                ["ir_k3_hs", 32, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k3_hs", 72, 2, 6, _ex(6), IRF_CFG],  ### 7 8 9
                ["ir_k3_hs", 72, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 72, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 72, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 72, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 72, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 120, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_hs", 216, 2, 6, _ex(6), IRF_CFG],  ### 10 11
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],  ### 12 13
        ],
    },
    "da_fbnetv2_6x_supernet": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ### 1
            # stage 1
            [["ir_k3", 16, 1, 1, e1, IRF_CFG]],  ### 2
            # stage 2
            [
                ["ir_k3_hs", 28, 2, 1, _ex(6), IRF_CFG],  ### 3 4
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k3_hs", 40, 2, 6, _ex(6), IRF_CFG],  ### 5 6
                ["ir_k3_hs", 40, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 40, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 40, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k3_hs", 96, 2, 6, _ex(6), IRF_CFG],  ### 7 8 9
                ["ir_k3_hs", 96, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_hs", 216, 2, 6, _ex(6), IRF_CFG],  ### 10 11
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 6, _ex(6), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],  ### 12 13
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET)
