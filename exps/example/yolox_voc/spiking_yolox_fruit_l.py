# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from spikingjelly.activation_based.surrogate import Sigmoid

import torch
import torch.nn as nn


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 1.0
        self.width = 1.0
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.T = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.spike_fn = Sigmoid(alpha=2.0)
        self.eval_interval = 1

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        import torch.nn as nn
        from yolox.models import YOLOPAFPN, SpikingYOLOXHead, SpikingYOLOX
        from yolox.utils.utils_snn import convert_to_spiking
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            backbone = convert_to_spiking(backbone, self.spike_fn)
            head = SpikingYOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                                    spike_fn=self.spike_fn, use_full_spike=True)
            self.model = SpikingYOLOX(backbone, head, T=self.T)
            # head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            # self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import FruitDetection, TrainTransform

        return FruitDetection(
            data_dir=os.path.join(get_yolox_datadir(), "fruit"),
            image_sets=[('2012', 'train')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import FruitDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return FruitDetection(
            data_dir=os.path.join(get_yolox_datadir(), "fruit"),
            image_sets=[('2012', 'val')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )

    # def get_optimizer(self, batch_size):
    #     if "optimizer" not in self.__dict__:
    #         if self.warmup_epochs > 0:
    #             lr = self.warmup_lr
    #         else:
    #             lr = self.basic_lr_per_img * batch_size
    #
    #         pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    #
    #         for k, v in self.model.named_modules():
    #             if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
    #                 pg2.append(v.bias)  # biases
    #             if isinstance(v, nn.BatchNorm2d) or "bn" in k:
    #                 pg0.append(v.weight)  # no decay
    #             elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
    #                 pg1.append(v.weight)  # apply decay
    #
    #         optimizer = torch.optim.AdamW(
    #             pg0, lr=lr, amsgrad=True
    #         )
    #         optimizer.add_param_group(
    #             {"params": pg1, "weight_decay": self.weight_decay}
    #         )  # add pg1 with weight_decay
    #         optimizer.add_param_group({"params": pg2})
    #         self.optimizer = optimizer
