import mindspore.nn as nn

from mindspore import context
import numpy as np
import matplotlib.pyplot as plt
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, Callback
from mindspore import load_checkpoint, load_param_into_net
import os
from mindspore import Model
from mindspore.nn import Accuracy
from network import effnetv2_s

from utils import create_dataset

import warnings



class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
        
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["Accuracy"])
            print(acc)

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    context.set_context(device_target="Ascend")
    net = effnetv2_s(num_classes=10)
    
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=ls, optimizer=opt, metrics={"Accuracy": Accuracy()})
# As for train, users could use model.train

    epoch_size = 50
    eval_per_epoch = 1
    ds_train_path = "./datasets/cifar-10-batches-bin/train/"
    ds_eval_path =  "./datasets/cifar-10-batches-bin/test/"
    model_path = "./models/ckpt/mindspore_vision_application/"
    os.system('rm -f {0}*.ckpt {0}*.meta {0}*.pb'.format(model_path))

    train_dataset = create_dataset(ds_train_path )
    eval_dataset = create_dataset(ds_eval_path)
    batch_num = train_dataset.get_dataset_size()//64
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 64, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="train_effnetv2_cifar10", directory=model_path, config=config_ck)
    loss_cb = LossMonitor(142)
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, eval_dataset, eval_per_epoch, epoch_per_eval)
    model.train(epoch_size, train_dataset, callbacks=[ckpoint_cb, loss_cb, eval_cb])