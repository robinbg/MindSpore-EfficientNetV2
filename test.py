import argparse
import mindspore.nn as nn
from mindspore import Model, load_checkpoint, load_param_into_net

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Accuracy
from network import effnetv2_s

from utils import create_dataset
from mindspore import context

context.set_context(device_target="Ascend")
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-model_path', required=True)

args = parser.parse_args()

net = effnetv2_s(num_classes=10)

ds_eval_path =  "./datasets/cifar-10-batches-bin/test/"
ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
model = Model(net, loss_fn=ls, optimizer=opt, metrics={"Accuracy": Accuracy()})

param_dict = load_checkpoint(args.model_path)
load_param_into_net(net, param_dict)
eval_dataset = create_dataset(ds_eval_path)
acc = model.eval(eval_dataset)

print("Accuracy on test dataset: ", acc)