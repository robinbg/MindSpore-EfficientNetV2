wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz
mkdir -p datasets
tar -xzf cifar-10-binary.tar.gz -C datasets
mkdir -p datasets/cifar-10-batches-bin/train datasets/cifar-10-batches-bin/test
mv -f datasets/cifar-10-batches-bin/test_batch.bin datasets/cifar-10-batches-bin/batches.meta.txt datasets/cifar-10-batches-bin/test
mv -f datasets/cifar-10-batches-bin/data_batch*.bin datasets/cifar-10-batches-bin/batches.meta.txt datasets/cifar-10-batches-bin/train
tree ./datasets/cifar-10-batches-bin
