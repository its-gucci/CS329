# CS329

To train a model use 
```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --sigma 0.12
```

To do prediction and certified accuracy, run notebook
```
pytorch-classification/certify.ipynb
```
