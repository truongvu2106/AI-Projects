# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### Training new model
```
python train.py flowers --save_dir densenet121_checkpoint.pth --gpu --hidden_units 2048 512 --arch densenet121 --epochs 5
python train.py flowers --save_dir vgg16_checkpoint.pth --gpu --hidden_units 2048 512 --arch vgg16 --epochs 5
```

### Continue training from trained model
```
python train.py flowers --checkpoint densenet121_checkpoint.pth --save_dir densenet121_checkpoint.pth --gpu --epochs 5
python train.py flowers --checkpoint vgg16_checkpoint.pth --save_dir vgg16_checkpoint.pth --gpu --epochs 5
```

### Predict
```
python predict.py flowers/test/102/image_08004.jpg densenet121_checkpoint.pth --gpu --top_k 5
python predict.py flowers/test/102/image_08004.jpg vgg16_checkpoint.pth --gpu --top_k 5
```
