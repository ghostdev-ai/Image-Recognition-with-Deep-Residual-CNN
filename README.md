# Image-Recognition-with-Deep-Residual-CNN
PyTorch implementation of Deep Residual CNNs described in the paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). This project demonstrates how one might implement a Deep Residual CNN with PyTorch.

## Deep Residual Learning Overview
Deep Residual Learning is a CNN-based approach for image classification that leverages residual blocks, or skip connections, to build deeper and deeper models that don't  succumb to the the vanishing gradient problem. 

![image](https://user-images.githubusercontent.com/51813928/218296774-9b6bc569-f8e3-4f05-86c8-07e0fd835cfe.png)


## Project Structure
* `data_setup.py` - prepares and downloads the data.
* `engine.py` - contains various functionality for training the model.
* `model.py` or `#_layer_resnet_model.py` - creates a PyTorch ResNet model. 
* `train.py` - leverages all utility files to train a PyTorch ResNet model. 
* `utils.py` - useful utility functions for training and saving models.  

## Usage
```
python3 train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

## License
This project is licensed under the terms of the [MIT license](https://choosealicense.com/licenses/mit/). 
