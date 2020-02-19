# SCR Object Detection with Tensorflow 2.0
SCR is an algorithm that uses deep convolutional neural networks to detect objects. <br> <br>

## Getting started

### Prerequisites
This project is written in Python 3.7 using Tensorflow 2.0 (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision), seaborn (visualization) packages, Flask (server), urllib (get images fron android app).

```
pip install -r requirements.txt
```

### Downloading official pretrained weights
For Linux: Let's download official weights pretrained on COCO dataset. 

```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and adding them to the weights folder.

### Using Custom trained weights
<strong> Learn How To Train Custom YOLOV3 Weights Here: https://www.youtube.com/watch?v=zJDUhGL26iU </strong>

Add your custom weights file to weights folder and your custom .names file into data/labels folder.

Change 'n_classes=80' on line 97 of load_weights.py to 'n_classes=<number of classes in .names file>'.

Change './weights/yolov3.weights' on line 107 of load_weights.py to './weights/<custom weights file>'.

Change './data/labels/coco.names' on line 25 of detection.py to './data/labels/<custom names files>'.
  
### Save the weights in Tensorflow format
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!

```
python load_weights.py
```

## Running the model
You can run the model using `SCR.py` script. The script works 'streaming cam'.
### Usage
```
python SCR.py 
```

## To-Do List
* Finish migration to full TF 2.0 (remove tf.compat.v1)
* Model training
* Tiny Yolo Configuration

## Acknowledgments
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
* [ResNet official implementation](https://github.com/tensorflow/models/tree/master/official/resnet)
* [DeviceHive video analysis repo](https://github.com/devicehive/devicehive-video-analysis)
* [A Street Walk in Shinjuku, Tokyo, Japan](https://www.youtube.com/watch?v=kZ7caIK4RXI)
