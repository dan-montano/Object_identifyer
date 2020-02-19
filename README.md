# SCR Deteccion de Objetos con Tensorflow 2.0
SCR es un algoritmo que utiliza redes neuronales convolucionales para detectar objetos <br> <br>

## Instrucciones

### Prerequisitos
Este proyecto esta escrito en Python 3.7 usando Tensorflow 2.0 (deep learning), NumPy, Pillow, OpenCV, seaborn, Flask, urllib (get images fron android app). Para intalar los prerequisitos ejecute el siguiente comando.

```
pip install -r requirements.txt
```

### Utilizando entrenamiento personalisado
<strong> Aprenda como entrenar pesos especificos de YOLOV3 aqui: https://www.youtube.com/watch?v=zJDUhGL26iU </strong>

Agrega el archivo de pesos a la carpeta de weights y tu archivo de clases.names al folder data/labels.

Cambia 'n_classes=80' en la linea 97 de load_weights.py a 'n_classes=<numero de clases en .names>'.

Cambia './weights/yolov3.weights' en la linea 107 de load_weights.py a './weights/<pesos personalizaos>'.

Cambia './data/labels/coco.names' en la linea 25 de SCR.py a './data/labels/<clases.names>'.
  
### Guarda los pesos en formato Tensorflow
Carga los pesos usando `load_weights.py` script. Esto convertira los pesos de YOLOV3 en un archivo de TensorFlow .ckpt!

```
python load_weights.py
```

## Corriendo el servicio
Puedes correr el servicio usando `SCR.py`. El script trabaja con 'streaming cam', obteniendo imagenes de una direccion IP especifica, en modo DEBUG. Dentro de local host.
### Uso
```
python SCR.py 
```

