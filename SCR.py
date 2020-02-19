# Uso
# python SCR.py
from flask import Flask, stream_with_context, Response, json, request
import tensorflow as tf
import cv2
import urllib
import numpy as np
from yolo_v3 import Yolo_v3
from utils import load_class_names, clasifier
import time

# Inicializando servidor en Flask
app = Flask(__name__)
app.config['DEBUG'] = True

tf.compat.v1.disable_eager_execution() #Remove to TF 2.0

#Parametros
_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20 #Numero maximo de cajas por clase
iou_threshold = 0.5   # Umbral de superposicion de cajas
confidence_threshold = 0.5 # Umbral de confidencia

@app.route('/')
def stream():
    def main():
        # Cargando el archivo de etiquetas
        class_names = load_class_names(_CLASS_NAMES_FILE)
        n_classes = len(class_names)

        #Llamada al modelo Yolo-v3
        model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                        max_output_size=_MAX_OUTPUT_SIZE,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)

        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        #iniciando secion si cuenta con tensorflow 1
        with tf.compat.v1.Session() as sess:
                
            saver.restore(sess, './weights/model.ckpt') # Pesos del modelo
            url='http://192.168.1.111:8080/shot.jpg' # Direccion de streaming
            # Convercion de las imagenes de streaming
            image = urllib.request.urlopen(url)
            img = np.array(bytearray(image.read()), dtype = np.uint8)
            frame = cv2.imdecode(img, -1)
            frame_size = (850, 500)


            try:
                while True:

                    # Obtencion de las imagenes                  
                    image = urllib.request.urlopen(url)
                    img = np.array(bytearray(image.read()), dtype = np.uint8)
                    frame = cv2.imdecode(img, -1)
                    frame = cv2.resize(frame, dsize=(850, 500),
                                                interpolation=cv2.INTER_NEAREST)

                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                                interpolation=cv2.INTER_NEAREST)

                    # Activamos yolo
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})
                    # Clasificacion del imagenes
                    clasification = clasifier(detection_result, class_names)

                    # Obtencion de los datos Diccionario
                    print ( json.dumps(clasification) )
                    time.sleep(0.05) # Time Slepeer
                    yield (json.dumps( clasification ) + '\n' )
        
            finally:
                print('FaceID end')

    return Response( stream_with_context(main()), mimetype='application/json' )        

if __name__ == '__main__':
    app.run(host='localhost') #direccion de posteo de datos