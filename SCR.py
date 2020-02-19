from flask import Flask, stream_with_context, Response, json, request
import tensorflow as tf
import cv2
import urllib
import numpy as np
from yolo_v3 import Yolo_v3
from utils import load_class_names, clasifier
import time

app = Flask(__name__)
app.config['DEBUG'] = True

tf.compat.v1.disable_eager_execution() #Remove to TF 2.0

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20
iou_threshold = 0.5
confidence_threshold = 0.5

@app.route('/')
def stream():
    def main():
        class_names = load_class_names(_CLASS_NAMES_FILE)
        n_classes = len(class_names)

        model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                        max_output_size=_MAX_OUTPUT_SIZE,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)

        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')
            url='http://192.168.1.111:8080/shot.jpg'
            image = urllib.request.urlopen(url)
            img = np.array(bytearray(image.read()), dtype = np.uint8)
            frame = cv2.imdecode(img, -1)
            frame_size = (850, 500)


            try:
                while True:
                                        
                    image = urllib.request.urlopen(url)
                    img = np.array(bytearray(image.read()), dtype = np.uint8)
                    frame = cv2.imdecode(img, -1)
                    frame = cv2.resize(frame, dsize=(850, 500),
                                                interpolation=cv2.INTER_NEAREST)

                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                                interpolation=cv2.INTER_NEAREST)

                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    clasification = clasifier(detection_result, class_names)

                    print ( json.dumps(clasification) )
                    time.sleep(0.05) # Time Slepeer
                    yield (json.dumps( clasification ) + '\n' )
        
            finally:
                print('FaceID end')

    return Response( stream_with_context(main()), mimetype='application/json' )        

if __name__ == '__main__':
    app.run(host='localhost')