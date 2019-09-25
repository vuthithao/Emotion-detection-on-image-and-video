from io import BytesIO
import base64
try:
    from PIL import Image
except:
    from pil import Image

from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer

import numpy as np
import cv2
import time

import os
import scipy
import tensorflow as tf
import align.detect_face

from keras.preprocessing import image
from keras.models import model_from_json

app = Flask(__name__)


# facenet
image_size=160
margin= 44
gpu_memory_fraction=1.0

def load_and_align_data(image_path, image_size,margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img = scipy.misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if (len(bounding_boxes)==0):
        bb=0
        have_face = 0
    else:
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2 - bb[0], img_size[1])
        bb[3] = np.minimum(det[3]+margin/2 - bb[1], img_size[0])
        have_face = 1
    return bb,have_face

model = model_from_json(open("models/model_4layer_2_2_pool.json", "r").read())
model.load_weights('models/model_4layer_2_2_pool.h5') #load weights
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')



def readb64(base64_string, rgb=True):
    """
    Đọc ảnh từ dạng base64 -> numpy array\n

    Input
    -------
    **base64_string**: Ảnh ở dạng base64\n
    **rgb**: True nếu ảnh là dạng RBG (đủ 3 channel), False nếu là ảnh xám (1 channel)

    Output:
    -------
    Ảnh dạng numpy array
    """
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    if rgb:
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2GRAY)

def emotion(data_type):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        input = dataDict.get(data_type, None)

    start = time.time()
    img = readb64(input, rgb=True)
    img_path = "tmp.jpg"
    cv2.imwrite(img_path, img)
    # img = cv2.imread(img_path)
    detect_face, have_face = load_and_align_data(img_path, image_size, margin, gpu_memory_fraction)
    if (have_face != 0):
        detect_face = np.reshape(detect_face, (-1, 4))
        result = []
        for (x, y, w, h) in detect_face:
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48
            cv2.rectangle(img, (x, y), (x + w, y + h), (64, 64, 64), 2)  # highlight detected face

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            # -----------------------------

            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])
            print(max_index)

            face = {}
            face['position'] = [float(x), float(y), float(w), float(h)]
            a = []
            for i in range(len(predictions[0])):
                a.append({'emotions': emotions[i], 'prob': round(predictions[0][i] * 100, 2), })
            face['emotion'] = a
            result.append(face)

    print(result)
    end = time.time() - start
    response = jsonify({"result": result, "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200


@app.route('/emotion', methods=['POST'])
def emotion_():
    return emotion(data_type="img")


if __name__ == "__main__":
    http_server = WSGIServer(('', 4000), app)
    http_server.serve_forever()

