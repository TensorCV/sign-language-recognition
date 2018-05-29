import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
from gtts import gTTS
import pyglet

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/output_labels.txt")]

with tf.gfile.FastGFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    textInput = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            x1, y1, x2, y2 = 200, 200, 400, 400
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            
            if i == 4:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        textInput += ' '
                    elif res == 'del':
                        textInput = textInput[:-1]
                    else:
                        textInput += res
                        # tts = gTTS(text = res,lang = 'en')
                        # tts.save("test.mp3")
                        # music = pyglet.media.load("test.mp3")
                        # music.play()
                        # time.sleep(music.duration)
                        # os.remove("test.mp3")
                    consecutive = 0
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (500,400), cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 4)
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.imshow("Recognizer ", img)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            cv2.putText(img_sequence, '%s' % (textInput.upper()), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            cv2.imshow('Alphebet', img_sequence)
            
            if a == 27: # when `esc` is pressed
                break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
