from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# cam
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from subprocess import call

import argparse
import imutils
import time
import cv2
import random

import pyttsx3
import os, subprocess
import os.path
import re
import sys
import tarfile
import speak

import numpy as np
from six.moves import urllib
import tensorflow as tf

# button

import RPi.GPIO as GPIO
import time
import datetime

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)


FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/home/pi/tensorflow/tensorflow/models/image/imagenet/model_dir',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long



# cam

print("[INFO] cam sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()




### more
def run_image(sess, img_id, img_url, node_lookup):
  from six.moves import urllib
  from urllib2 import HTTPError
  try:
    image_data = urllib.request.urlopen(img_url, timeout=1.0).read()
  except HTTPError:
    return (img_id, img_url, None)
  except:
    return (img_id, img_url, None)
  scores = []
  softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
  predictions = sess.run(softmax_tensor,
                         {'DecodeJpeg/contents:0': image_data})
  predictions = np.squeeze(predictions)
  top_k = predictions.argsort()[-num_top_predictions:][::-1]
  scores = []
  for node_id in top_k:
    if node_id not in node_lookup:
      human_string = ''
    else:
      human_string = node_lookup[node_id]
    score = predictions[node_id]
    scores.append((human_string, score))
  return (img_id, img_url, scores)


### end more







class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(fn='/home/pi/Desktop/camlive.jpg'):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """

  print("Starting tf.Session()")
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    print("Session initialized.")
    node_lookup = NodeLookup() # Creates node ID --> English string lookup.
    print("done node lookup")
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    print("got tensor")
    os.system("/usr/bin/pico2wave -w test.wav 'ready to use' | mplayer test.wav")

    while 1:
          input_state = GPIO.input(18)
          if input_state == False:
            print('Button Pressed')
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            print('Captured %dx%d image' % ( frame.shape[1], frame.shape[0]) )
            cv2.imwrite(fn, frame)
            print("written file")
            if not tf.gfile.Exists(fn):
              tf.logging.fatal('File does not exist %s', fn)
            image_data = tf.gfile.FastGFile(fn, 'rb').read()
            print("read file")
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
            print("got predictions")
            node_id = top_k[0]
            print("-------")
            print(node_id)
            score = predictions[node_id]
            print(score)
            print("-------")
            human_string = None
#        if(score > 0.20):
            human_string = node_lookup.id_to_string(node_id)
            arr = human_string.split(",")
            if(arr[0]):
              print('%s (score = %.5f)' % (arr[0], score))
              engine=pyttsx3.init()
              engine.say(arr[0])
              engine.runAndWait()
            else:
              os.system("/usr/bin/pico2wave -w test.wav 'nothing' | mplayer test.wav")

            
            #word=arr[0]
            #speak.voice(word)

   #     else:
    #        os.system("/usr/bin/pico2wave -w test.wav 'nothing' | mplayer test.wav")

        # save results
           #epoch = datetime.datetime.utcfromtimestamp(0)
           # dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
           # os.system("cp "+fn+" results/"+dt+".jpg")            
           # f1=open("results/"+dt+".txt", 'w+')
            #f1.write(str(human_string))
            #f1.write(str(score))
            #f1.write(str(top_k))
            #f1.close()

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  create_graph()   # Creates graph from saved GraphDef.
  run_inference_on_image()

if __name__ == '__main__':
  tf.app.run()
