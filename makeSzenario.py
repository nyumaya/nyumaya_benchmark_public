import numpy as np
import sys
import io
import os
#Don't let tensorflow hog the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from config import *
from utils import load_audio_file

import tensorflow as tf

sys.path.insert(0, nyumaya_basepath + '/python/src/')

from libnyumaya import FeatureExtractor
from auto_platform import default_libpath
from random import shuffle


def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

nyumaya_libpath = os.path.join("./nyumaya_audio_recognition/python/src/",default_libpath)
lib_extractor = FeatureExtractor(nyumaya_libpath,nfft=512,melcount=40,
	sample_rate=samplerate,lowerf=20,upperf=8000,window_len=0.03,shift=0.01)


def make_librispeech(in_dir = None):

	promptlist = []

	for root, dirs, files in os.walk(in_dir):
		for f in files:
			if "normalized.txt" in f:
				pr = os.path.join(root,f)
				with io.open(pr,'r') as prfile:
					text = prfile.read()
					filename  = pr.replace(".normalized.txt", ".wav",1)
					promptlist.append(filename +"|" +text)
					print("Appending: {}".format(filename))

	shuffle(promptlist)

	record_name = os.path.join(szenario_basepath,"libri_test_v1.0.tfrecords")
	with tf.io.TFRecordWriter(record_name) as writer:
		for line in promptlist:
			fpath,text = line.strip().split("|")
			text = text.encode('utf-8')
			wavdata,_ = load_audio_file(fpath)
			wavdata = wavdata.get_array_of_samples()
			wavdata = np.asarray(wavdata, dtype = np.int16)
			meldata = lib_extractor.signalToMel(wavdata.tobytes())
			meldata = np.reshape(meldata, (-1,40))
			example = tf.train.Example(
				features=tf.train.Features(
					feature={
						'meldata': bytes_feature(meldata.tobytes()),
						'utf_text': bytes_feature(text),
					}
				)
			)
			writer.write(example.SerializeToString())

make_librispeech("./LibriTTS/test-clean/")



