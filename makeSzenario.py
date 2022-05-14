import numpy as np
import sys
import io
import os
#Don't let tensorflow hog the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from config import *
from utils import *

import tensorflow as tf

sys.path.insert(0, nyumaya_basepath + '/python/src/')

from libnyumaya import FeatureExtractor
from auto_platform import default_libpath
from random import shuffle
from os.path import splitext

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

nyumaya_libpath = os.path.join(nyumaya_basepath, default_libpath)
lib_extractor = FeatureExtractor()

def write_example_to_record(meldata,text,writer):
	example = tf.train.Example(
		features=tf.train.Features(
			feature={
				'meldata': bytes_feature(meldata.tobytes()),
				'utf_text': bytes_feature(text),
			}
		)
	)
	writer.write(example.SerializeToString())

def make_librispeech(in_dir):

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
			write_example_to_record(meldata,text,writer)

# Take all audio files from a folder and
# write them to a scenario. Watch out, pydub
# might complain about long mp3 files
def make_folder(in_dir,record_name):
	promptlist = []
	for root, dirs, files in os.walk(in_dir):
			for f in files:
				extension = splitext(f)[1].lower()
				if(not (extension in extension_list)):
					continue
				pr = os.path.join(root,f)
				promptlist.append(pr)
				
	shuffle(promptlist)

	record_path = os.path.join(szenario_basepath,record_name)
	with tf.io.TFRecordWriter(record_path) as writer:
		for line in promptlist:
			print(line)
			sound,duration = load_audio_file(line)
			print("Duration: {}".format(duration))

			#Cut into 20 second slices
			slices = sound[0:-1:20*1000]
			for index,s in enumerate(slices):
				wavdata = s.get_array_of_samples()
				wavdata = np.asarray(wavdata, dtype = np.int16)
				meldata = lib_extractor.signalToMel(wavdata.tobytes())
				meldata = np.reshape(meldata, (-1,40))
				text = "".encode('utf-8')
				write_example_to_record(meldata,text,writer)

#Uncomment to run
#make_librispeech("./LibriTTS/test-clean/")


#Uncomment to run
#make_folder("./myfolder","ambient_test_v1.0.tfrecords")





