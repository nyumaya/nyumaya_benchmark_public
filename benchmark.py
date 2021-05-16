import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import glob
from config import *

sys.path.insert(0, nyumaya_basepath + '/python/src/')

from libnyumaya import AudioRecognition, FeatureExtractor
from pydub import AudioSegment
from pydub import effects
from os import walk
from os.path import splitext
from os.path import join

from random import randint, seed
from auto_platform import default_libpath
from utils import *
from multiprocessing import Process,Manager
from multiprocessing.managers import BaseManager
from benchmarkResult import *

nyumaya_lib_path = os.path.join("./nyumaya_audio_recognition/python/src/",default_libpath)

seed(1234)

def include_random_folder(path):
	file_list=[]
	for root, _, files in walk(path):
		for f in files:
			extension = splitext(f)[1].lower()
			if extension in extension_list:
				file_list.append(join(root, f))
	return file_list

def get_random_file(file_list):
	filelen = len(file_list)
	index  = randint(0,filelen-1)
	return file_list[index]

def split_sequence(a,seg_length):
	return [a[x:x+seg_length] for x in range(0,len(a),seg_length)]


def decode_szenario(serialized_example):

	features = tf.io.parse_single_example(
		serialized_example,
		features={
		'meldata': tf.io.FixedLenFeature([], tf.string),
		'utf_text':  tf.io.FixedLenFeature([], tf.string),
	})

	meldata = tf.io.decode_raw(features['meldata'], tf.uint8)
	text = tf.io.decode_raw(features['utf_text'], tf.uint8)

	return meldata,text

def include_good_folder(path,keyword):
	file_list = []
	for root, _, files in walk(path):
		for f in files:
			extension = splitext(f)[1].lower()
			if extension in extension_list:
				file_list.append(os.path.join(root,f))

	return file_list

def run_good(keyword,add_noise,version,noiseIdx,sensIdx,resultInst):

	sensitivity = sensitivitys[sensIdx]
	snr = noise_levels[noiseIdx-1]

	records_path = os.path.join(keyword_folder,keyword)
	filelist = include_good_folder(records_path,keyword)

	detector = AudioRecognition(nyumaya_lib_path)
	extractor = FeatureExtractor(nyumaya_lib_path)

	modelpath = os.path.join(nyumaya_basepath,"models/Hotword/{}_v{}.premium".format(keyword,version))
	print("Model path: {}".format(modelpath))
	detector.addModel(modelpath,sensitivity)

	samplenumber = 0
	detectionnumber = 0

	noise_list = []

	if(add_noise):
		for noise_folder in noise_folder_list:
			if(os.path.exists(noise_folder)):
				noise_list += include_random_folder(noise_folder)

	bufsize = detector.getInputDataSize() * 2
	print(bufsize)
	for f in filelist:

		wavdata,_ = load_audio_file(f)
		if(not wavdata):
			continue

		silence = AudioSegment.silent(duration=1000)
		wavdata = silence + wavdata + silence

		#FIXME: Find a better way to normalize the noise
		# level relative to the speech level
		#wavdata = effects.normalize(wavdata)

		if(add_noise):
			bg_noise_file = get_random_file(noise_list)
			noise,_ = load_audio_file(bg_noise_file)
			if(not noise):
				print("Couldn't load: " + bg_noise_file)
				continue

			noise = effects.normalize(noise)
			noise = noise.apply_gain(-snr)

			wavdata = wavdata.overlay(noise, gain_during_overlay=0)

		wavdata = wavdata.get_array_of_samples().tobytes()
		splitdata = split_sequence(wavdata,bufsize)

		has_detected_something = False

		for frame in splitdata:
			features = extractor.signalToMel(frame)
			prediction = detector.runDetection(features)

			if(prediction != 0):
				has_detected_something = True

		if(has_detected_something == True):
			detectionnumber += 1

		samplenumber += 1
	accuracy = detectionnumber / samplenumber

	print("{:.3f} @ {}".format(accuracy,sensitivity))
	resultInst.setAccuracy(noiseIdx,sensIdx,accuracy)



def run_szenario(szenario,sensitivity,keyword,version,szenIdx,sensIdx,resultInst):

	detector = AudioRecognition(nyumaya_lib_path)
	modelpath = os.path.join(nyumaya_basepath,"models/Hotword/{}_v{}.premium".format(keyword,version))
	print("Model path: {}".format(modelpath))
	if(not os.path.exists(modelpath)):
		print("Failed: {}".format(modelpath))
		return -1

	detector.addModel(modelpath,sensitivity)

	dataset = glob.glob(os.path.join(szenario_basepath,"{}.tfrecords".format(szenario)))
	dataset = tf.data.TFRecordDataset([dataset])
	false_alarms_per_hour = 0.0
	run_seconds = 0
	run_frames = 0
	false_alarms = 0
	bufsize = detector.getInputDataSize() * 2

	for i,elem in enumerate(dataset):

		mel,text = decode_szenario(elem)

		run_frames += (mel.shape[0] / 40)

		framelen_v2 = 640
		framelen_v1 = 800
		framelen = framelen_v2  #FIXME Hardcoded

		frames = split_sequence(mel,framelen)
		for frame in frames:
			prediction = detector.runDetection(frame)
			if(prediction != 0):
				try:
					text = text.numpy().tobytes().decode('utf-8')
					print("False Alarm:{} {}".format(prediction,text))
				except:
					print("Failed to decode text")

				false_alarms += 1


	print("False Alarms: {}".format(false_alarms))
	
	#The feature extractor creates 100 Frames per second
	run_seconds = run_frames / 100
	run_hours = run_seconds / 3600
	false_alarms_per_hour = false_alarms / run_hours
	print("Run Hours: {}".format(run_hours))
	print("False Alarms per hour {} @ {}".format(false_alarms_per_hour,sensitivity))

	resultInst.setFalseActivations(szenIdx,sensIdx,false_alarms_per_hour)

version=None

processes = []

BaseManager.register('benchmarkResult', benchmarkResult)
manager = BaseManager()
manager.start()
result = manager.benchmarkResult()

keyword = sys.argv[1]
keyword = keyword.lower()
version = sys.argv[2]
outfile = '{}_v{}.txt'.format(keyword,version)
outfile = os.path.join(result_folder,outfile)


#Clean Accuracy
for sensIdx,sens in enumerate(sensitivitys):
	p = Process(target=run_good, args=(keyword,False,version,0,sensIdx,result))
	processes.append(p)

#Noisy Accuracy
for noiseIdx,level in enumerate(noise_levels):
	for sensIdx,sens in enumerate(sensitivitys):
		p = Process(target=run_good, args=(keyword,True,version,noiseIdx+1,sensIdx,result))
		processes.append(p)

for szenIdx,szen in enumerate(szenarios):
	for sensIdx,sens in enumerate(sensitivitys):
		p = Process(target=run_szenario, args=(szen,sens,keyword,version,szenIdx,sensIdx,result))
		processes.append(p)

for pr in processes:
	pr.start()

for pr in processes:
	pr.join()


result.write(outfile)







