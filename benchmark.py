import sys
import os

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
from multiprocessing import Process,Queue
from benchmarkResult import *

nyumaya_dir =  os.path.join(nyumaya_basepath,"python/src/")
nyumaya_lib_path = os.path.join(nyumaya_dir,default_libpath)

print("Nyumaya Lib Path: {}".format(nyumaya_lib_path))
resultQueue = Queue()
benchmarkResultQueue = Queue()

def usage():
	print("benchmark.py <keyword> <version>")
	print("Example: python3 benchmark.py 'view glass' 2.0.23")

# Add all audio files to file list
def include_wavs_from_folder(path):
	file_list=[]
	for root, _, files in walk(path):
		for f in files:
			extension = splitext(f)[1].lower()
			if extension in extension_list:
				file_list.append(join(root, f))
	return file_list

def get_random_file(file_list):
	filelen = len(file_list)
	index = randint(0,filelen-1)
	return file_list[index]

def split_sequence(a,seg_length):
	return [a[x:x+seg_length] for x in range(0,len(a),seg_length)]

def get_framelen(version):

	framelen_v1 = 800
	framelen_v2 = 640
	framelen_v3 = 1280

	if version[0] == "1":
		return framelen_v1
	elif version[0] == "2":
		return framelen_v2
	elif version[0] == "3":
		return framelen_v3
	else:
		print("Unknown version")

def get_melcount(version):

	melcount_v1 = 40
	melcount_v2 = 40
	melcount_v3 = 80

	if version[0] == "1":
		return melcount_v1
	elif version[0] == "2":
		return melcount_v2
	elif version[0] == "3":
		return melcount_v3
	else:
		print("Unknown version")



# Run positive examples. Each positive example is surrounded by a short
# silence. 
def run_good(keyword,add_noise,version,noiseIdx,sensIdx):

	seed(1234)
	sensitivity = sensitivitys[sensIdx]
	snr = noise_levels[noiseIdx-1]

	records_path = os.path.join(keyword_folder,keyword)
	filelist = include_wavs_from_folder(records_path)

	print("Positive samples: {}".format(len(filelist)))
	detector = AudioRecognition(nyumaya_lib_path)
	extractor = FeatureExtractor(nyumaya_lib_path)

	modelpath = os.path.join(nyumaya_basepath,"models/Hotword/{}_v{}.premium".format(keyword,version))
	detector.addModel(modelpath,sensitivity)

	samplenumber = 0
	detectionnumber = 0

	noise_list = []

	for noise_folder in noise_folder_list:
		if(os.path.exists(noise_folder)):
			noise_list += include_wavs_from_folder(noise_folder)

	bufsize = detector.getInputDataSize() * 2

	for f in filelist:

		wavdata,_ = load_audio_file(f)
		if(not wavdata):
			#Better abort than get a wrong result
			print("Could not load file {}".format(f))
			exit(0)


		# Some audio files are recorded at very low volume
		# this is a configuration error. Boost the volume and
		# print a warning
		if(wavdata.dBFS < -40):
			wavdata = wavdata.apply_gain(20)
			print("WARNING: Eval sample volume too low")

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
			if(bufsize == len(frame)):
				features = extractor.signalToMel(frame)
				prediction = detector.runDetection(features)

				if(prediction != 0 and prediction != -1):
					has_detected_something = True

		if(has_detected_something == True):
			detectionnumber += 1

		samplenumber += 1

		# Test: Run long pauses in between activations
		# Use random files from the demand dataset
		if(True):
			wavdata = get_random_file(noise_list)
			wavdata,_ = load_audio_file(wavdata)
			if(not wavdata):
				#Better abort than get a wrong result
				print("Could not load file {}".format(f))
				exit(0)

			wavdata = wavdata.get_array_of_samples().tobytes()
			splitdata = split_sequence(wavdata,bufsize)

			for frame in splitdata:
				if(bufsize == len(frame)):
					features = extractor.signalToMel(frame)
					prediction = detector.runDetection(features)

	accuracy = detectionnumber / samplenumber

	result = {"type": "accuracy","noiseIdx":noiseIdx,"sensIdx": sensIdx,"value":accuracy}
	resultQueue.put(result)
	print("{:.3f} @ {}".format(accuracy,sensitivity))

# Gather the results from all worker processes
# and write them to the benchmark Result
def interpretResult():
	bResult = benchmarkResult()
	while(True):
		result = resultQueue.get()
		if(result["type"] == "accuracy"):
			bResult.setAccuracy(result["noiseIdx"],result["sensIdx"],result["value"])
		elif (result["type"] == "falseActivation"):
			bResult.setFalseActivations(result["szenIdx"],result["sensIdx"],result["value"])
		elif (result["type"] == "runHours"):
			bResult.setRunHours(result["szenIdx"],result["value"])
		elif (result["type"] == "finished"):
			break
		else:
			print("Invalid result type {}".format(result["type"]))
			exit(1)

	benchmarkResultQueue.put(bResult)

# Run a szenario which do not contain positive examples.
# So each activation is a false positive
def run_szenario(szenario,sensitivity,keyword,version,szenIdx,sensIdx):
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
	from tensorflow import io as tfio
	from tensorflow import data as tfdata
	from tensorflow import data as tfdata
	from tensorflow import string as tfstring
	from tensorflow import uint8 as tfuint8

	# Decode data from szenario tfrecord
	def decode_szenario(serialized_example):

		features = tfio.parse_single_example(
			serialized_example,
			features={
			'meldata': tfio.FixedLenFeature([], tfstring),
			'utf_text': tfio.FixedLenFeature([], tfstring),
		})

		meldata = tfio.decode_raw(features['meldata'], tfuint8)
		text = tfio.decode_raw(features['utf_text'], tfuint8)

		return meldata,text

	detector = AudioRecognition(nyumaya_lib_path)
	detector.addModel(modelpath,sensitivity)

	dataset = glob.glob(os.path.join(szenario_basepath,"{}.tfrecords".format(szenario)))
	dataset = tfdata.TFRecordDataset([dataset])
	false_alarms_per_hour = 0.0
	run_seconds = 0
	run_frames = 0
	false_alarms = 0
	bufsize = detector.getInputDataSize() * 2

	for i,elem in enumerate(dataset):

		mel,text = decode_szenario(elem)
		run_frames += (mel.shape[0] / get_melcount(version))
		framelen = get_framelen(version)

		frames = split_sequence(mel,framelen)
		for frame in frames:
			if(len(frame) == framelen):
				prediction = detector.runDetection(frame)
				if(prediction != 0 and prediction != -1):
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

	result = {"type": "falseActivation","szenIdx":szenIdx,"sensIdx": sensIdx,"value":false_alarms_per_hour}
	resultQueue.put(result)
	
	result = {"type": "runHours","szenIdx":szenIdx,"value":run_hours}
	resultQueue.put(result)

if(len(sys.argv) != 3):
	usage()
	exit(1)

version=None

workers = []

keyword = sys.argv[1]
keyword = keyword.lower()
version = sys.argv[2]
outfile = '{}_v{}.txt'.format(keyword,version)
outfile = os.path.join(result_folder,outfile)

modelpath = os.path.join(nyumaya_basepath,"models/Hotword/{}_v{}.premium".format(keyword,version))
print("Model path: {}".format(modelpath))

if(not os.path.exists(modelpath)):
	print("Model does not exist: {}".format(modelpath))
	os.exit(1)

#Clean Accuracy
for sensIdx,sens in enumerate(sensitivitys):
	p = Process(target=run_good, args=(keyword,False,version,0,sensIdx))
	workers.append(p)

#Noisy Accuracy
for noiseIdx,level in enumerate(noise_levels):
	for sensIdx,sens in enumerate(sensitivitys):
		p = Process(target=run_good, args=(keyword,True,version,noiseIdx+1,sensIdx))
		workers.append(p)

for szenIdx,szen in enumerate(szenarios):
	for sensIdx,sens in enumerate(sensitivitys):
		p = Process(target=run_szenario, args=(szen,sens,keyword,version,szenIdx,sensIdx))
		workers.append(p)


# FIXME: Modify starting processes so cpu_count is never
# exceeded.
for pr in workers:
	pr.start()

interpreter = Process(target=interpretResult, args=())
interpreter.start()

for pr in workers:
	pr.join()

resultQueue.put({"type": "finished"})
interpreter.join()

result = benchmarkResultQueue.get()
result.write(outfile)







