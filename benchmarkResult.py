
from config import *
import numpy as np
import subprocess

class benchmarkResult:

	def __init__(self):
		sensCount = len(sensitivitys)
		self.accuracy = np.ones(((len(noise_levels)+1), sensCount)) * -1
		self.falseActivations = np.ones((len(szenarios), sensCount)) * -1
	
	def setAccuracy(self,noiseIdx,sensIdx,value):
		self.accuracy[noiseIdx][sensIdx] = value
	
	def setFalseActivations(self,szenIdx,sensIdx,value):
		self.falseActivations[szenIdx,sensIdx] = value

	def write(self,outfile):

		gitId = "unknown"
		try:
			gitId = subprocess.check_output(["git", "rev-parse" ,"HEAD"]).strip()
		except:
			print("Failed to get git id")


		with open(outfile, "w+") as result_file:
			result_file.write("Git Commit Id: {}\n".format(gitId))
			result_file.write("Accuracy clean \n")
			for sensIdx,sens in enumerate(sensitivitys):
				result_file.write("{:.4f} @ {} \n".format(self.accuracy[0][sensIdx],sens))

			result_file.write("\n")
			for idx,level in enumerate(noise_levels):
				result_file.write("Accuracy noisy ({} db) \n".format(noise_levels[idx]))
				for sensIdx,sens in enumerate(sensitivitys):
					result_file.write("{:.4f} @ {} \n".format(self.accuracy[idx+1][sensIdx],sens))
				
				result_file.write("\n")

			for idx,szen in enumerate(szenarios):
				result_file.write("\n\nFalse alarms per hour: {}\n".format(szen))
				for sensIdx,sens in enumerate(sensitivitys):
					result_file.write("{:.4f} @ {} \n".format(self.falseActivations[idx][sensIdx],sens))




