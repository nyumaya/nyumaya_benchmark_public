
from config import *
import numpy as np
import subprocess

class benchmarkResult:

	def __init__(self):
		sensCount = len(sensitivitys)
		self.accuracy = np.ones(((len(noise_levels)+1), sensCount)) * -1
		self.falseActivations = np.ones((len(szenarios), sensCount)) * -1
		self.runHours = np.ones(len(szenarios)) * -1

	def setAccuracy(self,noiseIdx,sensIdx,value):
		self.accuracy[noiseIdx][sensIdx] = value
	
	def setFalseActivations(self,szenIdx,sensIdx,value):
		self.falseActivations[szenIdx,sensIdx] = value

	def setRunHours(self,szenIdx,value):
		self.runHours[szenIdx] = value


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
				result_file.write("Accuracy noisy ({} db Signal-to-noise ratio) \n".format(noise_levels[idx]))
				for sensIdx,sens in enumerate(sensitivitys):
					result_file.write("{:.4f} @ {} \n".format(self.accuracy[idx+1][sensIdx],sens))

				result_file.write("\n")

			for idx,szen in enumerate(szenarios):
				result_file.write("\n\nFalse alarms per hour: {}\n".format(szen))
				result_file.write("Szenario length: {:.4f} hours\n".format(self.runHours[idx]))
				for sensIdx,sens in enumerate(sensitivitys):
					result_file.write("{:.4f} @ {} \n".format(self.falseActivations[idx][sensIdx],sens))


			result_file.write("\n\nFalse alarms per hour: Combined\n")
			#Combined
			combinedRunHours = 0
			for idx,szen in enumerate(szenarios):
				combinedRunHours += self.runHours[idx]

			for sensIdx,sens in enumerate(sensitivitys):
				fa_sens=0.0
				for idx,szen in enumerate(szenarios):
					fa_sens += self.falseActivations[idx][sensIdx]

				result_file.write("{:.4f} @ {} \n".format(fa_sens/float(idx),sens))

			result_file.write("Combined length: {:.4f} hours\n".format(combinedRunHours))


