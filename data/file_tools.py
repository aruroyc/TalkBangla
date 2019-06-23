'''
Author: roychoud
'''
import pandas as pd

class FileUtils:
	def __init__(self):
		self.utterances = {}

	def create_utterance_map(self,path,utterance_files,file_type):
		seperator = file_type == 'csv' and ',' or '\t'
		for file in utterance_files:
			df = pd.read_csv(path+'/'+file,sep=seperator)
			for _,row in df.iterrows():
				self.utterances[path+'/'+row['file']+'.flac'] = row['utterance']

	def get_utterance(self,fileName):
		return self.utterances[fileName]

