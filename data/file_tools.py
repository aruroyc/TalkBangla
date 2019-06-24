'''
Author: roychoud
'''
import pandas as pd

class FileUtils:
	def __init__(self):
		self.utterances = {}
		self.charmap = {}
		self.index_map = {}
		self.max_str_length = -1;

	def create_utterance_map(self,path,utterance_files,file_type):
		index = 0
		seperator = file_type == 'csv' and ',' or '\t'
		for file in utterance_files:
			df = pd.read_csv(path+'/'+file,sep=seperator)
			for _,row in df.iterrows():
				utterance = row['utterance'].strip()
				utterance_encoded = []
				for eachChar in utterance:
					eachChar = eachChar==' ' and '<SPACE>' or eachChar
					if self.charmap.get(eachChar) is None:
						self.charmap[eachChar] = index
						self.index_map[index] = eachChar
						index=index+1;
					utterance_encoded.append(self.charmap[eachChar])
				self.utterances[path + '/' + row['file'] + '.flac'] = utterance_encoded
				self.max_str_length = self.max_str_length< len(utterance) and len(utterance) or self.max_str_length
		self.index_map[index] =' '
		self.charmap[' '] = index
	def get_utterance(self,fileName):
		return self.utterances[fileName]

