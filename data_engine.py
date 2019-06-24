'''
Author: roychoud
'''

from data.file_tools import *
from data.speech_tools import *
from utils import *
from concurrent.futures import ThreadPoolExecutor as Pool
import pickle
import os
class DataEngine:

	def __init__(self,parallelism=50):
		self.fileUtil = FileUtils()
		self.speechUtil = SpeechUtils()
		self.pool = Pool(parallelism)
		self.utils = Util()

	def __generate_utterances_map(self,path,files,file_type):
		return self.fileUtil.create_utterance_map(path,files,file_type)

	def __create_batch(self,fileName,utterance):
		return {"file":fileName,"utterance":utterance}

	def __get_mfcc(self,job):
		try:
			_,mfcc = self.speechUtil.get_mfcc_from_file(job['file'])
			return {'utterance': job['utterance'], 'mfcc': mfcc}
		except Exception as e:
			print('Failed for job: '+job['utterance'])
			return {}

	def prep_data(self,folder,file_type):
		files = [file for file in os.listdir(folder) if file.endswith('.'+file_type)]
		self.__generate_utterances_map(folder,files,file_type)
		jobs = []
		for name,utterance in self.fileUtil.utterances.items():
			jobs.append(self.__create_batch(name,utterance))
		mfccData = self.pool.map(self.__get_mfcc,jobs)
		final_df = pd.DataFrame()
		for data in mfccData:
			final_df=final_df.append(data,ignore_index=True)
		return final_df

	def __save_data(self,data,folder_name=''):
		with open(folder_name+'_mfcc_dump.pkl','wb') as file:
			pickle.dump(data,file)

	def create_data_pkl(self,folder,file_type='csv'):
		final_dict = self.prep_data(folder,file_type)
		self.__save_data(final_dict,folder)

	def load_pkl_data(self,folder):
		with open(folder+'_mfcc_dump.pkl','rb') as file:
			return pickle.load(file)


	def map_input_data(self,final_df):
		max_length = max([x.shape[0] for x in final_df['mfcc']])
		max_string_length = max([len(x) for x in final_df['utterance']])
		X_data = np.zeros([final_df.shape[0], max_length,15])
		labels = np.ones([final_df.shape[0], max_string_length]) * 28
		input_length = np.zeros([final_df.shape[0], 1])
		label_length = np.zeros([final_df.shape[0], 1])
		for i in range(0, final_df.shape[0]):
			feat = final_df.iloc[i]['mfcc']
			input_length[i] = feat.shape[0]
			X_data[i, :feat.shape[0], :] = feat

			# calculate labels & label_length
			label = np.array(final_df.iloc[i]['utterance'])
			labels[i, :len(label)] = label
			label_length[i] = len(label)

		# return the arrays
		outputs = {'ctc': np.zeros([final_df.shape[0]])}
		inputs = {'input': X_data,
				  'labels': labels,
				  'input_length': input_length,
				  'label_length': label_length
				  }
		return inputs,outputs