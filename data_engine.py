'''
Author: roychoud
'''

from data.file_tools import *
from data.speech_tools import *
from concurrent.futures import ThreadPoolExecutor as Pool
import pickle
import os
class DataEngine:

	def __init__(self,parallelism=50):
		self.fileUtil = FileUtils()
		self.speechUtil = SpeechUtils()
		self.pool = Pool(parallelism)

	def __generate_utterances_map(self,path,files,file_type):
		return self.fileUtil.create_utterance_map(path,files,file_type)

	def __create_batch(self,fileName,utterance):
		return {"file":fileName,"utterance":utterance}

	def __get_mfcc(self,job):
		try:
			_,mfcc = self.speechUtil.get_mfcc_from_file(job['file'])
			return {'utterance': str(job['utterance']).strip(), 'mfcc': mfcc}
		except Exception as e:
			print('Failed for job: '+job['utterance'])
			return {}

	def __prep_data(self,folder,file_type):
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
		final_dict = self.__prep_data(folder,file_type)
		self.__save_data(final_dict,folder)

#if __name__ == '__main__':
#	engine = DataEngine(20)
#	engine.create_data_pkl('/Users/roychoud/Downloads/asr/asr_bengali/data')