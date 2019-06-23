'''
Author: roychoud
'''
from pydub import AudioSegment
from pydub.silence import split_on_silence
from python_speech_features import mfcc
import numpy as np


class SpeechUtils:
	def __trim_silence(self,file, format='flac'):
		sound_file = AudioSegment.from_file(file, format=format)
		audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-20,keep_silence=100)
		silenced = AudioSegment.empty()
		for chunk in audio_chunks: silenced += chunk
		return sound_file

	def compute_mfcc(self,chunk, rate):
		np_audio = np.frombuffer(chunk.get_array_of_samples(), dtype=np.int16)
		mfcc_features = mfcc(np_audio, samplerate=rate, numcep=15, nfilt=35)
		return mfcc_features

	def get_mfcc_from_file(self,filePath):
		format = str(filePath)[filePath.rfind('.') + 1:]
		fileName = str(filePath)[filePath.rfind('/'):]
		silenced_audio = self.__trim_silence(filePath, format)
		mfcc = self.compute_mfcc(silenced_audio, silenced_audio.frame_rate)
		return fileName, mfcc

