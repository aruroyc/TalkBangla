'''
Author: roychoud
'''

from data.file_tools import FileUtils
class Util:
	def __init__(self):
		self.file_utils = FileUtils()
	def text_to_int_sequence(self,text):
		""" Convert text to an integer sequence """
		int_sequence = []
		for c in text:
			ch = self.file_utils.charmap[c]
			ch = ch == '<SPACE>' and ' ' or  ch
			int_sequence.append(ch)
		return int_sequence

	def int_sequence_to_text(self,int_sequence):
		""" Convert an integer sequence to text """
		text = []
		for c in int_sequence:
			c = c == '<SPACE>' and ' ' or c
			ch = self.file_utils.index_map[c]
			text.append(ch)
		return text
