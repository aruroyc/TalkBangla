'''
Author: roychoud
'''

import _pickle as pickle
from data_engine import *

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda, BatchNormalization,LSTM,TimeDistributed,Activation,Dense)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
import os

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
	the_labels = Input(name='labels', shape=(None,), dtype='float32')
	input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
	label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
	output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
		[input_to_softmax.output, the_labels, output_lengths, label_lengths])
	model = Model(
		inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
		outputs=loss_out)
	return model

def deep_rnn_model(input_dim, output_dim=29):
	units = 200
	recur_layers = 25
	input_data = Input(name='input', shape=(None, input_dim))
	layer = input_data
	for i in range(1,recur_layers):
		layer = LSTM(units, return_sequences=True, activation='relu')(layer)
		layer = BatchNormalization(name='bt_rnn_{}'.format(i))(layer)
	time_dense = TimeDistributed(Dense(output_dim))(layer)
	y_pred = Activation('softmax', name='softmax')(time_dense)
	model = Model(inputs=input_data, outputs=y_pred)
	model.output_length = lambda x: x
	return model

def train_model(X,Y,optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),epochs=20,verbose=1):

	model = deep_rnn_model(input_dim=15)
	model = add_ctc_loss(model)
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
	if not os.path.exists('results'):
		os.makedirs('results')
	checkpointer = ModelCheckpoint(filepath='results/ASR_model.h5', verbose=0)
	hist = model.fit(x=X,y=Y,
					 batch_size=10,epochs=epochs,
					 verbose=verbose,callbacks=[checkpointer],
					 validation_split=0.25)
	with open('results/model_data.pkl', 'wb') as f:
		pickle.dump(hist.history, f)


def main():
	folder = '/Users/roychoud/Downloads/asr/asr_bengali/data'
	engine = DataEngine(20)
	pklPath = folder+'_mfcc_dump.pkl'
	if not os._exists(pklPath):
		engine.create_data_pkl(folder,file_type='tsv')
	final_df = engine.load_pkl_data(folder)
	input_dict,output = engine.map_input_data(final_df)
	train_model(input_dict,output)


if __name__ == '__main__':
	main()
