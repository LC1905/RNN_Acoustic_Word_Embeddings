import pickle
import argparse
import numpy as np
from collections import OrderedDict
import os
import string

transcript_path = "data/full_swbd_trans/swb_ms98_transcriptions/"

'''
1/4 for training, 1/4 for dev, 1/2 for test
'''

def partition_file(dt):
	with open("data/full_utts_swbd/mfccs_file_1.pkl", "rb") as f:
		print("loading data...")
		data = pickle.load(f)
		sample_keys = np.array(list(data.keys()), dtype=object)
		total_num = len(sample_keys)

		print("shuffling the data")
		indices = np.arange(total_num)
		np.random.shuffle(indices)
		sample_keys = sample_keys[indices]

		print("processing training data..")
		train_keys = sample_keys[:total_num // 4]
		train_npz = OrderedDict()
		train_labels = []
		for key in train_keys:
			kid = key[3:7] + key[8] + "-" + key[10:16]
			transcription = find_transcription(key, dt)
			if transcription != None:
				train_labels.append(transcription + "_" + kid)
				train_npz[transcription + "_" + kid] = data[key]
		np.savez("sentence_data/train.npz", **train_npz)
		np.save("sentence_data/train_labels.npy", train_labels)
		print("size of training data: {}".format(len(train_labels)))
		
		print("processing dev data..")
		dev_keys = sample_keys[total_num // 4: total_num // 2]
		dev_npz = OrderedDict()
		dev_labels = []
		for key in dev_keys:
			kid = key[3:7] + key[8] + "_" + key[10:16]
			transcription = find_transcription(key, dt)
			if transcription != None:
				dev_labels.append(transcription + "_" + kid)
				dev_npz[transcription + "_" + kid] = data[key]
		np.savez("sentence_data/dev.npz", **dev_npz)
		np.save("sentence_data/dev_labels.npy", dev_labels)
		print("size of dev data: {}".format(len(dev_labels)))


		print("processing testing data..")
		test_keys = sample_keys[total_num // 2:]
		test_npz = OrderedDict()
		test_labels = []
		for key in test_keys:
			kid = key[3:7] + key[8] + "_" + key[10:16]
			transcription = find_transcription(key, dt)
			if transcription != None:
				test_labels.append(transcription + "_" + kid)
				test_npz[transcription + "_" + kid] = data[key]
		np.savez("sentence_data/test.npz", **test_npz)
		np.save("sentence_data/test_labels.npy", test_labels)
		print("size of test data: {}".format(len(test_labels)))


def input1(train_path):
	train = np.load(train_path)
	keys = list(train.keys())
	values = [train[key] for key in keys]
	np.save("input1.npy", values)


def input2(train_path):
	train = np.load(train_path)
	keys = list(train.keys())
	values = []
	for key in keys:
		key = key.lower()
		key = key.split("_")[0]
		#key = "".join(key)
		value = word2onehot(key)
		#print(value)
		values.append(np.array(value, dtype=float))
	values = np.asarray(values)
	np.save("input2.npy", values)



def word2onehot(word):
	alpha = string.ascii_lowercase
	res = []
	for s in word:
		#print(s)
		inc = [0] * 26
		ind = alpha.index(s)
		inc[ind] = 1
		res.append(inc)
	return res

def transcript_map(rg):
	transcript = {}
	for ids in rg:
		dir1 = transcript_path + str(ids) 
		print(dir1)
		if os.path.exists(dir1):
			for sub in range(100):
				sub = str(ids * 100 + sub)
				dir2 = dir1 + "/" + sub
				if os.path.exists(dir2):
					for char in ['A', 'B']:
						path = dir2 + "/sw" + sub + char + "-ms98-a-trans.text"
						file = open(path, "r")
						for line in file:
							line = line.split()
							key1 = line[0][2:7]
							key2 = str(round(float(line[1]), 2))
							key2 = "".join(k for k in key2 if k != ".")
							key2 = key2.rjust(6, "0")
							value = line[3:]
							#print("value = ", value)
							# ignore all signs and spaces
							value = ["".join([vi for vi in v if vi.isalpha()]) for v in value if v[0] != "["]
							value = "-".join(value)
							value = value.lower()
							if value != "":
								if key1 not in transcript:
									transcript[key1] = {}
								transcript[key1][key2] = value
						file.close()
	return transcript


def find_transcription(key, dt):
	k1 = key[3:7] + key[8]
	k2 = key[10:16]
	if k1 not in dt:
		#print("error: {} not found".format(k1))
		return
	else:
		if k2 not in dt[k1]:
			#print("error: {} not found".format(k2))
			return
		else:
			transcription = dt[k1][k2]
	return transcription



td = transcript_map([20, 21, 22, 23])

#partition_file()
#input1("sentence_Data/train.npz")

