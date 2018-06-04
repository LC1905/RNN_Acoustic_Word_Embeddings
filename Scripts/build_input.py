import numpy as np
import string

data_path = "../word_data/"
out_path = "../word_data/"

def input1(f):
        train = np.load(data_path + f)
        keys = list(train.keys())
        values = [train[key] for key in keys]
        np.save(out_path + "input1.npy", values)


def input2(f):
        train = np.load(data_path + f)
        keys = list(train.keys())
        values = []
        for key in keys:
                key = key.lower()
                key = list(key.split("_")[0])
                key = "".join([k for k in key if k.isalpha()])
                value = word2onehot(key)
                values.append(np.array(value))
        values = np.asarray(values)
        np.save(out_path + "input2.npy", values)

def word2onehot(word):
        alpha = string.ascii_lowercase
        res = []
        for s in word:
                inc = [0] * 26
                ind = alpha.index(s)
                inc[ind] = 1
                res.append(inc)
        return res

input2("train.npz")
