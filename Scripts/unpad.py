import numpy as np
from collections import OrderedDict
data_path = "../awe_data/"
out_path = "../word_data/"

def unpad_and_transpose(fpath, output):
	f = np.load(data_path + fpath)
	print(f)
	d = []
	for k in f.files:
		vec = f[k]
		vec = vec[vec != 0].reshape(-1, 39)
		d.append((k, vec))
	d = OrderedDict(d)
	np.savez(out_path + output, **d)

unpad_and_transpose("swbd.test.npz", "test.npz")
unpad_and_transpose("swbd.dev.npz", "dev.npz")		
