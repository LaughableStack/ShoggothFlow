from model import Adversarial
import numpy as np
import pickle
datapoints = 20000*10000000 #Set this to something lower than the set length to only use a part of it.
f=open("PreparedData.dat","rb+")
data = pickle.load(f)
f.close()
topwords = pickle.load(open("Topwords.dat","rb+"))
vocab=data[1]
print(vocab)
x = data[0]
print(x.shape)
textgen = Adversarial(x.shape[1],vocab,topwords,"Lovecraft")
textgen.train(x,1000,32,100,200)