from model import get_model
import numpy as np
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
datapoints = 20000*1000
f=open("PreparedData.dat","rb+")
data = pickle.load(f)
f.close()
vocab=data[1]
print(vocab)
x = data[0][0][:datapoints]
y = data[0][1][:datapoints]
print(len(x))
y=to_categorical(y,num_classes=vocab)
inputlength = x.shape[1]
model = get_model(vocab,inputlength)
#Turn this on if you want to use a test set #x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=30,test_size =0.2)
filepath="weights/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
pickle.dump((vocab,inputlength),open("model_data.dat","wb+"))
model.fit(x,y, batch_size=32, epochs=40, verbose=1,callbacks=callbacks_list)
model.save_weights("weights/newest.hdf5")