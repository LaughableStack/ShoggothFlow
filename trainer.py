from model import get_model
import numpy as np
import pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
datapoints = 200*1000
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
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=30,test_size =0.2)
filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
pickle.dump((vocab,inputlength),open("model_data.dat","wb+"))
model.fit(x_train,y_train, batch_size=64, epochs=60, verbose=1, validation_data = (x_test,y_test),callbacks=callbacks_list)