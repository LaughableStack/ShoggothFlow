from model import get_model
import pickle
import numpy as np
data = pickle.load(open("model_data.dat","rb+"))
topwords = pickle.load(open("Topwords.dat","rb+"))
model = get_model(data[0],data[1])
checkpoint_path = "weights/(newest weights file)"
model.load_weights(checkpoint_path)
def get_pred(inp):
    return model.predict_classes(np.array([np.array(inp)]))
moving_set = ([np.random.randint(0,data[0])-1,np.random.randint(0,data[0])-1,np.random.randint(0,data[0])-1])
print(moving_set)
words = []
sentlen = 10
for i in range(0,sentlen):
    res = (get_pred(moving_set))[0]
    words.append(res)
    moving_set.append(res)
    moving_set.pop(0)
print(' '.join(list(map(lambda x: topwords[x],words ))))
