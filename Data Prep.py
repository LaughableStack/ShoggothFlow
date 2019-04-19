import pickle
from collections import Counter
import numpy as np
from functools import partial
from operator import ne
from keras.utils import to_categorical
vocab = 15000
inputlength = 2
f = open("LovecraftPassages.dat","rb+")
passages = pickle.load(f)
passages = passages[0:10] #The first 10-20 passages seem to work best
f.close()
def cleanup(dep):
  dep = dep.replace("!"," !")
  dep = dep.replace("?"," ?")
  dep = dep.replace(";"," ;")
  #All of above characters end sentences/sequences of characters, so they'll become our sentence break character
  dep = dep.replace("\r\n"," ")
  dep = dep.replace("\n","")
  dep = dep.replace("\r","")
  dep = dep.replace(". ",".")
  dep = dep.replace("."," . ")
  #Add a space after every "."
  for i in range(0,5):
    dep = dep.replace("  "," ")
  #Get rid of long sequences of spaces
  dep = dep.replace(","," ,")
  dep = dep.replace('"',"")
  dep = dep.replace('”',"")
  dep = dep.replace('“',"")
  dep = dep.replace("-"," ")
  dep = dep.replace("—"," ")
  #Characters that aren't really part of words.
  dep = dep.replace("&","and")
  #fixing this so it's readable.
  dep = dep.replace("\xa0","")
  #Something that's present in some passages
  dep = dep.replace("'s","")
  #get rid of possesion to reveal the noun
  dep = dep.replace("(","")
  dep = dep.replace(")","")
  dep = dep.replace("'","")
  dep = dep.replace("’","")
  #More non-word characters
  dep = dep.lower()
  return dep
#Make a list of all the words, in order.
wordsequence = []
for story in passages:
  psent = cleanup(story).split(" ")
  for word in psent:
    wordsequence.append(word)
print(wordsequence[0:100])
wordsequence.remove("")
print(wordsequence[0:100])
#List the words by frequency of occurence, and take the top vocab
scount = Counter(wordsequence).most_common(vocab)
topwords = list(f[0] for f in scount)
vocab = len(topwords)
def getindex(c):
  if (c in topwords):
    return topwords.index(c)
  else:
    return vocab;
rawx = []
rawy = []
for indice in range(inputlength,len(wordsequence)):
  rawx.append(wordsequence[indice-inputlength:indice])
  rawy.append((wordsequence[indice]))
numx = list(map(lambda x: list(map(getindex,x)),rawx))
numy = list(map(getindex,rawy))
finalx = np.array(list(map(np.array,numx)))
finalx = finalx.reshape(-1,finalx.shape[1])
finaly = np.array(numy)
#---
#This part removes all outputs and their corresponding inputs where the output is "unknown"
tfinx = list(finalx)
tfiny = list(finaly)
popped = 0
for i in range(0,len(finaly)):
  if (finaly[i] == vocab):
    tfinx.pop(i-popped)
    tfiny.pop(i-popped)
    popped+=1
finalx = np.array(tfinx)
finaly = np.array(tfiny)
# You probably noticed we didn't make the outputs categorical.
# This is because such would increase the size of the data so
# much that pickle would return a memory error.
# We're going to do that in the model file
# Prepare object to save
#layout: ((x,y),topwords)
# x and y for training, topwords to translate to english
saveobject = ((finalx,finaly),vocab)
f = open("PreparedData.dat","wb+")
pickle.dump(saveobject,f)
f.close()
f = open("Topwords.dat","wb+")
pickle.dump(topwords,f)
f.close()
