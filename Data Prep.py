import pickle
from collections import Counter
import numpy as np
from functools import partial
from operator import ne
import keras
vocab = 10000
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
  pstor = cleanup(story).split(" ")
  for word in pstor:
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
sentences = np.array(wordsequence)
spoints = []
for wordind in range(len(wordsequence)):
  if wordsequence[wordind] in ".!?;":
    spoints.append(wordind+1)
sentences = np.split(sentences,spoints)
sentences = np.array(list(map(lambda x: np.array(list(map(getindex,x))),sentences)))
wvocab = 4*np.ceil(vocab/16)
sentences = keras.preprocessing.sequence.pad_sequences(sentences, padding='post', value=wvocab)
# Prepare object to save
#layout: (x,topwords)
saveobject = (sentences.reshape(sentences.shape[0],sentences.shape[1],1),vocab)
f = open("PreparedData.dat","wb+")
pickle.dump(saveobject,f)
f.close()
f = open("Topwords.dat","wb+")
pickle.dump(topwords,f)
f.close()