import keras
from keras import Sequential
from keras.layers import Dense, ZeroPadding1D,LeakyReLU, LSTM,Dropout, Activation, Embedding,Input,UpSampling1D,Reshape,Conv1D,BatchNormalization
from keras.optimizers import Adam
from keras import Model
from PIL import Image
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
class Adversarial:
    def __init__(self,inputlength,vocabulary,topwords,label):
        self.inputlength = inputlength
        self.vocab = vocabulary
        self.topwords = topwords
        self.label = label
        #Amount of randomly generated numbers for the first layer of the generator.
        self.random_noise_dimension = 100

        #Just 10 times higher learning rate would result in generator loss being stuck at 0.
        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.generator = self.build_generator()

        #A placeholder for the generator input.
        random_input = Input(shape=(self.random_noise_dimension,))

        #Generator generates images from random noise.
        generated_image = self.generator(random_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        #Discriminator attempts to determine if image is real or generated
        validity = self.discriminator(generated_image)

        #Combined model = generator and discriminator combined.
        #1. Takes random noise as an input.
        #2. Generates an image.
        #3. Attempts to determine if image is real or generated.
        self.combined = Model(random_input,validity)
        self.combined.compile(loss="binary_crossentropy",optimizer=optimizer)

    def get_training_data(self,datacont):
        training_data = datacont
        self.training_data = datacont
        return self.training_data


    def build_generator(self):
        #Generator attempts to fool discriminator by generating new images.
        model = Sequential()

        model.add(Dense(int(self.inputlength/16),activation="relu",input_dim=self.random_noise_dimension))
        model.add(Reshape((int(self.inputlength/16),1)))
        model.add(UpSampling1D())
        model.add(Conv1D(4,kernel_size=1,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling1D())
        model.add(Conv1D(4,kernel_size=1,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling1D())
        model.add(Conv1D(2,kernel_size=1,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling1D())
        model.add(Conv1D(2,kernel_size=1,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))


        model.add(Conv1D(1,kernel_size=1,padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        input = Input(shape=(self.random_noise_dimension,))
        generated_image = model(input)

        return Model(input,generated_image)


    def build_discriminator(self):
        #Discriminator attempts to classify real and generated images
        model = Sequential()

        model.add(LSTM(15,input_shape=[self.inputlength,1],return_sequences=True))
        model.add(Dropout(rate=0.25))
        model.add(LSTM(15))
        model.add(Dropout(rate=0.25))
        model.add(Dense(15,activation="relu"))
        #Outputs a value between 0 and 1 that predicts whether image is real or generated. 0 = generated, 1 = real.
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        input_t = Input(shape=[self.inputlength,1])

        #Model output given an image.
        validity = model(input_t)

        return Model(input_t, validity)

    def train(self, datacont ,epochs,batch_size,save_images_interval,save_model_interval):
        #Get the real texts
        training_data = self.get_training_data(datacont)
        print("Training Data Loaded")
        #Map all values to a range between -1 and 1.
        training_data = ( training_data / (self.vocab/2) ) - 1.
        print("Training Data Processed")
        #Two arrays of labels. Labels for real text: [1,1,1 ... 1,1,1], labels for generated text: [0,0,0 ... 0,0,0]
        labels_for_real = np.ones((batch_size,1))
        labels_for_generated = np.zeros((batch_size,1))
        print("Ready to train")
        for epoch in range(epochs):
            # Select a random half of images
            indices = np.random.randint(0,training_data.shape[0],batch_size)
            real_text = training_data[indices]

            #Generate random noise for a whole batch.
            random_noise = np.random.normal(0,1,(batch_size,self.random_noise_dimension))
            #Generate a batch of new sentences.
            generated_text = self.generator.predict(random_noise)

            #Train the discriminator on real sentences.
            discriminator_loss_real = self.discriminator.train_on_batch(real_text,labels_for_real)
            #Train the discriminator on generated sentences.
            discriminator_loss_generated = self.discriminator.train_on_batch(generated_text,labels_for_generated)
            #Calculate the average discriminator loss.
            discriminator_loss = 0.5 * np.add(discriminator_loss_real,discriminator_loss_generated)

            #Train the generator using the combined model. Generator tries to trick discriminator into mistaking generated images as real.
            generator_loss = self.combined.train_on_batch(random_noise,labels_for_real)
            print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))

            if epoch % save_images_interval == 0:
                print(self.gen_sentences(epoch))
            if epoch % save_model_interval == 0:
                self.generator.save_weights("saved_models/wordgenerator_" + self.label + ".hdf5")
                self.discriminator.save_weights("saved_models/worddiscriminator_" + self.label + ".hdf5")
        #Save the model for a later use
        self.generator.save_weights("saved_models/wordgenerator"+str(epochs)+".hdf5")
        self.discriminator.save_weights("saved_models/worddiscriminator"+str(epochs)+".hdf5")


    def gen_sentences(self,epoch):
        #Save 25 generated images for demonstration purposes using matplotlib.pyplot.
        noise = np.random.normal(0, 1, (3, self.random_noise_dimension))
        model = self.generator;
        generation = model.predict(noise)
        generation+=1
        generation*=(self.vocab/2)
        generation = generation.astype(np.uint16)
        generation = generation.reshape(generation.shape[0],generation.shape[1])
        outset = []
        for sent in generation:
            outpart = []
            for word in sent:
                outpart.append(self.topwords[word])
                if (self.topwords[word] == "."):
                    break
            outset.append(outpart)
        return (list(map("".join,outset)))
    def load_weighting(self,gen_path,disc_path):
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path);
