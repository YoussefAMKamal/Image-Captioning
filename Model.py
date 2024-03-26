import os
import keras
import numpy as np
from PIL import Image
from keras.models import Model
from keras.utils import plot_model
from keras.utils import pad_sequences
from keras.applications import Xception
from nltk.translate.bleu_score import corpus_bleu

import Dataset

def extractingFeaturesModel():

    model = Xception(include_top=False, pooling='avg')

    return model

def extractingFeatures(Images):

    features = {}
    location = "Flicker8k_Dataset\\Images\\"

    model = extractingFeaturesModel()

    print("Extracting Features...")

    for img in Images:

        image_path = os.path.join(location, img)
        image = Image.open(image_path)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0

        feature = model.predict(image, verbose = None)
        features[img] = feature

    np.save("Outputs\\Features.npy", features)

    print("Extracting Features Done.")

def buildModel(max_voc, max_sen):

    in1 = keras.layers.Input(shape=(2048,), name='Input_layer1')
    D1 = keras.layers.Dropout(0.5, name='Droupout_1')(in1)
    H1 = keras.layers.Dense(256, activation='relu', name='Hidden_layer1')(D1)

    in2 = keras.layers.Input(shape=(max_sen,), name='Input_layer2')
    Em = keras.layers.Embedding(max_voc, 256, mask_zero=True, name='Embedding_layer')(in2)
    D2 = keras.layers.Dropout(0.5, name='Droupout_2')(Em)
    LSTM = keras.layers.LSTM(256, name='LSTM_layer')(D2)

    Adder = keras.layers.add([H1, LSTM], name='Add_layer')
    H2 = keras.layers.Dense(256, activation='relu', name='Hidden_layer2')(Adder)
    Ou = keras.layers.Dense(max_voc, activation='softmax', name='Output_layer')(H2)

    model = Model(inputs=[in1, in2], outputs=Ou)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    plot_model(model, to_file='Outputs\\Model.png', show_shapes=True)

    return model

def fit_model(training_description, test_description, steps, batch_size, features, vectorize_layer, max_sen, max_voc):

    model = buildModel(max_voc, max_sen)
    
    epochs = 30

    for i in range(epochs):

        print("Epoch {}/{}".format(i + 1, epochs))

        training_generator = Dataset.data_generator(training_description, features, vectorize_layer, max_sen, max_voc, batch_size)
        
        model.fit(training_generator, epochs=1, steps_per_epoch= steps, verbose=1)

        model.save("Outputs\\Model\\Model Epoch "+ str(i + 1) + ".h5")

    print("Evaluating Model with Training Data...")

    training_loss, training_accuracy = model.evaluate(training_generator, steps=steps)

    training_loss = round(training_loss, 3)
    training_accuracy = round(training_accuracy * 100, 3)

    model.save("Outputs\\Model\\Model Epoch_"+ str(epochs) + " Loss_"+ str(training_loss) + " Acc_" + str(training_accuracy) + ".h5")
    model.save("Outputs\\Model.h5")

def test_evaluate(model, test_description, steps, batch_size, features, vectorize_layer, max_sen, max_voc):

    print("Evaluating Model with Test Data...")

    test_generator = Dataset.data_generator(test_description, features, vectorize_layer, max_sen, max_voc, batch_size)
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)

    print("Test Loss = " + str(round(test_loss,3)))
    print("Test Accuracy = " + str(round(test_accuracy * 100,3)) + "%")


def predict_caption(image, model, vectorize_layer, max_sen):

    text = ""

    for i in range(max_sen):

        vector = vectorize_layer(text)

        vector = pad_sequences([vector],max_sen)

        result = model.predict([image, vector], verbose = None)

        result = np.argmax(result)

        word = Dataset.getWord(result, vectorize_layer)

        if word is None: 
            break
        
        text += word + " "

    return text

def extractingFeatureOneImage(image_path):

    model = extractingFeaturesModel()

    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0

    feature = model.predict(image, verbose = None)

    return feature

def model_validate(model, features, descriptions, vectorize_layer, max_sen):

    actual, predicted = list(), list()
    num = 0

    print("Validating Model...")
    
    for key, captions in descriptions.items():

        num += 1

        if num%100 == 0:
            print(str(num) + " images are completed")

        pred = predict_caption(features.item().get(key), model, vectorize_layer, max_sen)
        pred = pred.split()
        predicted.append(pred)
        
        actual_captions = [caption.split() for caption in captions]
        actual.append(actual_captions)

    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

    print("\nCalculating Score...")
    print("BLEU-1: %f" % b1)
    print("BLEU-2: %f" % b2)
