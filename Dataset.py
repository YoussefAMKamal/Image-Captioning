import re
import os
import string
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from keras.utils import pad_sequences
from keras.utils import to_categorical

def open_file():

    loc = "Flicker8k_Dataset\\Labels\\Flickr8k.token.txt"

    if os.path.isfile(loc) == False:
        loc = "Dataset Tokens.txt"

    file = open(loc)

    labels = file.read()
    file.close()

    labels = labels.split("\n")

    return labels

def extract_text():

    labels = open_file()

    filenames = []
    captions = []
    vocablary = set()
    max_sen = 0

    for label in labels:

        filename, caption = label.split("\t")
        
        filename = filename[:-2]
        filenames.append(filename)

        captions.append(caption)
        caption = caption.replace("-", " ")

        for punctuation in string.punctuation:

            caption = caption.replace(punctuation, '')

        caption = caption.lower()
        caption = caption.split()
        caption = [word for word in caption if (len(word)>1)]
        caption = [word for word in caption if word.isalpha()]
        
        max_sen = max(len(caption), max_sen)

        vocablary.update(caption)

    names = []

    for i in range(len(filenames)):

        if(i%5 == 0):
            
            names.append(filenames[i])

    description = {}

    for j in range(len(names)):

        description[names[j]] = []

        for i in range(j*5,(j+1)*5):

            description[names[j]].append(captions[i])

    return description, vocablary, names, max_sen

def trainingData(description, features):

    file = open("Flicker8k_Dataset\\Labels\\Flickr_8k.trainImages.txt")
    labels = file.read()
    file.close()

    labels = labels.split("\n")

    training_features = {}
    training_description = {}

    for label in labels:

        label = str(label)

        training_features[label] = features.item().get(label)
        training_description[label] = description[label]

    np.save("Outputs\\Training Features.npy", training_features)
    np.save("Outputs\\Training Description.npy", training_description)

def testData(description, features):

    file = open("Flicker8k_Dataset\\Labels\\Flickr_8k.testImages.txt")
    labels = file.read()
    file.close()

    labels = labels.split("\n")

    test_features = {}
    test_description = {}

    for label in labels:

        label = str(label)

        test_features[label] = features.item().get(label)
        test_description[label] = description[label]

    np.save("Outputs\\Test Features.npy", test_features)
    np.save("Outputs\\Test Description.npy", test_description)

def dictToList(descriptions):

    list_of_descriptions = []

    for key in descriptions.keys():

        for caption in descriptions[key]:

            caption = caption.lower()
            caption = caption.split()
            caption = [word for word in caption if (len(word)>1)]
            caption = [word for word in caption if word.isalpha()]
            caption = ' '.join(caption)
            list_of_descriptions.append(caption)

    return list_of_descriptions


def normalize(text):

    punctuation = f'[{re.escape(string.punctuation)}]'

    result = tf.strings.lower(text, encoding='utf-8')
    result = tf.strings.regex_replace(result, "-", " ")
    result = tf.strings.regex_replace(result, punctuation, '')

    return result


def vectorization(descriptions, max_voc, max_sen):

    list_of_descriptions = dictToList(descriptions)

    vectorize_layer = TextVectorization(
        max_tokens = max_voc,
        output_mode = "int",
        output_sequence_length = max_sen,
        standardize = normalize
    )
    
    vectorize_layer.adapt(list_of_descriptions)

    return vectorize_layer

def cleanCaption(caption):

    caption = caption.lower()
    caption = caption.split()
    caption = [word for word in caption if (len(word)>1)]
    caption = [word for word in caption if word.isalpha()]
    caption = ' '.join(caption)

    return caption

def data_generator(descriptions, features, vectorize_layer, max_sen, max_voc, batch_size):
    
    a, b, c = list(), list(), list()

    while 1:
            
        num = 0

        for key, captions in descriptions.items():

            num += 1
            feature = features.item().get(key)[0]

            for caption in captions:
                
                caption = cleanCaption(caption)

                vector = vectorize_layer(caption)

                for i in range(len(vector)):

                    input = pad_sequences([vector[:i]], maxlen=max_sen)[0]
                    output = to_categorical([vector[i]], num_classes=max_voc)[0]
                    
                    a.append(feature)
                    b.append(input)
                    c.append(output)

            if(num == batch_size):

                num = 0
                an = np.array(a)
                bn = np.array(b)
                cn = np.array(c)
                a, b, c = list(), list(), list()

                yield [an, bn], cn
            
def getWord(digit, vectorize_layer):

    voc = vectorize_layer.get_vocabulary()

    return voc[digit]

def getString(vector, vectorize_layer):

    voc = vectorize_layer.get_vocabulary()
    
    str = list()

    for i in range(len(vector)):

        str.append(voc[vector[i]])
    
    str = " ".join(str)
    
    return str