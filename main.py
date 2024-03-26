#%%
import os
import numpy as np
import keras
import warnings
from PIL import Image
import matplotlib.pyplot as plt

import Dataset
import Model

warnings.simplefilter("ignore")

descriptions, vocablary, names, max_sen = Dataset.extract_text()

file = "Outputs\\Features.npy"

if os.path.isfile(file) == False:

    Model.extractingFeatures(names)

features = np.load(file, allow_pickle=True)

trainingFeatures = "Outputs\\Training Features.npy"
trainingDescription = "Outputs\\Training Description.npy"

if os.path.isfile(trainingFeatures) == False or os.path.isfile(trainingDescription) == False:

    Dataset.trainingData(descriptions, features)

training_features = np.load(trainingFeatures, allow_pickle=True)
training_description = np.load(trainingDescription, allow_pickle=True).item()

testFeatures = "Outputs\\Test Features.npy"
testDescription = "Outputs\\Test Description.npy"

if os.path.isfile(testFeatures) == False or os.path.isfile(testDescription) == False:

    Dataset.testData(descriptions, features)

test_features = np.load(testFeatures, allow_pickle=True)
test_description = np.load(testDescription, allow_pickle=True).item()

max_voc = len(vocablary) + 1

vectorize_layer = Dataset.vectorization(descriptions, max_voc, max_sen)

batch_size = 30

steps = len(training_description) / batch_size

model_file = "Outputs\\Model.h5"

if os.path.isfile(model_file) == False:

    Model.fit_model(training_description, test_description, steps, batch_size, features, vectorize_layer, max_sen, max_voc)

model = keras.models.load_model(model_file)

Model.test_evaluate(model, test_description, steps, batch_size, features, vectorize_layer, max_sen, max_voc)
#%%
def test(imgName, image_path):
    print("Image no " + str(no) +": " + imgName)

    img = Image.open(image_path)

    plt.figure(facecolor='#1f1f1f')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    print("-"*33 + "Actual Captions" + "-"*33)

    for caption in descriptions[imgName]:
        print(caption)

    image = Model.extractingFeatureOneImage(image_path)

    result = Model.predict_caption(image, model, vectorize_layer, max_sen)
    print("\n" + "-"*32 + "Predicted Caption" + "-"*32)
    print(result)
    print("="*81 + "\n")

print("Testing some samples...")
print("At First, testing some samples from training dataset...")
print("="*81)
loc = "Test Samples\\Training"
no = 0
for imgName in os.listdir(loc):
    no += 1
    image_path = os.path.join(loc, imgName)
    test(imgName, image_path)

print("Finally, testing some samples from testing dataset...")
print("="*81)
loc = "Test Samples\\Test"
no = 0
for imgName in os.listdir(loc):
    no += 1
    image_path = os.path.join(loc, imgName)
    test(imgName, image_path)
#%%
Model.model_validate(model, features, test_description, vectorize_layer, max_sen)
# %%