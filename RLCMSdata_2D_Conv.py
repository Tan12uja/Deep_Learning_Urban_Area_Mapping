# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:22:37 2021

@author: tanuj
"""

import ee #https://developers.google.com/earth-engine/guides/python_install
ee.Authenticate()

# Initialize the Earth Engine module.
ee.Initialize()

print(ee.__version__)


data = ee.FeatureCollection('users/tanujashrestha/Built_Area_Mapping/Built_non_built_balanced_data_from_RLCMS')

# set rice as 1
builtup = data.filter(ee.Filter.eq('class',0))

# set other class as 0
others = data.filter(ee.Filter.eq('class',1))


# set the pixel size
# note this is a 1 by 1 pixel patch
xs = 1
myList = ee.List.repeat(1, xs)
myLists = ee.List.repeat(myList, xs)
kernel = ee.Kernel.fixed(xs, xs,myLists)

# Cloud Storage bucket into which training, testing and validation 
# datasets will be written.  You must be able to write into this bucket.
#folder = "Nepal_cambodia"

folder = "Nepal" #this is the drive folder

# specify year
year = 2018
image = ee.Image("projects/servir-hkh/RLCMS/NEP/composites/2018").divide(10000);


## Exporting data to drive

trainData = builtup.merge(others).randomColumn("random")

# create the neighborhoodarray 
neighborhood = image.unmask(0).neighborhoodToArray(kernel);

# get the training data from 2018
#trainData = trainingData.filter(ee.Filter.eq("year",year))

# 70% for training, 20% for testing and 10% for validation 
training = trainData.filter(ee.Filter.lt("random",0.7))
testing = trainData.filter(ee.Filter.gt("random",0.7)).filter(ee.Filter.lt("random",0.9))
validation = trainData.filter(ee.Filter.gt("random",0.9))
    
# sample the image    
trainingSample= neighborhood.sampleRegions(collection=training,scale=30,tileScale=16)
testingSample= neighborhood.sampleRegions(collection=testing,scale=30,tileScale=16)
validationSample= neighborhood.sampleRegions(collection=validation,scale=30,tileScale=16)


year = '2020'


featureNames = ["blue","p20_blue","p80_blue","green","p80_green","nir","p20_nir","p80_nir",
                "red","p20_red","p80_red","swir1","p20_swir1","p80_swir1","swir2",
								        "p20_swir2","p80_swir2","class"]

# create an export task for training, testing and validation
trainingTask = ee.batch.Export.table.toDrive(collection=trainingSample,\
							description="train_Feb_2"+str(year),
              folder = folder, 
							fileNamePrefix= "train_Feb_2_2021"+str(year),
							fileFormat='TFRecord', 
							selectors=featureNames) 

testingTask = ee.batch.Export.table.toDrive(collection=testingSample,\
							description="test_Feb_2"+str(year), 
              folder = folder,
							fileNamePrefix="test_Feb_2_2021"+str(year), 
							fileFormat='TFRecord', 
							selectors=featureNames)
							
validationTask = ee.batch.Export.table.toDrive(collection=validationSample,\
							description="val_Feb_2"+str(year), 
              folder = folder,
							fileNamePrefix="val_Feb_2_2021"+str(year), 
							fileFormat='TFRecord', 
							selectors=featureNames)		
    
    
# Start the tasks
trainingTask.start()
testingTask.start()
validationTask.start()	

# model training and fitting
import tensorflow as tf
import numpy as np
from tensorflow import keras
import datetime
import subprocess
from tensorflow.keras import backend as K


#tf.enable_eager_execution()

# List of fixed-length features, all of which are float32.
size = 1

# How many classes there are in the model.
nClasses = 2

shuffle_size = 100
batch_size = 10



featureNames = ["blue","p20_blue","p80_blue","green","p80_green","nir","p20_nir","p80_nir",
                "red","p20_red","p80_red","swir1","p20_swir1","p80_swir1","swir2",
								        "p20_swir2","p80_swir2","class"]

label = "class"
# bands = ["blue","p20_blue","p80_blue","green","p20_blue","p80_green","nir","p20_nir","p80_nir",
#                 "red","p20_red","p80_red","swir1","p20_swir1","p80_swir1","swir2",
# 								        "p20_swir2","p80_swir2"]

bands = ["blue","p20_blue","p80_blue","green","p80_green","nir","p20_nir","p80_nir",
                "red","p20_red","p80_red","swir1","p20_swir1","p80_swir1","swir2",
								        "p20_swir2","p80_swir2"]


#bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

bands = ['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8','B9','B10','B11']

# sort and print the bands to ensure the band order
bands = sorted(bands)
print(bands)

# # get the location of all files
# calibrationData = '/content/mnt/My Drive/cambodia/training2018.tfrecord.gz'
# testingData = '/content/mnt/My Drive/cambodia/testing2018.tfrecord.gz'
# validationData = '/content/mnt/My Drive/cambodia/validation2018.tfrecord.gz'

# get the location of all files (from the drive)
calibrationData = '/content/mnt/My Drive/Nepal/training_latest_tanuja2018.tfrecord.gz'
testingData = '/content/mnt/My Drive/Nepal/testing_latest_tanuja2018.tfrecord.gz'
validationData = '/content/mnt/My Drive/Nepal/validation_latest_tanuja2018.tfrecord.gz'


# get the location of all files (from the computer)
calibrationData = 'train_Feb_2_20212020.tfrecord.gz'
testingData = 'test_Feb_2_20212020.tfrecord.gz'
validationData = 'val_Feb_2_20212020.tfrecord.gz'


## Create a dataset from the TFRecord file in Cloud Storage.
trainDataset = tf.data.TFRecordDataset(calibrationData, compression_type='GZIP')
testDataset = tf.data.TFRecordDataset(testingData, compression_type='GZIP')
validationDataset = tf.data.TFRecordDataset(validationData, compression_type='GZIP')



#it = iter(trainDataset)
# get length of input array and make list
l = len(bands)
featureNames = list(bands)
featureNames.append(label)


print("featurenames",featureNames)

columns = [
  tf.io.FixedLenFeature(shape=[size,size], dtype=tf.float32) for k in featureNames
]

## Dictionary with names as keys, features as values.
featuresDict = dict(zip(featureNames, columns))

print(featuresDict.values) #see this part on another colab notebook

def parse_tfrecord(example_proto):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.
  
  Returns: 
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
  labels = parsed_features.pop(label)
  return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
trainDataset= trainDataset.map(parse_tfrecord)
testDataset = testDataset.map(parse_tfrecord)
validationDataset = validationDataset.map(parse_tfrecord)

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.
def toTuple(Dict, label):
  return tf.transpose(list(Dict.values())), tf.one_hot(indices=label, depth=nClasses)



def to_tuple(inputs, label):
  return (tf.transpose(list(inputs.values())),
          tf.one_hot(indices=label, depth=N_CLASSES))


# Repeat the input dataset as many times as necessary in batches.
trainDataset = trainDataset.map(toTuple).shuffle(shuffle_size,reshuffle_each_iteration=True).batch(batch_size).repeat()
testDataset = testDataset.map(toTuple).shuffle(shuffle_size).batch(batch_size).repeat()
validationDataset = validationDataset.map(toTuple).shuffle(shuffle_size).batch(batch_size).repeat()



# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((size, size, l,)),
  tf.keras.layers.Conv2D(64, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.50),
  tf.keras.layers.Conv2D(32, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Conv2D(16, (1, 1), activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.15),
  tf.keras.layers.Conv2D(2, (1, 1), activation=tf.nn.softmax)
])

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(64, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(N_CLASSES, activation=tf.nn.softmax)
# ])


# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


#early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')


# Fit the model to the training data.
# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
training = model.fit(x=trainDataset,\
					 epochs=5,\
					 steps_per_epoch=100,\
					 validation_data = testDataset,\
					 validation_steps = 10,\
					 callbacks=[early_stop])
    
history = model.fit(x=trainDataset,\
					 epochs=5,\
					 steps_per_epoch=5,\
					 validation_data = testDataset,\
					 validation_steps = 10)

%pylab inline
#to see the model learning curves 
def summarize_diagnostics(training):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    line1, =pyplot.plot(history.history['loss'], color='blue', label='train')
    line2, =pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend(handles=[line1], loc='upper right')
    pyplot.legend(handles=[line2], loc='upper left')
    first_legend = pyplot.legend(handles=[line1], loc='upper right')
    second_legend = pyplot.legend(handles=[line2], loc='upper left')
    pyplot.gca().add_artist(first_legend)
    pyplot.gca().add_artist(second_legend)

    	# plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.subplots_adjust(hspace=0.5)
    pyplot.ylabel('*100 (%)')
    pyplot.xlabel('Number of iterations (Epoch)')
    #pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #filename = sys.argv[0].split('/')[-1]
    #pyplot.savefig(filename + '_plot.png')
    pyplot.savefig( '2D_Conv_.png')
    pyplot.close()

summarize_diagnostics(training)  
