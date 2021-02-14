# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:34:03 2020

@author: tanuj
"""
import tensorflow as tf
import ee #https://developers.google.com/earth-engine/guides/python_install
ee.Authenticate()

# Initialize the Earth Engine module.
ee.Initialize()

print(ee.__version__)


# Use Landsat 8 surface reflectance data for predictors.
L8SR = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
# Use these bands for prediction.
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

LABEL_DATA = ee.FeatureCollection('users/tanujashrestha/new_polygons_02_02_2021')

LABEL = 'class'
#LABEL = 'landcover'
# Number of label values, i.e. number of classes in the classification.
#N_CLASSES = 3

N_CLASSES = 2

# These names are used to specify properties in the export of
# training/testing data and to define the mapping between names and data
# when reading into TensorFlow datasets.
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append(LABEL)


# Cloud masking function.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  return image.updateMask(mask).select(BANDS).divide(10000)

# The image input data is a 2018 cloud-masked median composite.
image = L8SR.filterDate('2018-01-01', '2018-12-31').map(maskL8sr).median()

# Sample the image at the points and add a random column.
sample = image.sampleRegions(
  collection=LABEL_DATA, properties=[LABEL], scale=30).randomColumn()


# Partition the sample approximately 70-30.
training = sample.filter(ee.Filter.lt('random', 0.7))
testing = sample.filter(ee.Filter.gte('random', 0.7))


# training = trainData.filter(ee.Filter.lt("random",0.7))
# testing = trainData.filter(ee.Filter.gt("random",0.7)).filter(ee.Filter.lt("random",0.9))
# validation = trainData.filter(ee.Filter.gt("random",0.9))


from pprint import pprint

# Print the first couple points to verify.
print({'training': training.first().getInfo()})
print({'testing': testing.first().getInfo()})


folder = 'cambodia' 
#export the training sample
trainingTask = ee.batch.Export.table.toDrive(collection=training,\
							description="train",
              folder = folder, 
							fileNamePrefix= "hello_train_new_polygons_02_02_2021",
							fileFormat='TFRecord', 
							selectors=FEATURE_NAMES) 

testingTask = ee.batch.Export.table.toDrive(collection=testing,\
							description="test", 
              folder = folder,
							fileNamePrefix="hello_test_new_polygons_02_02_2021", 
							fileFormat='TFRecord', 
							selectors=FEATURE_NAMES)
    
# Start the tasks.
trainingTask.start()
testingTask.start()

#the data was downloaded from the 
TRAIN_FILE_PATH = 'hello_train_new_polygons_02_02_2021.tfrecord.gz'
TEST_FILE_PATH = 'hello_test_new_polygons_02_02_2021.tfrecord.gz'

##################### latest data with approximately same amount of data for urban and non-urban

# Create a dataset from the TFRecord file in Cloud Storage.
train_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type='GZIP')
test_dataset = tf.data.TFRecordDataset(TRAIN_FILE_PATH, compression_type='GZIP')
# Print the first record to check.
print(iter(train_dataset).next())
print(iter(test_dataset).next())


# List of fixed-length features, all of which are float32.
columns = [
  tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES
]

# Dictionary with names as keys, features as values.
features_dict = dict(zip(FEATURE_NAMES, columns))

print(features_dict)


def parse_tfrecord(example_proto):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.

  Returns:
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, features_dict)
  labels = parsed_features.pop(LABEL)
  return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
train_parsed_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=5)
test_parsed_dataset = test_dataset.map(parse_tfrecord, num_parallel_calls = 5)

# Print the first parsed record to check.
print(iter(train_parsed_dataset).next())
print(iter(test_parsed_dataset).next())


from tensorflow import keras

# # Add NDVI.
# input_dataset = parsed_dataset.map(add_NDVI)

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.

def to_tuple(inputs, label):
  return (tf.transpose(list(inputs.values())),
          tf.one_hot(indices=label, depth=N_CLASSES))

# # Map the to_tuple function, shuffle and batch.
# input_dataset = input_dataset.map(to_tuple).batch(8)

# Map the to_tuple function, shuffle and batch.
train_dataset = train_parsed_dataset.map(to_tuple).batch(8)
test_dataset = test_parsed_dataset.map(to_tuple).batch(8)


train_dataset = train_parsed_dataset.map(to_tuple).batch(16)
test_dataset = test_parsed_dataset.map(to_tuple).batch(16)


## here find the number of training and testing samples and see what is going wrong

# Define the layers in the model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(N_CLASSES, activation=tf.nn.softmax)
])

# Compile the model with the specified loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0009),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data.
history = model.fit(x=train_dataset, epochs=40, validation_data = test_dataset, verbose=1)
history1 = model.fit(x=train_dataset, epochs=40, validation_data = test_dataset, verbose=1)

history1 = model.fit(x=train_dataset, epochs=40, verbose=1)

history2 = model.fit(x=train_dataset, epochs=40, verbose=1)

batch_size = 16
training = model.fit(x=train_dataset,\
					 epochs=5,\
					 steps_per_epoch=100,\
					 validation_data =  test_dataset,\
					 validation_steps = 10)
    
%pylab inline
plot(history.history['loss'],'x--', label='tr_loss')
plot(history.history['accuracy'], 'o--', label='tr_acc')
plot(history.history['val_loss'], '+--', label='va_loss')
plot(history.history['val_accuracy'], '*--', label='va_acc')


#to see the model learning curves 
def summarize_diagnostics(history):
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
    pyplot.savefig( 'new_polygons_02_02_2021).png')
    pyplot.close()

summarize_diagnostics(history)  
##this is practice 
dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 2, 100)).batch(10)

#Model saving for prediction
model.save('saved_model_new_polygons_02_02_2021/my_model') 

#Prediction part is carried out in google earth engine



