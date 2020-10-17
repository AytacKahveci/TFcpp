#!/usr/bin/env python
import os
import sys 
import argparse 
from datetime import datetime

import numpy as np

import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers import ZeroPadding2D

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K



def build_classifier(img_shape = 150, num_categories = 2):
  classifier = Sequential()

  # First convolutional layer
  classifier.add(Conv2D(  filters=32, 
                          kernel_size=(2,2),
                          padding="same",
                          data_format="channels_last",
                          input_shape=(img_shape, img_shape, 3),
                          activation="relu",
                          name="Conv1_layer"))
  
  # Pooling
  classifier.add(MaxPooling2D(pool_size=(2, 2), name="Pool1_layer"))

  # Second convolutional layer
  classifier.add(Conv2D(filters=64,
                        kernel_size=(2,2),
                        activation="relu",
                        padding="same",
                        name="Conv2_layer"))

  # Pooling
  classifier.add(MaxPooling2D(pool_size=(2,2), name="Pool2_layer"))

  # Second convolutional layer
  classifier.add(Conv2D(filters=64,
                        kernel_size=(2,2),
                        activation="relu",
                        padding="same",
                        name="Conv3_layer"))

  # Pooling
  classifier.add(MaxPooling2D(pool_size=(2,2), name="Pool3_layer"))

  # Flatten
  classifier.add(Flatten(name="flat"))

  # Fully connected NN
  classifier.add(Dense(units=256, activation="relu", name="fc256"))
  classifier.add(Dense(units=num_categories, activation="softmax", name="fcfinal"))

  # Compile the CNN
  classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

  return classifier


def train_model(classifier, trainloc, test_loc, img_shape, out_dir=".", batch_size=20, num_epochs=20):
  train_datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=1.5,
                                     zoom_range=0.5)

  test_datagen = ImageDataGenerator(rescale=1./255)

  training_set = train_datagen.flow_from_directory(trainloc,
                                                   target_size=(img_shape, img_shape),
                                                   batch_size=batch_size,
                                                   class_mode="categorical")

  test_set = test_datagen.flow_from_directory(test_loc,
                                              target_size=(img_shape, img_shape),
                                              batch_size=batch_size,
                                              class_mode="categorical")

  out_dir = os.path.abspath(out_dir)
  now = datetime.now().strftime('k2tf-%Y%m%d%H%M%S')
  now = os.path.join(out_dir, now)
  os.makedirs(now)

  # Create our callbacks
  savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
  checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_best_only=True)
  fout = open( os.path.join(now, 'indices.txt'), 'wt' )
  fout.write( str(training_set.class_indices) + '\n' )

  # train the model on the new data for a few epochs
  classifier.fit_generator(training_set,
                            steps_per_epoch = len(training_set.filenames)//batch_size,
                            epochs = num_epochs,
                            validation_data = test_set,
                            validation_steps = len(test_set.filenames)//batch_size,
                            workers=32, 
                            max_queue_size=32,
                            callbacks=[checkpointer]
                            )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # Required
  parser.add_argument('--test', dest='test', default="/media/aytac/Tank/machine-learning/TFcpp/data/cats_and_dogs_small/cats_and_dogs_small/test", required=False, help='(REQUIRED) location of the test directory')
  parser.add_argument('--train', dest='train', default="/media/aytac/Tank/machine-learning/TFcpp/data/cats_and_dogs_small/cats_and_dogs_small/train", required=False, help='(REQUIRED) location of the test directory')
  parser.add_argument('--cats', '-c', dest='categories', type=int, required=True, help='(REQUIRED) number of categories for the model to learn')
  # Optional
  parser.add_argument('--output', '-o', dest='output', default='./', required=False, help='location of the output directory (default:./)')
  parser.add_argument('--batch', '-b', dest='batch', default=20, type=int, required=False, help='batch size (default:32)')
  parser.add_argument('--epochs', '-e', dest='epochs', default=20, type=int, required=False, help='number of epochs to run (default:30)')
  parser.add_argument('--shape','-s', dest='shape', default=150, type=int, required=False, help='The shape of the image, single dimension will be applied to height and width (default:128)')
  
  args = parser.parse_args()
  
  classifier = build_classifier( args.shape, args.categories)
  train_model( classifier, args.train, args.test, args.shape, args.output, batch_size=args.batch, num_epochs=args.epochs )