# TFCpp

Tensorflow example implementation with C++ to distinguish dogs from cat. This implementation was developed based on the following projects; [[1]](http://www.bitbionic.com/2017/08/18/run-your-keras-models-in-c-tensorflow/), [[2]](https://towardsdatascience.com/creating-a-tensorflow-cnn-in-c-part-2-eea0de9dcada).

# Install Build tools

## Install Bazel
  Instructions can be found its documentation [page](https://docs.bazel.build/versions/master/install-ubuntu.html)
  ```bash
  chmod +x bazel-3.5.1-installer-linux-x86_64.sh
  ./bazel-3.5.1-installer-linux-x86_64.sh --user
  ```

  Check the bazel version
  ```bash
  bazel version
  ```

## Build Tensorflow
  Clone the repository
  ```bash
  git clone https://github.com/tensorflow/tensorflow.git
  ```

  Build
  ```bash
  cd tensorflow
  ./configure
  bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
  bazel build -c opt --verbose_failures //tensorflow:libtensorflow_framework.so
  bazel build //tensorflow:install_headers
  ```

## Install Keras and Tensorflow
  Create virtual environment
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```

  Install packages in virtual environment
  ```bash
  pip install keras==2.1.2
  pip install tensorflow==1.14.0
  ```

## Download Keras to Tensorflow Script
  Clone the repository
  ```bash
  git clone https://github.com/bitbionic/keras-to-tensorflow.git
  ```

# Download dataset
  Dataset can be download from [here](https://s3.amazonaws.com/img-datasets/cats_and_dogs_small.zip)

# Train the model
  Train:
  ```bash
  python TFcpp/scripts/train.py
  ```

# Convert Keras to Tensorflow
  ```bash
  python keras_to_tensorflow.py --input_model=PATH_TO_MODEL.h5 --output_model=PATH_TO_MODEL.pb --save_graph_def=true
  ```

# Test the model
  Test:
  ```bash
  python TFcpp/scripts/test.py -m PATH_TO_MODEL.pb -i PATH_TO_TEST_IMAGE
  ```

# Compile and run the project
  Copy .so files to lib and headers to include folder.
  ```bash
  cp -R tensorflow/tensor/bazel-bin/tensorflow/libtensorflow_cc.so TFcpp/lib
  cp -R tensorflow/tensor/bazel-bin/tensorflow/libtensorflow_framework.so TFcpp/lib 
  cp -R tensorflow/bazel-bin/tensorflow TFcpp/include
  cp -R tensorflow/third_party/eigen3/Eigen TFcpp/include  
  ```

  Compile the project
  ```bash
  cd TFCpp
  mkdir build
  cd build
  cmake ..
  make
  ```

  Test the model
  ```bash
  cd TFcpp/build
  ./model_run
  ```