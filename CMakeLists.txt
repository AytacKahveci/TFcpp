cmake_minimum_required(VERSION 3.1)
project(TFcpp)

include_directories(
  include
  include/tensorflow
)

link_directories(
  lib
)

add_executable(model_train src/main.cpp src/CatDogCNN.cpp)
target_link_libraries(model_train tensorflow_cc tensorflow_framework)

add_executable(model_run src/model_run.cpp src/CatDogCNN.cpp)
target_link_libraries(model_run tensorflow_cc tensorflow_framework)