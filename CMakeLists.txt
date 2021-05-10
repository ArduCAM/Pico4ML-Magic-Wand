cmake_minimum_required(VERSION 3.13.1)

set(MODEL_FOLDER .)
set(EI_SDK_FOLDER edge-impulse-sdk)

include(pico_sdk_import.cmake)

project(magic_wand C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

pico_sdk_init()

add_subdirectory(ICM20948)
include_directories(ICM20948)


add_executable(magic_wand
  source/main.cpp
  source/ei_classifier_porting.cpp
  source/imu_provider.h
  source/rasterize_stroke.cpp
  source/rasterize_stroke.h
  )

include(${MODEL_FOLDER}/edge-impulse-sdk/cmake/utils.cmake)

target_link_libraries(magic_wand pico_stdlib ICM20948)

# enable usb output, disable uart output
pico_enable_stdio_usb(magic_wand 1)
pico_enable_stdio_uart(magic_wand 0)

target_include_directories(magic_wand PRIVATE
  ${MODEL_FOLDER}
  ${MODEL_FOLDER}/classifer
  ${MODEL_FOLDER}/tflite-model
  ${MODEL_FOLDER}/model-parameters
  )

target_include_directories(magic_wand PRIVATE
  ${EI_SDK_FOLDER}
  ${EI_SDK_FOLDER}/third_party/ruy
  ${EI_SDK_FOLDER}/third_party/gemmlowp
  ${EI_SDK_FOLDER}/third_party/flatbuffers/include
  ${EI_SDK_FOLDER}/third_party
  ${EI_SDK_FOLDER}/tensorflow
  ${EI_SDK_FOLDER}/dsp
  ${EI_SDK_FOLDER}/classifier
  ${EI_SDK_FOLDER}/anomaly
  ${EI_SDK_FOLDER}/CMSIS/NN/Include
  ${EI_SDK_FOLDER}/CMSIS/DSP/PrivateInclude
  ${EI_SDK_FOLDER}/CMSIS/DSP/Include
  ${EI_SDK_FOLDER}/CMSIS/Core/Include
  )

include_directories(${INCLUDES})

# find model source files
RECURSIVE_FIND_FILE(MODEL_FILES "${MODEL_FOLDER}/tflite-model" "*.cpp")
RECURSIVE_FIND_FILE(SOURCE_FILES "${EI_SDK_FOLDER}" "*.cpp")
RECURSIVE_FIND_FILE(CC_FILES "${EI_SDK_FOLDER}" "*.cc")
RECURSIVE_FIND_FILE(S_FILES "${EI_SDK_FOLDER}" "*.s")
RECURSIVE_FIND_FILE(C_FILES "${EI_SDK_FOLDER}" "*.c")
list(APPEND SOURCE_FILES ${S_FILES})
list(APPEND SOURCE_FILES ${C_FILES})
list(APPEND SOURCE_FILES ${CC_FILES})
list(APPEND SOURCE_FILES ${MODEL_FILES})

# add all sources to the project
target_sources(magic_wand PRIVATE ${SOURCE_FILES})

pico_add_extra_outputs(magic_wand)
