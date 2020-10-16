import  sys
import numpy as np
#import matplotlib.pyplot as plt
import math
from beam_test_frontend import *
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Conv3D,Reshape,ConvLSTM2D,Input,concatenate,SpatialDropout3D,Dropout
from keras.metrics import top_k_categorical_accuracy
#from sklearn import preprocessing
import argparse
parser = argparse.ArgumentParser(description='Test ML model')
parser.add_argument('--path',type=str,
                    help='Add appropriate path for adding data')
args = parser.parse_args()
Path=args.path
#Path = '/gpfs-volume/Beam_Selection/baseline_data/'

lidar_path_test=Path+"/lidar_input/lidar_test.npz"

image_path_test=Path+"/image_v2_input/img_input_test_20.npz"



lidar_data_test=np.load(lidar_path_test);

image_data_test = np.load(image_path_test);


lst=lidar_data_test.files
for item in lst:
    lidar_test=lidar_data_test[item]
print(lidar_test.shape)


lst=image_data_test.files
for item in lst:
    image_test=image_data_test[item]
print(image_test.shape)




lidar_model_in,image_model_in = featureGen_test(lidar_test,image_test)


from tensorflow.keras.models import model_from_json
with open('my_model_50_drop.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.compile(optimizer='adagrad',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('my_model_weights_50_drop.h5')

Y_test = model.predict(lidar_model_in)

np.savetxt('beam_test_pred.csv', Y_test, delimiter=',')
