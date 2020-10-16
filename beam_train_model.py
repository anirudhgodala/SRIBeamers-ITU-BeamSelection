import  sys
import numpy as np
#import matplotlib.pyplot as plt
import math
from beam_train_frontend import *
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Conv3D,Reshape,ConvLSTM2D,Input,concatenate,SpatialDropout3D,Dropout
from keras.metrics import top_k_categorical_accuracy
#from sklearn import preprocessing
import argparse
parser = argparse.ArgumentParser(description='Train ML model')
parser.add_argument('--path',type=str,
                    help='Add appropriate path for adding data')
args = parser.parse_args()
Path=args.path
#Path = '/gpfs-volume/Beam_Selection/baseline_data/'
lidar_path_train=Path+"/lidar_input/lidar_train.npz"
lidar_path_validate=Path+"/lidar_input/lidar_validation.npz"
#lidar_path_test=Path+"/lidar_input/lidar_test.npz"

coord_path_train=Path+"/coord_input/coord_train.npz"
coord_path_validate=Path+"/coord_input/coord_validation.npz"
#coord_path_test=Path+"/coord_input/coord_test.npz"

image_path_train=Path+"/image_v2_input/img_input_train_20.npz"
image_path_validate=Path+"/image_v2_input/img_input_validation_20.npz"
#image_path_test=Path+"/image_input/image_test.npz"

beam_path_train=Path+"/beam_output/beams_output_train.npz"
beam_path_validate=Path+"/beam_output/beams_output_validation.npz"
#beam_path_test=Path+"/beam_output/beam_output_test.npz"

lidar_data_train = np.load(lidar_path_train);
lidar_data_validate=np.load(lidar_path_validate);
#lidar_data_test=np.load(lidar_path_test);

coord_data_train = np.load(coord_path_train);
coord_data_validate=np.load(coord_path_validate);
#coord_data_test = np.load(coord_path_test);

image_data_train = np.load(image_path_train);
image_data_validate=np.load(image_path_validate);
#image_data_test = np.load(image_path_test);

beam_output_train = np.load(beam_path_train);
beam_output_validate=np.load(beam_path_validate);
#beam_output_test = np.load(beam_path_test);

lst=lidar_data_train.files
for item in lst:
    lidar_train=lidar_data_train[item]
print(lidar_train.shape)
lst=lidar_data_validate.files
for item in lst:
    lidar_validate=lidar_data_validate[item]
print(lidar_validate.shape)


lst=coord_data_train.files
for item in lst:
    coord_train=coord_data_train[item]
print(coord_train.shape)
lst=coord_data_validate.files
for item in lst:
    coord_validate=coord_data_validate[item]
print(coord_validate.shape)


lst=image_data_train.files
for item in lst:
    image_train=image_data_train[item]
print(image_train.shape)
lst=image_data_validate.files
for item in lst:
    image_validate=image_data_validate[item]
print(image_validate.shape)


lst=beam_output_train.files
for item in lst:
    beam_train=beam_output_train[item]
print(beam_train.shape)
lst=beam_output_validate.files
for item in lst:
    beam_validate=beam_output_validate[item]
print(beam_validate.shape)


lidar_model_in,image_model_in,coord_model_in,beam_model_out = featureGen_train(lidar_train,image_train,coord_train,beam_train)
lidar_model_val,image_model_val,coord_model_val,beam_model_val = featureGen_train(lidar_validate,image_validate,coord_validate,beam_validate)

lidar_final_in = np.concatenate((lidar_model_in,lidar_model_val),axis = 0)
beam_final_out = np.concatenate((beam_model_out,beam_model_val),axis = 0)
def top_10_accuracy(y_true,y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=10)

model = Sequential()
model.add(Conv3D(20,kernel_size = (5,11,3),input_shape=(20,200,10,1),strides = (1,2,1)))
model.add(SpatialDropout3D(rate = 0.6))
model.add(Conv3D(20,kernel_size = (5,11,3),strides = (1,2,1)))
model.add(SpatialDropout3D(rate = 0.4))        
model.add(Flatten())
model.add(Dense(3000, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.0001),activity_regularizer=tf.keras.regularizers.l2(1e-6)))
model.add(Dropout(rate = 0.3))
model.add(Dense(256, activation = tf.nn.softmax,kernel_regularizer=tf.keras.regularizers.l1(0.0001),activity_regularizer=tf.keras.regularizers.l2(1e-6)))
model.compile(optimizer='adagrad',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(lidar_final_in , beam_final_out , epochs = 50, validation_data= (lidar_model_val,beam_model_val))


# After training the model, its parameters are saved
model.save_weights('my_model_weights_50_drop.h5') 
model_json = model.to_json()
with open('my_model_50_drop.json', "w") as json_file:
    json_file.write(model_json)


    
test_result = model.predict(lidar_model_val)

test_result = np.argsort(-test_result,axis = 1)

np.where(test_result[0,:] == beam_model_val[0])[0][0]

res_val_raw = np.zeros((beam_model_val.shape[0]))

for i in range(beam_model_val.shape[0]):
    res_val_raw[i] = np.where(test_result[i,:] == beam_model_val[i])[0][0]

print('top 10 validation accuracy : ',np.where(res_val_raw  < 10)[0].shape[0] *100/1960)