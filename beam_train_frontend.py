import numpy as np
import math
def featureGen_train(lidar_input,image_input,coord_in,beam_input):
    
    # change tx position value to -127
    ind = np.where(lidar_input == -1)
    for i in range(ind[0].shape[0]):
        lidar_input[ind[0][i],ind[1][i],ind[2][i],ind[3][i]] = 127
    # change tx position value to 127
    ind = np.where(lidar_input == -2)
    for i in range(ind[0].shape[0]):
        lidar_input[ind[0][i],ind[1][i],ind[2][i],ind[3][i]] = 127
    lidar_input_mod = np.reshape(lidar_input,(lidar_input.shape[0],lidar_input.shape[1],lidar_input.shape[2],lidar_input.shape[3],1))
    
    #image data
    image_input_mod = image_input[:,:,27:81,:]
    
    #beam_change
    
    beam_rsrp = np.zeros((beam_input.shape[0],beam_input.shape[1]*beam_input.shape[2]))
    for i in range(beam_input.shape[0]):
        count = 0
        for k in range(beam_input.shape[2]):
            for j in range(beam_input.shape[1]):
                beam_rsrp[i,count] = 10*math.log10(beam_input[i,j,k])
                count = count+1
    beam_rsrp_mod = np.argmax(beam_rsrp,axis = 1)
    print('train features done')
    return lidar_input_mod,image_input_mod,coord_in,beam_rsrp_mod
    