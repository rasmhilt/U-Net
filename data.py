import numpy as np
import h5py
import tensorflow as tf
import random
import math

# For randomizing indexes for train and validation lists.
# For example if datasize is 10 then the corresponding list
# is [1,...,10]. Split using validPer = 0.2 for example 
# to [1,...,8] and [9,10] and those indexes are then randomized. 
def indexes_random(dataSize,validPer,BATCH_SIZE):
    middleInd = math.ceil((1-validPer)*dataSize)
    train_list = list(range(1,int(middleInd)+1))
    random.shuffle(train_list)
    valid_list = list(range(int(middleInd)+1,dataSize+1))
    random.shuffle(valid_list)
    
    return train_list, valid_list

# For converting numpy arrays to tensors. Using float32 because
# of single precision used in MATLAB. Can be changed. 
def convert_to_tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

# For scaling a picture to the range of [0,1]
def minmaxScaler(x):
    return (x-np.min(x)) / (np.max(x) - np.min(x))

# If one wants to L2-normalize the dataset.
def normalize_with_moments(x, axes=[0, 1, 2], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon)
    return x_normed

# Input image to be in format (OUTPUT_CHANNELS,IMG_WIDTH,IMG_HEIGHT)
# Below changed to (IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS). So if different
# input format then change the code (np.moveaxis part).
# Change data_name depending on which name the data is saved under in MATLAB.
def loadBatch(featureName,labelName,BATCH_SIZE,n,valPer,indexList,training):
    data_name = 'mesh_cart'
    listSize = np.size(indexList)

    # In this part taking a batch of a given randomized index list. Depends on
    # whether taking from a training or a validation list. If the last batch to be taken
    # would result in going over the boundaries the batch size will just be smaller. For
    # example with a list [1,...,10] and batch size 8 the second batch would go over, but
    # this in this case this results in the last (second) batch being [9,10].
    if training == True:
        if listSize - (n+1)*BATCH_SIZE > 0:
            startInd = (n*BATCH_SIZE)
            endInd = (n+1)*BATCH_SIZE
            SET_BATCH_SIZE = BATCH_SIZE
        elif listSize - (n+1)*BATCH_SIZE <= 0:
            if abs(listSize - (n+1)*BATCH_SIZE) < BATCH_SIZE:
                startInd = (n*BATCH_SIZE)
                endInd = listSize
                SET_BATCH_SIZE = endInd-startInd
            else:
                print('Error. Outside of dataset size...')
    elif training == False:
        if listSize - (n+1)*BATCH_SIZE > 0:
            startInd = (n*BATCH_SIZE)
            endInd = (n+1)*BATCH_SIZE
            SET_BATCH_SIZE = BATCH_SIZE
            
        elif listSize - (n+1)*BATCH_SIZE <= 0:
            if abs(listSize - (n+1)*BATCH_SIZE) < BATCH_SIZE:
                startInd = (n*BATCH_SIZE)
                endInd = listSize
                SET_BATCH_SIZE = endInd-startInd
            else:
                print('Error. Outside of dataset size...')

    indexes = indexList[startInd:endInd]
    #print('Taking files {}'.format(indexes))

    # Read individual files, reformat them, concenate them, transform to tensors,
    # add a None dimension (None,IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS) and then shuffle
    # and finally make it to be a tensorflow databatch.
    features = h5py.File(featureName + '_' + str(indexes[0]) + '.mat', 'r')
    features = np.moveaxis(np.array(features.get(data_name)), 2, 0)
    labels = h5py.File(labelName + '_' + str(indexes[0]) + '.mat', 'r')
    labels = np.moveaxis(np.array(labels.get(data_name)), 2, 0)
    features = minmaxScaler(features)
    labels = minmaxScaler(labels)

    for i in range(1, SET_BATCH_SIZE):
        feature = h5py.File(featureName + '_' + str(indexes[i]) + '.mat', 'r')
        label = h5py.File(labelName + '_' + str(indexes[i]) + '.mat', 'r')
        # To numpy arrays
        feature = np.moveaxis(np.array(feature.get(data_name)), 2, 0)
        label = np.moveaxis(np.array(label.get(data_name)), 2, 0)
        # To [0,1]
        feature = minmaxScaler(feature)
        label = minmaxScaler(label)
        # Concatenating
        features = np.concatenate((features,feature),axis=0)
        labels = np.concatenate((labels,label),axis=0)
    # To tensors
    features = convert_to_tensor(features)
    labels = convert_to_tensor(labels)
    # Add channels (None)
    features = tf.expand_dims(features, -1)
    labels = tf.expand_dims(labels, -1)
    # To dataset
    dataBatch = tf.data.Dataset.from_tensor_slices((features, labels))
    # Shuffle even more. (Can be removed)
    dataBatch = dataBatch.shuffle(SET_BATCH_SIZE)
    # Make it a batch dataset
    dataBatch = dataBatch.batch(SET_BATCH_SIZE)

    return dataBatch

# This is just for loading test data.
def loadTestData(featureName,labelName,dataSize):
    data_name = 'mesh_cart'
    # Read individual files and concenate them to one of batch size
    features = h5py.File(featureName + '_' + str(1) + '.mat', 'r')
    features = np.moveaxis(np.array(features.get(data_name)), 2, 0)
    labels = h5py.File(labelName + '_' + str(1) + '.mat', 'r')
    labels = np.moveaxis(np.array(labels.get(data_name)), 2, 0)
    features = minmaxScaler(features)
    labels = minmaxScaler(labels)

    for i in range(2, dataSize+1):
        feature = h5py.File(featureName + '_' + str(i) + '.mat', 'r')
        label = h5py.File(labelName + '_' + str(i) + '.mat', 'r')
        # To numpy arrays
        feature = np.moveaxis(np.array(feature.get(data_name)), 2, 0)
        label = np.moveaxis(np.array(label.get(data_name)), 2, 0)
        # To [0,1]
        feature = minmaxScaler(feature)
        label = minmaxScaler(label)
        # Concatenating
        features = np.concatenate((features,feature),axis=0)
        labels = np.concatenate((labels,label),axis=0)
        
        
    # To tensors and normalization
    features = convert_to_tensor(features)
    labels = convert_to_tensor(labels)
    # labels = normalize_with_moments(labels)
    # features = normalize_with_moments(features)
    # Add channels
    features = tf.expand_dims(features, -1)
    labels = tf.expand_dims(labels, -1)
    # To dataset
    dataBatch = tf.data.Dataset.from_tensor_slices((features, labels))
    # Make it a batch dataset
    dataBatch = dataBatch.batch(1)

    return dataBatch