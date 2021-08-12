import datetime
from train import *


# Set the numerical parameters here
dataSize = 16000         # Size of the dataset
validPer = 0.2           # Percentile of data to be used for validation
BATCH_SIZE = 16          # Amount of images sent to training at once (Bigger = faster runs, but more RAM needed)
BUFFER_SIZE = BATCH_SIZE # For shuffling the batches. Set size equal or bigger than batch size (Unused)
IMG_WIDTH = 128          # Image width
IMG_HEIGHT = 128         # Image height
OUTPUT_CHANNELS = 1      # Output channels. (f.ex. 1 for conductivity, 3 for rgb images etc.)
EPOCHS = 300             # Number of training epochs

# Directories for logs and checkpoints.
log_dir = "logs/"
checkpoint_dir = './training_checkpoints/' + datetime.date.today().strftime("%Y%m%d") + '/' + str(dataSize) 

# Locations of feature and label datas.
features_loc =  + '/rec_' + str(dataSize) # In format ..Folder/rec_dataSize
labels_loc =  + '/true_' + str(dataSize) # In format ..Folder/true_dataSize

# Choose optimizer, loss, metrics and U-Net.
select_unet = 'UNet'        # Select type of UNet (UNet,UNetPlus,UNetMinus,UNetMinustwo. More or less layers.)
select_optimizer = 'Adam'   # Select the optimizer 
learning_rate = 1e-4        # Learning rate for the optimizer
select_loss = 'l2norm'      # Type of loss. (L2 or normed L2)
select_train_metric = 'BinaryCrossentropy' # Train metric
select_test_metric = 'BinaryCrossentropy'  # Validation metric

# Train the U-Net
fit(dataSize,validPer,BATCH_SIZE,BUFFER_SIZE,IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS,EPOCHS,
log_dir,checkpoint_dir,features_loc,labels_loc,select_unet,select_optimizer,learning_rate,
select_loss,select_train_metric,select_test_metric)