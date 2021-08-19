import tensorflow as tf
import os
from matplotlib import pyplot as plt
import datetime

def allocateGPU():
    # Allocating the minimum amount of memory for the gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def initCheckpoint(checkpoint_dir,unet,unet_optimizer):
    # Directory to save checkpoints in
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(unet_optimizer=unet_optimizer,                               
                                 unet=unet)

    return checkpoint, checkpoint_prefix

def initLog(log_dir):
    # Log writer
    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return summary_writer

# Function to plot input,target,prediction images
def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [tf.image.rot90(test_input[0]), tf.image.rot90(tar[0]), tf.image.rot90(prediction[0])]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('Pictures/Epoch {}.png'.format(epoch))
    # plt.show()
    # plt.close()
