import tensorflow as tf
import keras

def chooseLoss(unet_output, target, chosen_loss):
    if chosen_loss == 'l2':
        return l2loss(unet_output, target)
    elif chosen_loss == 'l2norm':
        return norm_l2loss(unet_output, target)

# L2 loss function
def l2loss(unet_output, target):
    return tf.nn.l2_loss(unet_output-target)

# Normalized L2 loss function
def norm_l2loss(unet_output, target):
    return tf.nn.l2_loss(unet_output-target)/tf.nn.l2_loss(target)

def choose_metrics(train_name,test_name):
    if train_name == 'BinaryCrossentropy':
        train_acc_metric = keras.metrics.BinaryCrossentropy()
    elif train_name == 'MSE':
        train_acc_metric = keras.metrics.MeanSquaredError()

    if test_name == 'BinaryCrossentropy':
        test_acc_metric = keras.metrics.BinaryCrossentropy()
    elif test_name == 'MSE':
        test_acc_metric = keras.metrics.MeanSquaredError()

    return train_acc_metric,test_acc_metric

def choose_optimizer(optimizername,lr):
    if optimizername == 'Adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizername == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(lr)
    elif optimizername == 'SGD':
        optimizer = tf.keras.optimizers.SGD(lr)

    return optimizer