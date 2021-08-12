import tensorflow as tf
import time
import math
from model import *
from data import *
from scripts import *
from statistics import *

@tf.function
def test_step(input_batch,target_batch, unet, select_loss):
    val_logits = unet(input_batch,training=False)
    loss_value = chooseLoss(val_logits,target_batch, select_loss)
    
    return loss_value, val_logits

@tf.function
def train_step(input_batch, target_batch, epoch, unet, select_loss, unet_optimizer, train_acc_metric):
    with tf.GradientTape() as tape:
        unet_output = unet(input_batch, training=True)
        loss_value = chooseLoss(unet_output,target_batch, select_loss)
     
    unet_gradients = tape.gradient(loss_value,unet.trainable_variables)                                        
    return loss_value, unet_gradients, unet_output

def fit(dataSize,validPer,BATCH_SIZE,BUFFER_SIZE,IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS,epochs,
log_dir,checkpoint_dir,featureName,labelName,select_unet,select_optimizer,learning_rate,
select_loss,select_train_metric,select_test_metric):

    # Setting up GPU space. Comment if not using a GPU. (Runs with CPU if no suitable GPU)
    allocateGPU()

    # Initializing stuff
    unet = chooseUnet(select_unet,IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS)
    unet_optimizer = choose_optimizer(select_optimizer,learning_rate)
    (train_acc_metric,val_acc_metric) = choose_metrics(select_train_metric,select_test_metric)
    (checkpoint, checkpoint_prefix) = initCheckpoint(checkpoint_dir,unet,unet_optimizer)
    summary_writer = initLog(log_dir)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start = time.time()

        # These are the amount of loops (training and validation) needed for the given
        # data and batch sizes divided by percentage used for training and validating.
        n_train = math.ceil((1-validPer)*dataSize/BATCH_SIZE)
        n_valid = math.ceil((validPer)*dataSize/BATCH_SIZE)

        # Creating list of random indexes for training and validation datasets to iterate over
        (train_list,valid_list) = indexes_random(dataSize,validPer,BATCH_SIZE)

        # This can be used to save a random picture each epoch during the training process.
        # Alternatively can be also used in a notebook to watch the pictures to pop up. 
        # (by editing the generate_images function)
        validation_dataset = loadBatch(featureName,labelName,BATCH_SIZE,0,validPer,valid_list,training=False)
        for example_input, example_target in validation_dataset.take(1):
            generate_images(unet, example_input, example_target,epoch)
            print('Epoch {} of {} \n'.format(epoch,epochs))
        
        # The main training loop
        total_train_loss = 0
        for n in range(n_train):
            train_dataset = loadBatch(featureName,labelName,BATCH_SIZE,n,validPer,train_list,training=True)
 
            for (input_batch,target_batch) in train_dataset:
                (train_loss, unet_gradients, unet_output) = train_step(input_batch, target_batch, epoch, unet, select_loss, unet_optimizer, train_acc_metric)
                total_train_loss += train_loss
                unet_optimizer.apply_gradients(zip(unet_gradients,unet.trainable_variables))
                train_acc_metric.update_state(target_batch,unet_output)

            if n % 200 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (n, float(train_loss)))
                print("Seen so far: %d samples" % ((n + 1) * BATCH_SIZE))   
        total_train_loss /= n_train     
            
        # Saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print('Saving checkpoint...')

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        total_valid_loss = 0
        for n in range(n_valid):
            validation_dataset = loadBatch(featureName,labelName,BATCH_SIZE,n,validPer,valid_list,training=False)
            for x_batch_val, y_batch_val in validation_dataset:
                (valid_loss, val_logits) = test_step(x_batch_val, y_batch_val, unet, select_loss)
                total_valid_loss += valid_loss
                val_acc_metric.update_state(y_batch_val, val_logits)
        total_valid_loss /= n_valid

        # Reset validation metrics
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        # This writes to logs file that can be viewed with tensorboard.
        # One can add stuff to write if wanted.                              
        with summary_writer.as_default():
            tf.summary.scalar('train loss (one batch)', train_loss, step=epoch)
            tf.summary.scalar('valid loss (one batch)', valid_loss, step=epoch)
            tf.summary.scalar('train loss (total)', total_train_loss, step=epoch)
            tf.summary.scalar('valid loss (total)', total_valid_loss, step=epoch)
            tf.summary.scalar('train acc', train_acc, step=epoch)
            tf.summary.scalar('valid acc', val_acc, step=epoch)
            
        print("Validation acc: %.4f" % (float(val_acc),))

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
                                                        
    checkpoint.save(file_prefix=checkpoint_prefix)