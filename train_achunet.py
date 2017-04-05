
# coding: utf-8

# In[1]:

from create_achu_net import create_achu_net
from prep_data import read_video_to_frames, get_video_ready_for_training
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np


# In[2]:

def get_new_sample(i=1):
    X_train = get_video_ready_for_training(i=i, root = 'Left', output_resolution = (150,150))
    y_total = np.load('new_emotion_labels_27_2400')
    y_train = y_total[i]
    Y_train = np_utils.to_categorical(y_train, 7)
    return X_train, Y_train


# In[3]:

def train_network(n_timesteps=16, augment_data=True, nb_epochs=500, batch_size=64, loss='categorical_crossentropy', optim = 'adam', model_from = None, logger=True, lr_reduce=True, min_lr = 0.00001, metrics = ['accuracy']):
    nb_classes = 7
    X_train = get_video_ready_for_training(i=0, root = 'Left', output_resolution = (150,150))
    X_test = np.concatenate(get_video_ready_for_training(i=25, root = 'Left', output_resolution = (150,150)),get_video_ready_for_training(i=26, root = 'Left', output_resolution = (150,150)))
    y_total = np.load('new_emotion_labels_27_2400')
    y_test = np.concatenate(y_total[25], y_total[26])
    y_train = y_total[i]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #------------------------dataaugmentation---------------------------------------------------#

    if (augment_data):

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        print "X_training_shape: ",X_train.shape
        print "X_testing_shape: ",X_test.shape
        print "Y_training_shape: ",Y_train.shape
        print "training_shape: ",Y_test.shape
    if model_from:
        model = model_from
    else:
        model = create_achu_net(weights_path=None, img_chs = 3, img_rows=150, img_cols=150 , nb_output_classes = nb_classes, drop_out_rate = 0.4, n_hidden = 256, n_samples = 150, n_timesteps = n_timesteps)
    #nb_epochs = int(argv[0])
    if logger:
        csv_logger = CSVLogger('training_'+'achunet_'+str(nb_epochs)+'.log', separator=',', append=True)
    if lr_reduce:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=min_lr)

    callbacks = []
    if logger:
        callbacks.append(csv_logger)
    if lr_reduce:
        callbacks.append(reduce_lr)
    #-----------------------------------model compilation------------------------------------#

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    #-----------------------------Actual Training-------------------------------------------#
    for i in range(1,24,1):
        #get new sample and train again
        s1 = X_train.shape
        s2 = Y_train.shape
        if (augment_data):
            print "\nData Augmentation is being done while training for epochs: ",(int(nb_epochs))
            model.fit_generator(datagen.flow(X_train.reshape((s1[0]/n_timesteps),n_timesteps, s1[1], s1[2], s1[3]), Y_train.reshape((s2[0]/n_timesteps),n_timesteps, s2[1]), batch_size=batch_size), callbacks=callbacks, nb_epoch= int(nb_epochs), samples_per_epoch=len(X_train), validation_data=(X_test, Y_test))
        else:
            model.fit(X_train, Y_train, callbacks = callbacks , batch_size=batch_size, nb_epoch = nb_epochs, verbose=1, validation_data=(X_test, Y_test))
        X_train, Y_train = get_new_sample(i)

        loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
        metrics_file = open('achunet_data_augmented_metrics.txt', 'a')
    loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
    metrics_file = open('achunet_data_augmented_metrics.txt', 'a')
    metrics_file.write("\nloss: "+str(float(loss))+"\n")
    metrics_file.write("accuracy: "+str(float(accuracy)))
    #metrics_file.write("\nOptimizer, epochs: sgd_initlr=0.01, "+str(nb_epochs))
    metrics_file.close()

    # serialize weights to HDF5
    model.save_weights("achunet_model_" + str(nb_epochs) + "_epochs.h5")
    print("Saved weights to disk\n")


# In[ ]:



