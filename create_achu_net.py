from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from googlenet_custom_layers import PoolHelper,LRN
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras import backend as K

def create_achu_net(weights_path=None, img_chs = 3, img_rows=200, img_cols=200 , nb_output_classes = 7, drop_out_rate = 0.4, n_hidden = 256, n_samples = 100, n_timesteps = 16):
    # creates alinet based LSTM
    input_shape = (n_timesteps,  img_chs, img_rows, img_cols)
    input = Input(shape=input_shape)
    
    K.set_image_dim_ordering('th')
    
    conv1_7x7_s2 = TimeDistributed(Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002)))(input)
    
    conv1_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(conv1_7x7_s2)
    
    pool1_helper = TimeDistributed(PoolHelper())(conv1_zero_pad)
    
    pool1_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1/3x3_s2'))(pool1_helper)
    
    pool1_norm1 = TimeDistributed(LRN(name='pool1/norm1'))(pool1_3x3_s2)
    
    conv2_3x3_reduce = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu',name='conv2/3x3_reduce',W_regularizer=l2(0.0002)))(pool1_norm1)
    
    conv2_3x3 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2/3x3',W_regularizer=l2(0.0002)))(conv2_3x3_reduce)
    
    conv2_norm2 = TimeDistributed(LRN(name='conv2/norm2'))(conv2_3x3)
    
    conv2_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(conv2_norm2)
    
    pool2_helper = TimeDistributed(PoolHelper())(conv2_zero_pad)
    
    pool2_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2/3x3_s2'))(pool2_helper)
    
    
    inception_3a_1x1 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002)))(pool2_3x3_s2)
    
    inception_3a_3x3_reduce = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002)))(pool2_3x3_s2)
    
    inception_3a_3x3 = TimeDistributed(Convolution2D(128,3,3,border_mode='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002)))(inception_3a_3x3_reduce)
    
    inception_3a_5x5_reduce = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002)))(pool2_3x3_s2)
    
    inception_3a_5x5 = TimeDistributed(Convolution2D(32,5,5,border_mode='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002)))(inception_3a_5x5_reduce)
    
    inception_3a_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3a/pool'))(pool2_3x3_s2)
    
    inception_3a_pool_proj = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002)))(inception_3a_pool)
    
    inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=2,name='inception_3a/output')
    
    
    inception_3b_1x1 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/1x1',W_regularizer=l2(0.0002)))(inception_3a_output)
    
    inception_3b_3x3_reduce = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/3x3_reduce',W_regularizer=l2(0.0002)))(inception_3a_output)
                                       
    inception_3b_3x3 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu',name='inception_3b/3x3',W_regularizer=l2(0.0002)))(inception_3b_3x3_reduce)
    
    inception_3b_5x5_reduce = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3b/5x5_reduce',W_regularizer=l2(0.0002)))(inception_3a_output)
    
    inception_3b_5x5 = TimeDistributed(Convolution2D(96,5,5,border_mode='same',activation='relu',name='inception_3b/5x5',W_regularizer=l2(0.0002)))(inception_3b_5x5_reduce)
    
    inception_3b_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3b/pool'))(inception_3a_output)
    
    inception_3b_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3b/pool_proj',W_regularizer=l2(0.0002)))(inception_3b_pool)
    
    inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=2,name='inception_3b/output')
    
    
    inception_3b_output_zero_pad = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(inception_3b_output)
    
    pool3_helper = TimeDistributed(PoolHelper())(inception_3b_output_zero_pad)
    
    pool3_3x3_s2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool3/3x3_s2'))(pool3_helper)
    
    
    inception_4a_1x1 = TimeDistributed(Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_4a/1x1',W_regularizer=l2(0.0002)))(pool3_3x3_s2)
    
    inception_4a_3x3_reduce = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_4a/3x3_reduce',W_regularizer=l2(0.0002)))(pool3_3x3_s2)
    
    inception_4a_3x3 = TimeDistributed(Convolution2D(208,3,3,border_mode='same',activation='relu',name='inception_4a/3x3',W_regularizer=l2(0.0002)))(inception_4a_3x3_reduce)
    
    inception_4a_5x5_reduce = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_4a/5x5_reduce',W_regularizer=l2(0.0002)))(pool3_3x3_s2)
    
    inception_4a_5x5 = TimeDistributed(Convolution2D(48,5,5,border_mode='same',activation='relu',name='inception_4a/5x5',W_regularizer=l2(0.0002)))(inception_4a_5x5_reduce)
    
    inception_4a_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4a/pool'))(pool3_3x3_s2)
    
    inception_4a_pool_proj = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4a/pool_proj',W_regularizer=l2(0.0002)))(inception_4a_pool)
    
    inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=2,name='inception_4a/output')
    
    
    loss1_max_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),name='loss1/max_pool'))(inception_4a_output)
    
    loss1_conv = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',W_regularizer=l2(0.0002)))(loss1_max_pool)
    
    loss1_flat = TimeDistributed(Flatten())(loss1_conv)
    
    #LSTMs
    
    lstm_bd_1 = Bidirectional(LSTM(n_hidden, return_sequences=True, name='lstm1'))(loss1_flat)
    
    lstm_drop_1 = Dropout(drop_out_rate)(lstm_bd_1)
                              
    lstm_bd_2 = Bidirectional(LSTM(n_hidden, return_sequences=True, name='lstm2'))(lstm_drop_1)
    
    lstm_drop_2 = Dropout(drop_out_rate)(lstm_bd_2)
    
    lstm_bd_3 = Bidirectional(LSTM(n_hidden, return_sequences=True, name='lstm3'))(lstm_drop_2)
    
    lstm_drop_3 = Dropout(drop_out_rate)(lstm_bd_3)
                              
    lstm_bd_4 = Bidirectional(LSTM(n_hidden, return_sequences=True, name='lstm4'))(lstm_drop_3)
    
    lstm_drop_4 = Dropout(drop_out_rate)(lstm_bd_4)
    
    #lstm_max_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),name='lstm/max_pool'))(lstm_drop_4)
    
    loss1_fc_2 = TimeDistributed(Dense(1024,activation='relu',name='loss1/fc_2',W_regularizer=l2(0.0002)))(lstm_drop_4)
    
    loss1_drop_fc_2 = Dropout(drop_out_rate)(loss1_fc_2)
    
    loss1_classifier = Dense(nb_output_classes,name='loss1/classifier',W_regularizer=l2(0.0002))(loss1_drop_fc_2)
    
    loss1_classifier_act = Activation('softmax')(loss1_classifier)
    
    achu_net = Model(input=input, output=[loss1_classifier_act])
    if weights_path:
        achu_net.load_weights(weights_path)
    
    return achu_net
