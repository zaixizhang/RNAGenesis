# import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Dropout, Concatenate, Lambda, BatchNormalization, Add, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
# from tensorflow.keras import losses
# from tensorflow.keras.utils import Sequence
tf.compat.v1.enable_eager_execution()


def apply_pad_mask(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    return tf.multiply(tensor, mask)

class LogNonhomogenousGeometric(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)    

    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, x):
        log_P = tf.log_sigmoid(x)
        log_inverse_P = -x + log_P
        cumul_P = tf.cumsum(log_inverse_P, axis=1, exclusive=True) # exclusive ensures correct index
        Q = log_P + cumul_P
        return Q
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Function to compute an interaction term between a value and a one-hot vector
def interaction_term(tensors):
    prediction = tensors[0]
    experiment_indicator = tensors[1]
    return tf.multiply(prediction, experiment_indicator)

# Layer which slices input tensor into three tensors, one for each frame w.r.t. the canonical start
class FrameSliceLayer(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape) 
    
    def call(self, x):
        shape = K.shape(x)
        x = K.reverse(x, axes=1) # reverse, so that frameness is related to fixed point (start codon)
        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
        return [frame_1, frame_2, frame_3]
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return [(input_shape[0], None),(input_shape[0], None),(input_shape[0], None)]
        return [(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2])]
    
# Masking to prevent zero padding to influence results
def compute_pad_mask(x):
    return K.sum(x, axis=2)

def apply_pad_mask(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    return tf.multiply(tensor, mask)

# Average pooling that accounts for masking
def global_avg_pool_masked(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    return K.sum(tensor, axis=1)/K.sum(mask, axis=1)

def convolve_and_mask(conv_features, pad_mask, n_filters, kernel_size, suffix, prefix="",
                      padding="causal", dilation=1, batchnorm=False, conv_dropout=0.0):
    convolution = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation, activation='relu', 
                           padding=padding, name=prefix+"convolution_"+suffix)
    conv_features = convolution(conv_features)
    conv_features = Lambda(apply_pad_mask, name=prefix+"apply_pad_mask_"+suffix)([conv_features, pad_mask]) # Mask padding
    if batchnorm:
        conv_features = BatchNormalization(axis=2, name="batchnorm_"+suffix)(conv_features)
    if conv_dropout > 0.0:
        conv_features = SpatialDropout1D(conv_dropout, name=prefix+"1d_dropout_"+suffix)(conv_features)
    return conv_features

def inception_block(conv_features, pad_mask, n_filters, suffix, prefix=""):
    conv_features_3 = convolve_and_mask(conv_features, pad_mask, n_filters[0], kernel_size=3, suffix="incept3_"+suffix, prefix=prefix)
    conv_features_5 = convolve_and_mask(conv_features, pad_mask, n_filters[1], kernel_size=5, suffix="incept5_"+suffix, prefix=prefix)
    conv_features_7 = convolve_and_mask(conv_features, pad_mask, n_filters[2], kernel_size=7, suffix="incept7_"+suffix, prefix=prefix)
    conv_features = Concatenate(name="incept_concat"+suffix)([conv_features_3, conv_features_5, conv_features_7])
    return conv_features

def create_frame_slice_model(n_conv_layers=3, 
                        kernel_size=[8,8,8], n_filters=128, dilations=[1, 1, 1],
                        padding="causal", use_batchnorm=False,
                        conv_dropout=[0.0, 0.0, 0.0],
                        use_inception=False, skip_connections="", 
                        n_dense_layers=1, fc_neurons=[64], fc_drop_rate=0.2,
                        only_max_pool=False,
                        loss='mean_squared_error',
                        use_counter_input=False,
                        use_scaling_regression=False, library_size=6):
    # Inputs
    input_seq = Input(shape=(None, 4), name="input_seq")
    inputs = input_seq
    conv_features = input_seq
    # Compute presence of zero padding
    pad_mask = Lambda(compute_pad_mask, name="compute_pad_mask")(conv_features)

    layer_list = []
    for i in range(n_conv_layers):
        if skip_connections:
            conv_features_shortcut = conv_features #shortcut connections
        if use_inception:
            conv_features = inception_block(conv_features, pad_mask, n_filters, suffix=str(i))   
        else:
            conv_features = convolve_and_mask(conv_features, pad_mask, n_filters, kernel_size[i], 
                                                          suffix=str(i), padding=padding, 
                                                          dilation=dilations[i], 
                                                          batchnorm=use_batchnorm,
                                                          conv_dropout=conv_dropout[i])   
        if skip_connections == "residual" and i > 0:
            conv_features = Add(name="add_residual_"+str(i))([conv_features, conv_features_shortcut])
        elif skip_connections == "dense":
            conv_features = Concatenate(axis=-1, name="concat_dense_"+str(i))([conv_features,
                                                                               conv_features_shortcut])
    # Frame based masking    
    frame_masked_features = FrameSliceLayer(name="frame_masking")(conv_features)
    frame_masked_pad_mask = FrameSliceLayer(name="frame_masking_padmask")(pad_mask)
    # Pooling
    pooled_features = []
    max_pooling = GlobalMaxPooling1D(name="pool_max_frame_conv")
    avg_pooling = Lambda(global_avg_pool_masked, name="pool_avg_frame_conv")
    pooled_features = pooled_features + \
                    [max_pooling(frame_masked_features[i]) for i in range(len(frame_masked_features))]
    if not only_max_pool:
        pooled_features = pooled_features + [avg_pooling([frame_masked_features[i], frame_masked_pad_mask[i]]) for i in 
                     range(len(frame_masked_features))]
    pooled_features = Concatenate(axis=-1, name="concatenate_pooled")(pooled_features)
    # Add tis_context if necessary
    concat_features = pooled_features
    # Prediction (Dense layer)
    predict = concat_features
    for i in range(n_dense_layers):
        predict = Dense(fc_neurons[i], activation='relu', name="fully_connected_"+str(i))(predict)
        predict = Dropout(rate=fc_drop_rate, name="fc_dropout_"+str(i))(predict)
    predict = Dense(1, name="mrl_output_unscaled")(predict) 

    model = Model(inputs=inputs, outputs=predict)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss, optimizer=adam)
    return model

def load_framepool(path="models/guidance/framepool/utr_model_combined_residual_new.h5"):
    model = create_frame_slice_model(kernel_size=[7,7,7],
                            only_max_pool=False,
                            padding="same",
                            skip_connections="residual",
                            use_scaling_regression=True, library_size=2)
    
    model.load_weights(path)
    return model