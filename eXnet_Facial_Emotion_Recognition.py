import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,MaxPooling2D,\
                                    AveragePooling2D, Flatten, Dense, Add, ZeroPadding2D, concatenate, GlobalAveragePooling2D,Lambda,GlobalMaxPooling2D,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras.callbacks import *
from keras.utils.generic_utils import get_custom_objects
import math
from augmentations import *




class Preprocessing:
    
    '''
    Transforming and Normalising data between 0-1
    '''

    def __init__(self, file, num_of_instances=None, lines=None):
      self.num_of_instances = num_of_instances
      self.lines = lines
      self.file = file
      self.num_classes = 6

      self.x_train, self.y_train, self.x_test, self.y_test, self.x_valid, self.y_valid = ([] for i in range(6))


    def read_data(self):
      with open(self.file) as f:
        content = f.readlines()

      self.lines = np.array(content)
      self.num_of_instances = self.lines.size


    def segregate_data(self):

      '''
      segregating train,validation,test set
      '''

      for i in range(1, self.num_of_instances):
        emotion, img, usage = self.lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')

        emotion = tf.keras.utils.to_categorical(emotion, self.num_classes)

        if 'Training' in usage:
            self.y_train.append(emotion)
            self.x_train.append(pixels)
        elif 'PublicTest' in usage:
            self.y_test.append(emotion)
            self.x_test.append(pixels)
        elif 'PrivateTest' in usage:
            self.y_valid.append(emotion)
            self.x_valid.append(pixels)

      print(len(self.x_train),len(self.y_train),len(self.x_test),len(self.y_test),len(self.x_valid),len(self.y_valid))


    def transfrom_data(self):

      #data transformation for train and test sets
      self.x_train = np.array(self.x_train, 'float32')
      self.y_train = np.array(self.y_train, 'float32')
      self.x_test = np.array(self.x_test, 'float32')
      self.y_test = np.array(self.y_test, 'float32')


      self.x_valid = np.array(self.x_valid, 'float32')
      self.y_valid = np.array(self.y_valid, 'float32')

      #normalize inputs between [0, 1]
      self.x_train /= 255 
      self.x_test /= 255
      self.x_valid /= 255

      self.x_train = self.x_train.reshape(self.x_train.shape[0], 48, 48, 1)
      self.x_train = self.x_train.astype('float32')
      self.x_test = self.x_test.reshape(self.x_test.shape[0], 48, 48, 1)
      self.x_test = self.x_test.astype('float32')

      self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], 48, 48, 1)
      self.x_valid = self.x_valid.astype('float32')

      return self.x_train,self.y_train,self.x_test,self.y_test,self.x_valid,self.y_valid


class Generator:
    '''
    multiple variations of images
    '''
    def generator_data(self, x_train, y_train, batch_size):

        datagen = ImageDataGenerator(
            featurewise_std_normalization=False,              
            rotation_range = 30, width_shift_range = 0.15,
            height_shift_range = 0.15, shear_range = 0.15,
            zoom_range = 0.15,horizontal_flip=True, 
            fill_mode="nearest",
            preprocessing_function = get_random_eraser(v_l=0, v_h=10, p =0.3))

        train_generator = MixupGenerator(x_train, y_train, alpha=0.6, batch_size=batch_size, datagen=datagen)()

        return train_generator
   

# From Pyimagesearch Search Adrian Rosebrock
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.01, max_lr=0.0001, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


class Modelling:
          

    def swish(self, x):
        return (K.sigmoid(x) * x)
    get_custom_objects().update({'swish': swish})


    def relu_bn(self,inputs: Tensor) -> Tensor:
        bn = BatchNormalization()(inputs)
        relu = ReLU()(bn)
    
        return relu


    def create_convnet(self):
        input_shape = Input(shape=(48, 48, 1))

        initial = Conv2D(64, (3, 3),padding='same')(input_shape)
        x = self.relu_bn(initial)
        x = Conv2D(64, (3, 3),padding='same')(x)
        x = self.relu_bn(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
        initial_block = BatchNormalization()(x)
        
        para_block_1 = Conv2D(128, (1, 1), padding='same')(initial_block)
        p_1 = self.relu_bn(para_block_1)
        p_1 = Conv2D(128, (3, 3), padding='same')(p_1)
        p_1 = self.relu_bn(p_1)
        
        para_block_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(initial_block)
        p_2 = Conv2D(128, (1, 1), padding='same')(para_block_2)
        p_2 = self.relu_bn(p_2)
        p_2 = Conv2D(128, (3, 3), padding='same')(p_2)
        p_2 = self.relu_bn(p_2)
        
        merged_para_1 = tf.keras.layers.concatenate([p_1, p_2], axis=1)
        merged_para_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(merged_para_1)
        
        para_block_3 = Conv2D(256, (1, 1), padding='same')(merged_para_1)
        p_21 = self.relu_bn(para_block_3)
        p_21 = Conv2D(256, (3, 3), padding='same')(p_21)
        p_21 = self.relu_bn(p_21)
        
        para_block_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(merged_para_1)
        p_22 = Conv2D(256, (1, 1), padding='same')(para_block_4)
        p_22 = self.relu_bn(p_22)
        p_22 = Conv2D(256, (3, 3), padding='same')(p_22)
        p_22 = self.relu_bn(p_22)
        
        merged_para_2 = tf.keras.layers.concatenate([p_21, p_22], axis=1)
        merged_para_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(merged_para_2)
        
        x = Conv2D(512, (1, 1), padding='same')(merged_para_2)
        x = self.relu_bn(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = self.relu_bn(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)   
        x = AveragePooling2D((2,2),strides=(1, 1))(x)
        
        final = GlobalAveragePooling2D()(x)

        final = Dense(512, activation='relu')(final)
        final = Dropout(0.4)(final)
        final = Dense(6, activation='softmax')(final)
        
        model = Model(input_shape, final)
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, decay=4e-5, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        return model




if __name__ == "__main__":
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', min_lr=0.00001, factor=0.5, verbose=1, patience=5)
    num_classes = 6 #angry, neutral, fear, happy, sad, surprise
    batch_size = 64

    file = "C:/Arun/GitHub/Facial_emotion/Data/Copy of fer2013_modified.csv"
    
    preprocess = Preprocessing(file)
    preprocess.read_data()
    preprocess.segregate_data()
    x_train, y_train, x_test, y_test, x_valid, y_valid = preprocess.transfrom_data()
    print(x_train.shape, y_test.shape, x_valid.shape, y_valid.shape)

    img_generator = Generator()
    train_generator = img_generator.generator_data(x_train, y_train, batch_size)
    
    modelling = Modelling()
    model = modelling.create_convnet()
    print(model.summary())


    clr = CyclicLR(mode="triangular",base_lr=0.01,max_lr=0.0001,\
        step_size= 60 * (x_train.shape[0] // 64))

    model.fit(
              train_generator,
              epochs = 200,
              validation_data=(x_valid,y_valid),
              steps_per_epoch = len(x_train)//batch_size,
              shuffle = True, verbose=1,
                callbacks=[clr]
              )

predicted_test_labels = np.argmax(model.predict(x_test), axis=1)
test_labels = np.argmax(y_test, axis=1)
print ("Accuracy score = ", accuracy_score(test_labels, predicted_test_labels))
print ("precision score = ", precision_score(test_labels, predicted_test_labels , average='weighted'))
print ("recall score = ", recall_score(test_labels, predicted_test_labels, average='weighted'))
print ("f1 score = ", f1_score(test_labels, predicted_test_labels, average='weighted'))