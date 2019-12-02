import tensorflow as tf
import numpy as np
import time

class Model():
    _kwords=['batchsize','earlystopping',]


    def __init__(self,**kwargs):

        self.pre_trained=tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_tensor=None, input_shape=(96,96,3), pooling=None)
        self.preprocessing=tf.keras.applications.vgg16.preprocess_input
        
        self.args={'patience':1,
              'batch_size':32,
              'learing_rate_1':1e-3,
              'learing_rate_2':1e-4,
              'phase_1_epochs':1,
              'phase_2_epochs':1,
              'early_stopping':0,
              'dropout_rate_1':.2,
              'dropout_rate_2':.2,
              'layer_size':128,
              'test':False
        }

        for arg,val in kwargs.items():
            if arg not in self.args:
                raise( arg+'  is an unkown keyword passed to Model')
            else:
                self.args[arg]=val
        # Below is what creates the datasests from files saved on talapas

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                    preprocessing_function=self.preprocessing,
                    width_shift_range=0,  # randomly shift images horizontally
                    height_shift_range=0,  # randomly shift images vertically 
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=True,
                    shear_range=1,
                    zoom_range=.05,
                    rotation_range=15                           
                    )  # randomly flip images

        self.train_generator=data_gen.flow_from_directory('/projects/datascience/shared/2019_ML_workshop/datasets/pcamv1/images/train',
                                                    target_size=(96,96), 
                                                    color_mode='rgb', 
                                                    classes=['normal','tumor'],
                                                    class_mode='binary',
                                                    batch_size=self.args['batch_size'],
                                                    shuffle=True)

        develop_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                    preprocessing_function=self.preprocessing,
                    width_shift_range=0,  # don't do anything but preprocess
                    height_shift_range=0,  
                    horizontal_flip=False,  
                    vertical_flip=False,
                    shear_range=0,
                    zoom_range=.00,
                    rotation_range=0                          
                    )  # randomly flip images

        self.develop_generator=develop_gen.flow_from_directory('/projects/datascience/shared/2019_ML_workshop/datasets/pcamv1/images/develop',
                                                    target_size=(96,96), 
                                                    color_mode='rgb', 
                                                    classes=['normal','tumor'],
                                                    class_mode='binary',
                                                    batch_size=32,
                                                    shuffle=False)

        self.test_generator=develop_gen.flow_from_directory('/projects/datascience/shared/2019_ML_workshop/datasets/pcamv1/images/test',
                                                    target_size=(96,96), 
                                                    color_mode='rgb', 
                                                    classes=['normal','tumor'],
                                                    class_mode='binary',
                                                    batch_size=32,
                                                    shuffle=False)




    def print_args(self):
        for arg,item in self.args.items():
            print(arg + " "+str(item))

    def buildmodel(self):
        print("Building Model")
        self.print_args()
        # Fix all these layers so we don't train them right away
        for l in self.pre_trained.layers:
            l.trainable=False


        # Add our own new Dense Layers    
        flat=tf.keras.layers.Flatten()(self.pre_trained.output)
        top=tf.keras.layers.Dropout(self.args['dropout_rate_1'])(flat)
        top=tf.keras.layers.Dense(self.args['layer_size'])(top)
        top=tf.keras.layers.LeakyReLU()(top)
        top=tf.keras.layers.Dropout(self.args['dropout_rate_2'])(top)
        classification=tf.keras.layers.Dense(1,activation='sigmoid')(top)

        # Model/compile like before
        self.model=tf.keras.models.Model([self.pre_trained.input],classification)
        self.model.summary()
        self.model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=self.args['learing_rate_1']),metrics=['accuracy'])
        

    def fitmodel(self):    
        print("Starting Fit")
        self.buildmodel()
        
        # Create Early Stopping Callback
        es=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.args['patience'], verbose=0, mode='auto')


        #Use only 10 steps per phase if test is true

        #Phase 1
        if self.args['test']:
            self.model.fit_generator(self.train_generator,steps_per_epoch=10,epochs=self.args['phase_1_epochs'],validation_data=self.develop_generator,callbacks=[es],validation_steps=10)
        else:
            self.model.fit_generator(self.train_generator,epochs=self.args['phase_1_epochs'],validation_data=self.develop_generator,callbacks=[es])

        #Phase 2
        for l in self.pre_trained.layers:
            l.trainable=True
        self.model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=self.args['learing_rate_2']),metrics=['accuracy'])

        if self.args['test']:
            self.model.fit_generator(self.train_generator,steps_per_epoch=10,epochs=self.args['phase_2_epochs'],validation_data=self.develop_generator,callbacks=[es],validation_steps=10)
        else:
            self.model.fit_generator(self.train_generator,epochs=self.args['phase_2_epochs'],validation_data=self.develop_generator,callbacks=[es])


    def runexample(self):
        stime=time.time()
        self.fitmodel()
        print("Evaluating Fit")
        if self.args['test']:
            output=self.model.evaluate_generator(self.test_generator,steps=10)
        else:
            output=self.model.evaluate_generator(self.test_generator)

        total_time=time.time()-stime        
        return output,total_time



if __name__=='__main__':
    new_model=Model()
    new_model.runexample()
