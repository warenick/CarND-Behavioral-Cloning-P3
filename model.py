# csv_fname = 'driving_log.csv'


from os import listdir
files = [] 
data_path = './IMG/'
# open data files
for file in listdir(data_path):
    if file.endswith(".csv"):
        files.append(file)


from os import path
from pandas import DataFrame
import csv
dfs = DataFrame(columns=['image', 'measure'])
# 
for file in files:
    with open(path.join(data_path, file)) as csvf:
        next(csvf)
        reader = csv.reader(csvf)
        for line in reader:
            for i in range(3):
                source_path = line[i]
                file_path = source_path.split('/')[-1]
                if i == 0:
                    measurement = float(line[3])
                elif i == 1:
                    measurement = float(line[3]) + 0.1
                elif i == 2:
                    measurement = float(line[3]) - 0.1
                df = DataFrame([[file_path, measurement]], columns=['image', 'measure'])
                dfs = dfs.append(df, ignore_index=True)
                
# show dataframes info
dfs.info()


# add some normalization to the data and cut some really strait steering angles examples
# to make data normaly distributed
ndfs= []
ndfs = dfs.drop(dfs[(dfs['measure'] < 0.02) & (dfs['measure'] > -0.01)].index)
ndfs = ndfs.drop(ndfs[(ndfs['measure'] < -0.099) & (ndfs['measure'] > -0.12)].index)
ndfs = ndfs.drop(ndfs[(ndfs['measure'] < 0.12) & (ndfs['measure'] > 0.095)].index)


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D

# tensorflow.keras.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define net
model_name = "model" 
# Nvidia net without batch normalization
model = Sequential()
input_shape = (80, 160, 3)
model.add(Cropping2D(cropping=((25,10), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: tf.cast(x, tf.float32) / 255.0 - 0.5))
# first convolution layer
model.add(layers.Conv2D(24, (5, 5), 1,
                        padding="valid"))  # filter num, kernel size, stride, padding
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# second convolution layer, kernel_regularizer=regularizers.l2(0.1)
model.add(layers.Conv2D(36, (5, 5), 2,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# third conv layer
model.add(layers.Conv2D(48, (3, 3), 2,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# forth conv layer
model.add(layers.Conv2D(64, (3, 3), 1,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Flatten())
# first fully connected layer
model.add(layers.Dense(120))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# second fully connected layer
model.add(layers.Dense(64))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# third fully connected layer
model.add(layers.Dense(10))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
# output layer
model.add(layers.Dense(1))
print(model)





from tensorflow.keras.preprocessing.image import ImageDataGenerator
# define generator for learing
BS = 256
target_size = (80, 160)
dataframe = ndfs
directory = data_path
# define data to train and validation
datagen=ImageDataGenerator(validation_split=0.2)
train_generator=datagen.flow_from_dataframe(dataframe=dataframe,
                                            directory=directory,
                                            x_col="image",
                                            y_col="measure",
                                            subset="training",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="raw",
                                            target_size=target_size)

valid_generator=datagen.flow_from_dataframe(dataframe=dataframe,
                                            directory=directory,
                                            x_col="image",
                                            y_col="measure",
                                            subset="validation",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="raw",
                                            target_size=target_size)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

NUM_EPOCHS = 50
INIT_LR = 2*1e-3
        
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model.compile(loss="mse", optimizer=opt, metrics=["mse"])

# use checkpointer to save best model by validation mean squared error
checkpointer = ModelCheckpoint(filepath=f"{model_name}.h5", 
                               monitor = 'val_loss',
                               verbose=1, 
                               save_best_only=True)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# train model
H = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=NUM_EPOCHS,
                    callbacks=[checkpointer]
)

print("finish training")
