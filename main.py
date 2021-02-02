#Tensorflow and keras imports
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

#Other imports
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

number_of_dataset_classes = 10
number_of_K_folds = 10
dataset = 'mnist'

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    return (X_train, Y_train), (X_test, Y_test)

def separate_dataset_into_K_folds(X_train, Y_train, number_of_K_folds):
    if number_of_K_folds == 10:
        folds = get_10_folds(X_train, Y_train)
    elif number_of_K_folds == 5:
        folds = get_5_folds(X_train, Y_train)
    return folds

def get_10_folds(X_train, Y_train, number_of_dataset_classes = number_of_dataset_classes):
    dataset_classes = get_dataset_classes(X_train, Y_train, number_of_dataset_classes)
    X_folds = [[],[],[],[],[],[],[],[],[],[]]
    Y_folds = [[],[],[],[],[],[],[],[],[],[]]
  
    #Pick each dataset class
    for jj, dataset_class in enumerate(dataset_classes):
        image_index = 0
        while image_index < len(dataset_class):
            for ii, fold in enumerate(X_folds):
                try:
                    #print('image index is: ', image_index, 'dataset_class_index is: ', dataset_class_index)
                    X_folds[ii].append(dataset_class[image_index])
                    Y_folds[ii].append(jj)
                    image_index += 1
                except Exception as e:
                    continue
        
    #Convert X_folds and Y_folds to numpy arrays
    for ii, fold in enumerate(X_folds):
        X_folds[ii] = np.array(fold)
    for ii, fold in enumerate(Y_folds):
        Y_folds[ii] = np.array(fold)
    X_folds = np.array(X_folds)
    Y_folds = np.array(Y_folds)

    for ii, X_fold in enumerate(X_folds):
        Y_fold = Y_folds[ii]
        #c = np.array([X_fold, Y_fold])
        indices = np.arange(X_fold.shape[0])
        np.random.shuffle(indices)
        # c[0] = c[0][indices]
        # c[1] = c[1][indices]
        # X_fold, Y_fold = c[0], c[1]
        X_fold = X_fold[indices]
        Y_fold = Y_fold[indices]
        X_folds[ii] = X_fold
        Y_folds[ii] = Y_fold

    return X_folds, Y_folds

def get_dataset_classes(X_train, Y_train, number_of_dataset_classes):
    if number_of_dataset_classes == 10:
        dataset_classes = [[],[],[],[],[],[],[],[],[],[]]
    elif number_of_dataset_classes == 5:
        dataset_classes = [[],[],[],[],[]]
    for dataset_class_index in range(number_of_dataset_classes):
        for item in range(X_train.shape[0]):
            if Y_train[item] == dataset_class_index:
                dataset_classes[dataset_class_index].append(X_train[item])
    return np.array(dataset_classes)

def create_fold_iterables(X_folds, Y_folds):
  iterables = []
  for ii, val_fold in enumerate(X_folds):
    X_stack = 0
    Y_stack = 0
    for jj, train_fold in enumerate(X_folds):
      if ii != jj:
        if type(X_stack) is int:
          X_stack = train_fold
          Y_stack = Y_folds[jj]
        else:
          X_stack= np.vstack((X_stack, train_fold))
          Y_stack= np.hstack((Y_stack, Y_folds[jj]))

    iterables.append([X_stack, Y_stack, val_fold, Y_folds[ii]])
  iterables = np.array(iterables)
  return iterables

#https://medium.com/towards-artificial-intelligence/the-architecture-implementation-of-lenet-5-eef03a68d1f7
def create_lenet_model(kernel_size, dr, layers):
    
    model = Sequential()

    # Adding a Convolution Layer C1
    # Input shape = N = (28 x 28)
    # No. of filters  = 6
    # Filter size = f = (5 x 5)
    # Padding = P = 0
    # Strides = S = 1
    # Size of each feature map in C1 is (N-f+2P)/S +1 = 28-5+1 = 24
    # No. of parameters between input layer and C1 = (5*5 + 1)*6 = 156
    model.add(Conv2D(filters=6, kernel_size=(kernel_size, kernel_size), padding='valid', input_shape=(28,28,1), activation='relu'))

    # Adding an Average Pooling Layer S2
    # Input shape = N = (24 x 24)
    # No. of filters = 6
    # Filter size = f = (2 x 2)
    # Padding = P = 0
    # Strides = S = 2
    # Size of each feature map in S2 is (N-f+2P)/S +1 = (24-2+0)/2+1 = 11+1 = 12
    # No. of parameters between C1 and S2 = (1+1)*6 = 12
    model.add(MaxPool2D(pool_size=(2,2)))

    # Adding a Convolution Layer C3
    # Input shape = N = (12 x 12)
    # No. of filters  = 16
    # Filter size = f = (5 x 5)
    # Padding = P = 0
    # Strides = S = 1
    # Size of each feature map in C3 is (N-f+2P)/S +1 = 12-5+1 = 8
    # No. of parameters between S2 and C3 = (5*5*6*16 + 16) + 16 = 2416
    if layers >= 2:
      model.add(Conv2D(filters=16, kernel_size=(kernel_size, kernel_size), padding='valid', activation='relu'))

    # Adding an Average Pooling Layer S4
    # Input shape = N = (8 x 8)
    # No. of filters = 16
    # Filter size = f = (2 x 2)
    # Padding = P = 0
    # Strides = S = 2
    # Size of each feature map in S4 is (N-f+2P)/S +1 = (8-2+0)/2+1 = 3+1 = 4
    # No. of parameters between C3 and S4 = (1+1)*16 = 32
    if layers >= 2:
      model.add(MaxPool2D(pool_size=(2,2)))

    if dr:
      model.add(Dropout(0.2))

    # As compared to LeNet-5 architecture there was one more application of convolution but in our code  further application of 
    # convolution with (5 x 5) filter would result in a negative dimension which is not possible. So we aren't applying any more
    # convolution here.
    if layers >= 3 and kernel_size == 3:
      model.add(Conv2D(filters=120, kernel_size=(kernel_size, kernel_size), padding='valid', activation='relu'))
    if layers >= 3 and kernel_size == 3:
      model.add(MaxPool2D(pool_size=(2,2)))

    # Flattening the layer S4
    # There would be 16*(4*4) = 256 neurons
    model.add(Flatten())

    # Adding a Dense layer with `tanh` activation+# 
    # No. of inputs = 256
    # No. of outputs = 120
    # No. of parameters = 256*120 + 120 = 30,840
    model.add(Dense(120, activation='relu'))

    # Adding a Dense layer with `tanh` activation
    # No. of inputs = 120
    # No. of outputs = 84
    # No. of parameters = 120*84 + 84 = 10,164
#     if ~flatten_as_only_dense:
#       model.add(Dense(84, activation='relu'))

    # Adding a Dense layer with `softmax` activation
    # No. of inputs = 84
    # No. of outputs = 10
    # No. of parameters = 84*10 + 10 = 850
    model.add(Dense(10, activation='softmax'))

    #model.summary()
    
    return model

def compile_and_fit_model(model, train_x, train_y, opt, lr, loss_function):#, val_x, val_y):
    #Reshape data
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    #val_x = val_x.reshape(val_x.shape[0], 28, 28, 1)
    
    #Normalize data
    train_x = train_x/255.0
    #val_x = val_x/255.0
    
    #One-hot encode the labels
    train_y = to_categorical(train_y, num_classes=10)
    #print('train_y.shape is: ', train_y.shape)
    #val_y = to_categorical(val_y, num_classes=10)
    
    if opt == 'Adam':
        opt = Adam(learning_rate = lr)
    elif opt == 'SGD':
        opt = SGD(learning_rate = lr)
    elif opt == 'RMSprop':
        opt = RMSprop(learning_rate = lr)
    
    if loss_function == 'categorical_crossentropy':
        loss='categorical_crossentropy'
    elif loss_function == 'kl_divergence':
        loss='kl_divergence'
        
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=128, epochs=20, verbose=0)#, validation_data=(val_x, val_y))
    return model

def evaluate_model(model, test_x, test_y):
    #val_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    print('val_x.shape: ', val_x.shape)
    val_x = test_x/255.0
    val_y = to_categorical(test_y, num_classes=10)
    print('val_y.shape: ', val_y.shape)
    score = model.evaluate(val_x, val_y, batch_size=128)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    
    
#######################################################################################################
(X_train, Y_train), (X_test, Y_test) = get_dataset(dataset)

X_folds, Y_folds = separate_dataset_into_K_folds(X_train, Y_train, 10)

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
#learning_rates = [0.00001]
optimizers = ['Adam', 'SGD', 'RMSprop']
num_layers = [1, 2, 3]
#num_layers = [2, 3]
kernel_sizes = [3, 5, 7]
#kernel_sizes = [3, 5]
#add_dropout = [True, False]
add_dropout = [False]
loss_functions = ['categorical_crossentropy', 'kl_divergence']
#loss_functions = ['kl_divergence']

iterables = create_fold_iterables(X_folds, Y_folds)
for lr in learning_rates:
    for layers in num_layers:
        for kernel_size in kernel_sizes:
            for dr in add_dropout:
                for loss_function in loss_functions:
                    for opt in optimizers:
                        scores = []
                        for iterable in tqdm(iterables):
                            if (kernel_size == 7 and layers == 3) or (kernel_size == 5 and layers == 3):
                                continue
                            K_fold_X_train, K_fold_Y_train, K_fold_X_test, K_fold_Y_test = iterable
                            # print(K_fold_X_train.shape)
                            # print(K_fold_Y_train.shape)
                            # print(K_fold_X_test.shape)
                            # print(K_fold_Y_test.shape)
                            val_x = K_fold_X_test.reshape(K_fold_X_test.shape[0], 28, 28, 1)
                            val_y = to_categorical(K_fold_Y_test, num_classes=10)
                            model = create_lenet_model(kernel_size, dr, layers)
                            model = compile_and_fit_model(model, K_fold_X_train, K_fold_Y_train, opt, lr, loss_function)
                            #evaluate_model(model, K_fold_X_test, K_fold_Y_test)
                            score = model.evaluate(val_x, val_y, verbose=0)
                            #print('\n', score[0])
                            #print(score[1])
                            scores.append(score[1])
                        #print(opt, '|', lr, '|', np.mean(scores), '|', np.std(scores), scores)
                        #print('\n', 'mean is: ', np.mean(scores))
                        #print('std is: ', np.std(scores))
                        f = open("run_stats.txt", "a")
                        s = '\n' +  str(lr) + '|' + str(layers) + '|' + str(kernel_size) + '|' + str(dr) + '|' + str(loss_function) + '|' + str(opt) + '|' +  str(np.mean(scores)) + '|' + str(np.std(scores))
                        print(s, np.mean(scores), '|', np.std(scores), scores)
                        f.write(s)
                        f.close()