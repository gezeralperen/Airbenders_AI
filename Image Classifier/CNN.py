import cv2
import numpy as np
import os
from random import shuffle, sample
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout


def shape_input(img, input_size=(64, 64)):
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.reshape(img, (64,64,1))/255

def prepare_data():
    objs = os.listdir('data')
    print(objs)
    data = []
    for obj in objs:
        for file in os.listdir('data/' + obj):
            one_hot = np.zeros(len(objs))
            one_hot[objs.index(obj)] = 1
            data.append(['data/' + obj + '/' + file, one_hot])
    shuffle(data)
    data = np.array(data)
    train_size = int(len(data)*0.8)

    y_test = np.vstack(data[train_size:,1])
    y_train = np.vstack(data[:train_size,1])


    img_data = []
    for data_point in data:
        img = cv2.imread(data_point[0])
        img_data.append(shape_input(img))
    img_data = np.array(img_data)
    # img_data = np.reshape(img_data, (img_data.shape[0],img_data.shape[1],img_data.shape[2],1))

    x_train = img_data[:train_size]
    x_test = img_data[train_size:]

    x_train_file = open('x_train_file', 'wb')
    y_train_file = open('y_train_file', 'wb')
    x_test_file = open('x_test_file', 'wb')
    y_test_file = open('y_test_file', 'wb')
    objs_file = open('objs_file', 'wb')
    pickle.dump(x_train, x_train_file)
    pickle.dump(y_train, y_train_file)
    pickle.dump(x_test, x_test_file)
    pickle.dump(y_test, y_test_file)
    pickle.dump(objs, objs_file)

def import_data():
    x_train_file = open('x_train_file', 'rb')
    y_train_file = open('y_train_file', 'rb')
    x_test_file = open('x_test_file', 'rb')
    y_test_file = open('y_test_file', 'rb')
    objs_file = open('objs_file', 'rb')
    x_train = pickle.load(x_train_file)
    y_train = pickle.load(y_train_file)
    x_test = pickle.load(x_test_file)
    y_test = pickle.load(y_test_file)
    objs = pickle.load(objs_file)
    return x_train,y_train,x_test,y_test, objs

def train_data():
    x_train,y_train,x_test,y_test, objs = import_data()
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,1)))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(len(objs), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)

    pickle.dump(model, open('NN', 'wb'))

#prepare_data()
#train_data()

objs = os.listdir('data')
images = []
for obj in objs:
    for file in os.listdir('data/' + obj):
        images.append('data/' + obj + '/' + file)


model = pickle.load(open('NN', 'rb'))
while True:
    random_pick = cv2.imread(sample(images,1)[0])
    prediction = model.predict(np.reshape(shape_input(random_pick), (1,64,64,1)))
    print("\nPredictions:")
    for obj in objs:
        print(f"{obj}: {int(prediction[0][objs.index(obj)]*100)}%")
    cv2.imshow('Predicted Image', random_pick)
    cv2.waitKey()

