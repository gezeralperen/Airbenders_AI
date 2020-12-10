import csv
import numpy as np
from keras.models import Sequential
import keras.models
from keras import layers, optimizers, losses
import pickle



class PipeLine():
    def __init__(self, data=[]):
        self.data = data
        self.train_model()

    def preprocess(self, new_list, train=False):
        y = np.array(new_list[:, -1:]).astype(float)
        x = new_list[:, 3:-1]
        if train:
            self.count = []
            for country in x[:, 1]:
                if country not in self.count:
                    self.count.append(country)

        x_country_encoder = []
        for country in x[:, 1]:
            row = [0, 0]
            if self.count.index(country) == 1:
                row[0] = 1
            if self.count.index(country) == 2:
                row[1] = 1
            x_country_encoder.append(row)

        x = np.append(x, x_country_encoder, axis=1)
        x = np.delete(x, 1, 1)

        for i, gender in enumerate(x[:, 1]):
            if gender == 'Female':
                x[i][1] = 1
            else:
                x[i][1] = 0


        if train:
            self.maxes = np.zeros(11)
            self.mins = np.zeros(11)

            for row in x:
                for i, value in enumerate(row):
                    if self.maxes[i] < float(value):
                        self.maxes[i] = float(value)
                    if self.mins[i] > float(value):
                        self.mins[i] = float(value)

        x_final = []
        for row in x:
            new_row = []
            for i, value in enumerate(row):
                new_row.append((float(value) - self.mins[i]) / (self.maxes[i] - self.mins[i]))
            x_final.append(new_row)
        x = np.array(x_final)
        return x, y

    def train_model(self):
        processed_x, processed_y = self.preprocess(self.data, train=True)
        data_rows = processed_x.shape[0]
        train_size = int(0.8*data_rows)

        x_train = processed_x[:train_size, :]
        y_train = processed_y[:train_size, :]

        x_test = processed_x[train_size:, :]
        y_test = processed_y[train_size:, :]

        self.model = Sequential(
            [
                layers.Dense(10, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Dense(3, activation="relu"),
                layers.Dense(1, activation='sigmoid'),
            ]
        )
        self.model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

        self.model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=100,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_test, y_test),
        )

    def predict(self, data, have_y=False):
        if not have_y:
            zeros = np.zeros((data.shape[0],1))
            data = np.hstack((data, zeros))
        x,y = self.preprocess(data)
        return self.model.predict(x)

    def save(self, file_path):
        pickle.dump(self, open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        return pickle.load(open(file_path, 'rb'))



file = open('Churn_Modelling.csv', 'r')
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append(row)

data = np.array(data)

# pipe = PipeLine(data=data[1:,:])
# pipe.save('pipeline.object')

pipe = PipeLine.load('pipeline.object')
y_predicted = pipe.predict(data[1:,:-1])
y_real = data[:,-1:]

compare = np.hstack((y_real[1:,:],y_predicted))
breakpoint()
