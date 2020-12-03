import csv
import numpy as np
from keras.models import Sequential
from keras import layers, optimizers, losses

file = open('Churn_Modelling.csv', 'r')
csv_reader = csv.reader(file, delimiter=',')


data = []
for row in csv_reader:
    data.append(row)

data = np.array(data)


data = data[1:,3:]
y = np.array(data[:,-1:]).astype(float)
x = data[:,:-1]


count = []
for country in x[:,1]:
    if country not in count:
        count.append(country)


x_country_encoder = []

for country in x[:,1]:
    row = [0,0]
    if count.index(country) == 1:
        row[0] = 1
    if count.index(country) == 2:
        row[1] = 1
    x_country_encoder.append(row)

x_appended = np.append(x, x_country_encoder, axis=1)
x_appended = np.delete(x_appended, 1, 1)

gender_encoder = []
for gender in x_appended[:,1]:
    if gender == 'Female':
        gender_encoder.append([1])
    else:
        gender_encoder.append([0])

x_appended = np.append(x_appended, gender_encoder, axis=1)
x_appended = np.delete(x_appended, 1, 1)

maxes = np.zeros(11)
mins = np.zeros(11)

for row in x_appended:
    for i, value in enumerate(row):
        if maxes[i] < float(value):
            maxes[i] = float(value)
        if mins[i] > float(value):
            mins[i] = float(value)


x_final = []

for row in x_appended:
    new_row = []
    for i, value in enumerate(row):
        new_row.append((float(value)-mins[i])/(maxes[i]-mins[i]))
    x_final.append(new_row)
x_final = np.array(x_final)
x_train = x_final[:8000,:]
y_train = y[:8000,:]


x_test = x_final[8000:,:]
y_test = y[8000:,:]

model = Sequential(
    [
        layers.Dense(6, activation="relu"),
        layers.Dense(6, activation="relu"),
        layers.Dense(1, activation='sigmoid'),
    ]
)
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
breakpoint()
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_test, y_test),
)

