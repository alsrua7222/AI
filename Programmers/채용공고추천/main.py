import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Data import *

path = r"C:\Users\KMK\Desktop\train_job"
if not os.path.isfile(path + r"\train2.csv"):
    trains = pd.read_csv(path + r"\train.csv")
    tags = pd.read_csv(path + r"\tags.csv")
    job_tags = pd.read_csv(path + r"\job_tags.csv")
    user_tags = pd.read_csv(path + r"\user_tags.csv")
    trains = getUserTagRatio(trains, job_tags, user_tags)
    if trains is None:
        print("getUserTagRatio return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train2.csv", index=False, sep=',')
else:
    trains = pd.read_csv(path + r"\train2.csv")

if not os.path.isfile(path + r"\train3.csv"):
    job_companies = pd.read_csv(path + r"\job_companies.csv")
    trains = getJobCompanySize(trains, job_companies)
    if trains is None:
        print("getJobCompanySize return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train3.csv", index=False, sep=',')
else:
    trains = pd.read_csv(path + r"\train3.csv")

if not os.path.isfile(path + r"t\rain4.csv"):
    user_tags = pd.read_csv(path + r"\user_tags.csv")
    trains = getUserTagCounts(trains, user_tags)
    if trains is None:
        print("getUserTagCounts return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train4.csv", index=False, sep=",")
else:
    trains = pd.read_csv(path + r"train4.csv")

# companySize에 따라 지원 횟수가 늘어나는지 검증해본다.
# x축은 NaN, 1-10, 11-50, 51-100, 101-200 인지 체크해본다.
# y축은 지원자의 기술스택이 회사가 요구하는 기술스택와 일치한비율을 나타낸다.

# seaborn.stripplot(x="companySize", y="UserTagRatio", hue="applied", data=trains)
# 선형 회귀 분석 시도.
# seaborn.lmplot(x="companySize", y="UserTagRatio", hue="applied", data=trains)

# UserTagRatio와 UserTagCounts가 서로 선형 관계인지 본다.
# seaborn.lmplot(y="UserTagRatio", x="UserTagCounts", hue="applied", data=trains)

# 3차원 lm분석
# seaborn.lmplot(y="UserTagRatio", x="UserTagCounts", hue="applied", col="companySize", data=trains)
# plt.show()

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import binary_crossentropy

x_train, x_test, y_train, y_test = train_test_split(trains[['UserTagCounts', 'UserTagRatio']], trains[['applied']], test_size=0.2, random_state=123)

mlp = Sequential()
mlp.add(Dense(2, input_shape=(2, 1)))
# mlp.add(Dense(3))
mlp.add(Dense(1, activation="sigmoid"))

mlp.summary()

# default - learning_rate = 0.01
sgd = SGD()

mlp.compile(optimizer=sgd, loss=binary_crossentropy, metrics=['accuracy'])

history = mlp.fit(x_train, y_train, batch_size=1, epochs=20, validation_split=0.2, verbose=2)

plt.plot(history.history['loss'], 'b-', label='loss')
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.title("accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# 손실 함수 계산

mlp.evaluate(x_test, y_test, batch_size=1, verbose=2)
line_x = np.arange(min(x_test), max(x_test), 0.01)
line_y = mlp.predict(line_x)

plt.plot(line_x, line_y, 'r-')
plt.plot(x_test, y_test, 'bo')
plt.title("Model")
plt.xlabel("test")
plt.ylabel("predict")
plt.legend(['predict', 'test'])
plt.show()


