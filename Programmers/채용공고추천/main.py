import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Data import *

path = r"C:\Users\KMK\Desktop\train_job"
if not os.path.isfile(path + r"\train2.csv") or not os.path.isfile(path + r"\test_job2.csv"):
    trains = pd.read_csv(path + r"\train.csv")
    test_job = pd.read_csv(path + r"\test_job.csv")
    tags = pd.read_csv(path + r"\tags.csv")
    job_tags = pd.read_csv(path + r"\job_tags.csv")
    user_tags = pd.read_csv(path + r"\user_tags.csv")
    trains = getUserTagRatio(trains, job_tags, user_tags)
    test_job = getUserTagRatio(test_job, job_tags, user_tags)
    if trains is None or test_job is None:
        print("getUserTagRatio return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train2.csv", index=False, sep=',')
        test_job.to_csv(path + r"\test_job2.csv", index=False, sep=',')
else:
    trains = pd.read_csv(path + r"\train2.csv")
    test_job = pd.read_csv(path + r"\test_job2.csv")

if not os.path.isfile(path + r"\train3.csv") or not os.path.isfile(path + r"\test_job3.csv"):
    job_companies = pd.read_csv(path + r"\job_companies.csv")
    trains = getJobCompanySize(trains, job_companies)
    test_job = getJobCompanySize(test_job, job_companies)
    if trains is None:
        print("getJobCompanySize return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train3.csv", index=False, sep=',')
        test_job.to_csv(path + r"\test_job3.csv", index=False, sep=',')
else:
    trains = pd.read_csv(path + r"\train3.csv")
    test_job = pd.read_csv(path + r"\test_job3.csv")

if not os.path.isfile(path + r"t\rain4.csv") or not os.path.isfile(path + r"\test_job4.csv"):
    user_tags = pd.read_csv(path + r"\user_tags.csv")
    trains = getUserTagCounts(trains, user_tags)
    test_job = getUserTagCounts(test_job, user_tags)
    if trains is None:
        print("getUserTagCounts return => None")
        exit(0)
    else:
        trains.to_csv(path + r"\train4.csv", index=False, sep=",")
        test_job.to_csv(path + r"\test_job4.csv", index=False, sep=',')
else:
    trains = pd.read_csv(path + r"train4.csv")
    test_job.to_csv(path + r"\test_job4.csv")

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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(trains[['UserTagCounts', 'UserTagRatio']], trains[['applied']], test_size=0.2, random_state=123)
# x_train, x_test, y_train, y_test = train_test_split(trains[['UserTagCounts', 'UserTagRatio', 'companySize']], trains[['applied']], test_size=0.2, random_state=123)

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# mlp = DecisionTreeClassifier(max_depth=5, random_state=5)
# mlp = RandomForestClassifier(random_state=5)
# mlp = KNeighborsClassifier()
# mlp = SVC()
# mlp = ExtraTreeClassifier(random_state=5)
mlp = BaggingClassifier(RandomForestClassifier())
mlp.fit(x_train, y_train)
print(mlp.score(x_test, y_test))

x_input = test_job[["UserTagCounts", "UserTagRatio"]]
# x_input = test_job[["UserTagCounts", "UserTagRatio", "companySize"]]
y_output = mlp.predict(x_input)

df_y = pd.DataFrame(y_output)
print(df_y[df_y[0] == 1])
df_y.to_csv(path + r"\result.csv", index=False, sep=',')
print("추출 완료.")
