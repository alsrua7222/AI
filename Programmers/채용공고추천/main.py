import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn

from Data import *

if not os.path.isfile(r"C:\Users\KMK\Desktop\train_job\train2.csv"):
    trains = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\train.csv")
    tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\tags.csv")
    job_tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\job_tags.csv")
    user_tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\user_tags.csv")
    trains = getUserTagRatio(trains, job_tags, user_tags)
    if trains is None:
        print("getUserTagRatio return => None")
        exit(0)
    else:
        trains.to_csv(r"C:\Users\KMK\Desktop\train_job\train2.csv", index=False, sep=',')
else:
    trains = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\train2.csv")

if not os.path.isfile(r"C:\Users\KMK\Desktop\train_job\train3.csv"):
    job_companies = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\job_companies.csv")
    trains = getJobCompanySize(trains, job_companies)
    if trains is None:
        print("getJobCompanySize return => None")
        exit(0)
    else:
        trains.to_csv(r"C:\Users\KMK\Desktop\train_job\train3.csv", index=False, sep=',')
else:
    trains = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\train3.csv")

# companySize에 따라 지원 횟수가 늘어나는지 검증해본다.
# x축은 NaN, 1-10, 11-50, 51-100, 101-200 인지 체크해본다.
# y축은 지원자의 기술스택이 회사가 요구하는 기술스택와 일치한비율을 나타낸다.

seaborn.stripplot(x="companySize", y="UserTagRatio", hue="applied", data=trains)
plt.show()
