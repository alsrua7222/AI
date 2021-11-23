import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# job_companies = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\job_companies.csv")
trains = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\train.csv")
tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\tags.csv")
job_tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\job_tags.csv")
user_tags = pd.read_csv(r"C:\Users\KMK\Desktop\train_job\user_tags.csv")

# 데이터 분석 결과
# 지원자 총 196명.

# train.csv
# # trains applied == 1의 총 개수 => 857 rows
# print(trains[trains['applied'] == 1])
# # 지원자가 여러번 지원할 수 있다. 최다 지원 개수 106개, 최저 지원 개수 2개.
# print(trains['userID'].value_counts())

# user_tags
# # 데이터 안에 들어있는 user 수가 196명.
# # 유저 중 최다 기술스택 572개
# # 유저 중 최저 기술스택 4개
# tmp = user_tags['userID'].value_counts()
# print(tmp[0], tmp[-1])
# # 가장 많이 사용한 기술 스택 => Java
# # 가장 많이 사용허지 않는 기술 스택 => Apache Pig
# tmp2 = list(user_tags['tagID'].value_counts().items())
# print(tags[tags['tagID'] == tmp2[0][0]])
# print(tags[tags['tagID'] == tmp2[-1][0]])

def getUserTagRatio(trains, job_tags, user_tags):
    """
    :param train: 학습 파일 데이터
    :param tag: 태그 파일 데이터
    :return result: train데이터에 UserTagRatio 열을 추가한 pandas 객체 생성 및 반환.
    :except param::train is not pandas class: None 반환.
    """

    # trains type이 pandas 아니라면 컷트.
    if type(trains) != type(pd.DataFrame()):
        return None

    # 반환할 객체에 train 데이터 복사.
    result = trains.copy()

    # 지원자가 가지고 있는 기술 스택 수집.
    # 회사가 요구하는 기술 스택 수집.
    # 수집하는 이유는 (시간 절약을 위해 HashMap 활용).
    User_Tags = {}
    for userID in trains['userID'].values:
        User_Tags[userID] = set()
        User_Tags[userID].update(user_tags[user_tags['userID'] == userID]['tagID'].values)
    Job_Tags = {}
    for jobID in trains['jobID'].values:
        Job_Tags[jobID] = set()
        Job_Tags[jobID].update(job_tags[job_tags['jobID'] == jobID]['tagID'].values)

    # 1부터 끝까지 탐색하면서 비율 추가한다.
    UserTagRatio = []
    for userID, jobID, applied in trains.values:
        count = 0
        length = len(Job_Tags[jobID])
        for tagID in Job_Tags[jobID]:
            if tagID in User_Tags[userID]:
                count += 1
        UserTagRatio.append(count / length)

    result.loc[:, 'UserTagRatio'] = UserTagRatio
    return result.copy()

res = getUserTagRatio(trains, job_tags, user_tags)
for v in res.values[:5]:
    print(v)
