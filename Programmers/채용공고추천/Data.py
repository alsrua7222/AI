import pandas as pd


def IsPandasDataFrame(trains):
    """
    :param trains: 학습 파일 데이터
    :return bool: Pandas DataFrame 객체 이면 True 아니라면 False
    """
    # trains type이 pandas 아니라면 컷트.
    if type(trains) != type(pd.DataFrame()):
        return False
    return True


def getUserTagRatio(trains, job_tags, user_tags) -> pd.DataFrame:
    """
    :param train: 학습 파일 데이터
    :param tag: 태그 파일 데이터
    :return pd.DataFrame: train데이터에 UserTagRatio 열을 추가한 pandas 객체 생성 및 반환.
    :except param::train is not pandas class: None 반환.
    """

    if not IsPandasDataFrame(trains):
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
    return result


def getJobCompanySize(trains, job_companies: pd.DataFrame) -> pd.DataFrame:
    """
    :param train: 학습 파일 데이터
    :param tag: 회사규모 파일 데이터
    :return pd.DataFrame: train데이터에 companySize 열을 추가한 pandas 객체 생성 및 반환.
    :except param::train is not pandas class: None 반환.
    """
    if not IsPandasDataFrame(trains):
        return None

    result = trains.copy()

    companySize = []
    for jobID in trains['jobID'].values:
        companySize.append(job_companies[job_companies['jobID'] == jobID]['companySize'].values[0])

    # 범주형 순서값은 전부 수치형 순서로 바꿔준다.
    # 범주형 목록 - nan, 1-10, 11-50, 51-100, 101-200, 201-500, 501-1000, 1000 이상
    # 수치형 목록 - 0, 1, 2, 3, 4, 5, 6, 7
    for i in range(len(companySize)):
        if type(companySize[i]) == type(float()):
            companySize[i] = 0
        elif '501-100' in companySize[i]:
            companySize[i] = 6
        elif '201-50' in companySize[i]:
            companySize[i] = 5
        elif '101-20' in companySize[i]:
            companySize[i] = 4
        elif '51-10' in companySize[i]:
            companySize[i] = 3
        elif '11-50' in companySize[i]:
            companySize[i] = 2
        elif '1-10' in companySize[i]:
            companySize[i] = 1
        else:
            companySize[i] = 7

    result.loc[:, 'companySize'] = companySize
    return result


def getUserTagCounts(trains, user_tags) -> pd.DataFrame:
    if not IsPandasDataFrame(trains):
        return None

    result = trains.copy()

    # 유저 태그 수집하면서 기록.
    UserTagCounts = []
    User_Tags = {}
    for userID in trains['userID'].values:
        if userID not in User_Tags:
            User_Tags[userID] = 0
            tmp = set()
            tmp.update(user_tags[user_tags['userID'] == userID]['tagID'].values)
            User_Tags[userID] = len(tmp)
        UserTagCounts.append(User_Tags[userID])

    result.loc[:, 'UserTagCounts'] = UserTagCounts
    return result
