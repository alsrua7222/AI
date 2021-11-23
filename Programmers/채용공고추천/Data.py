import pandas as pd

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


def IsPandasDataFrame(trains):
    # trains type이 pandas 아니라면 컷트.
    if type(trains) != type(pd.DataFrame()):
        return False
    return True


def getJobCompanySize(trains, job_companies) -> pd.DataFrame:

    if not IsPandasDataFrame(trains):
        return None
    result = trains.copy()

    companySize = []
    for jobID in trains['jobID'].values:
        companySize.append(job_companies[job_companies['jobID'] == jobID]['companySize'].values)
    print(companySize)

    result.loc[:, 'companySize'] = companySize
    return result
