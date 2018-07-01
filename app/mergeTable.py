# _*_ coding:utf-8 _*_

import pandas as pd

def combineTable():
    '''
    combine three table to one
    :return: None
    '''

    userRegisterData = pd.read_table("user_register_log.txt",
                                     names=['user_id', 'register_day', 'register_type', 'device_type'])

    '''''
        expand table
    '''''
    ID = userRegisterData["user_id"].unique()
    addID = pd.DataFrame({'user_id': ID})

    expandTable = pd.DataFrame(index=["user_id", "day"])
    for i in range(30):
        tmp = pd.DataFrame({"user_id": addID["user_id"], "day": i + 1})
        expandTable = expandTable.append(tmp, ignore_index=True)
    expandTable = expandTable.dropna().sort_values(["user_id", "day"]).reset_index(drop=True)

    '''''
        combine 1
    '''''
    finalTable = pd.merge(userRegisterData,expandTable,how="left")

    '''''
        simplify app_lunch table
    '''''
    appLaunchData = pd.read_table("app_launch_log.txt",names=['user_id', 'day'])
    ldf1 = appLaunchData.groupby(["user_id",'day']).size().reset_index()
    columns = ["user_id", "day", "total_launch"]
    ldf1.columns = columns

    '''''
        combine 2
    '''''
    ldf2 = pd.merge(finalTable,ldf1,how='left')

    '''''
            simplify user_activity_log table
    '''''
    userActivity = pd.read_table("user_activity_log.txt",
                                 names=["user_id","day","page","video_id","author_id","action_type"])

    actData2 = userActivity[["user_id","day"]]

    adf1 = actData2.groupby(["user_id","day"]).size().reset_index()
    columns = ["user_id", "day", "total_activity"]
    adf1.columns = columns

    '''''
           combine 3
    '''''
    adf2=pd.merge(ldf2,adf1,how="left")

    '''''
        simplify video_create_log table
    '''''
    videoCreateData = pd.read_table("video_create_log.txt",names=["user_id","day"])

    cdf1 = videoCreateData.groupby(["user_id","day"]).size().reset_index()
    columns = ["user_id", "day", "total_video"]
    cdf1.columns = columns
    '''''
        combine 4
    '''''
    cdf2 = pd.merge(adf2,cdf1,how="left").sort_values(["user_id", "day"]).reset_index(drop=True)
    cdf2 = cdf2.fillna(0)

    '''''
        save
    '''''
    cdf2.to_csv("finalTable.csv")

