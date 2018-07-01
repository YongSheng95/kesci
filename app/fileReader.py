# _*_ coding:utf-8 _*_

import pandas as pd

FINALTABLEPATH = "../src/finalTable4.csv"
LAUNCHPATH = "../src/app_launch_log.txt"
REGISTERPATH = "../src/user_register_log.txt"
ACTIVITYPATH = "../src/user_activity_log.txt"
VIDEOPATH = "../src/video_create_log.txt"

def read_register():
    head = ['user_id', 'register_day', 'register_type', 'device_type']
    return pd.read_table(REGISTERPATH, names=head)

def read_launch():
    head = ("user_id", "day")
    return pd.read_table(LAUNCHPATH, names=head)

def read_activity():
    head = ["user_id", "day", "page", "video_id", "author_id", "action_type"]
    return pd.read_table(ACTIVITYPATH, names=head)

def read_video():
    head = ["user_id", "day"]
    return pd.read_table(VIDEOPATH, names=head)

def read_blend(newBlend = False, path = FINALTABLEPATH):
    '''
    read blend data
    :param: newBlend, if = true, write new file to desk and return blend data
    :param: path, blend csv disk position
    :return: DataFrame
    '''
    if (newBlend == True):
        '''
        combine three table to one
        :return: None
        '''
    
        userRegisterData = read_register()
    
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
        finalTable = pd.merge(userRegisterData, expandTable, how="left")
    
        '''''
            simplify app_lunch table
        '''''
        appLaunchData = read_launch()
        ldf1 = appLaunchData.groupby(["user_id", 'day']).size().reset_index()
        columns = ["user_id", "day", "total_launch"]
        ldf1.columns = columns
    
        '''''
            combine 2
        '''''
        ldf2 = pd.merge(finalTable, ldf1, how='left')
    
        '''''
                simplify user_activity_log table
        '''''
        userActivity = read_activity()
    
        actData2 = userActivity[["user_id", "day"]]
    
        adf1 = actData2.groupby(["user_id", "day"]).size().reset_index()
        columns = ["user_id", "day", "total_activity"]
        adf1.columns = columns
    
        '''''
               combine 3
        '''''
        adf2 = pd.merge(ldf2, adf1, how="left")
    
        '''''
            simplify video_create_log table
        '''''
        videoCreateData = read_video()
    
        cdf1 = videoCreateData.groupby(["user_id", "day"]).size().reset_index()
        columns = ["user_id", "day", "total_video"]
        cdf1.columns = columns
        '''''
            combine 4
        '''''
        cdf2 = pd.merge(adf2, cdf1, how="left").sort_values(["user_id", "day"]).reset_index(drop=True)
        cdf2 = cdf2.fillna(0)
    
        '''''
            save and return
        '''''
        cdf2.to_csv(path, index=None)
        
        return cdf2
    else:
        return pd.read_csv(path)