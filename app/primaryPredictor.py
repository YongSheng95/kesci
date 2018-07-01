# _*_ coding:utf-8 _*_

import lightgbm as lgb
import pandas as pd
import app.util as util

def primaryPredict(data=None, totalDay=30):
    
    featureNames = ["user_id", "device_type","register_type",
                    "feature_day_7_launch", "feature_day_7_activity", "feature_day_7_video",
                    "feature_day_5_7_launch", "feature_day_5_7_activity", "feature_day_5_7_video",
                    "feature_day_3_7_launch", "feature_day_3_7_activity", "feature_day_3_7_video",
                    "feature_day_1_7_launch", "feature_day_1_7_activity", "feature_day_1_7_video",
                    "feature_day_1-3_total", "feature_day_3-5_total", "feature_day_5-7_total",
                    "feature_trend"]
    featureUse = ["feature_day_7_launch", "feature_day_7_activity",
                  "feature_day_5_7_launch", "feature_day_5_7_activity",
                  "feature_day_3_7_launch", "feature_day_3_7_activity",
                  "feature_day_1_7_launch", "feature_day_1_7_activity", "feature_day_1_7_video",
                  "feature_trend"]

    columns = featureNames + ["Y"]
    
    # get train data
    trainData = pd.DataFrame(columns=columns)
    slide = util.slideWindow(data, totalDays=totalDay)
    for train, predictLabel in slide:
        # data statistics
        trainData = trainData.append(dataStatistics(train, columns, predictLabel))
    trainData = trainData.reset_index(drop=True).fillna(0)

    # get predict data
    predictData = pd.DataFrame(columns=columns)
    predictData = predictData.append(dataStatistics(data.loc[(data["register_day"] <= totalDay - 6) &
                                               (data["day"] >= totalDay - 6) &
                                               (data["day"] <= totalDay)], columns)).reset_index(drop=True)
    model = trainProcess(trainData, featureUse)
    predict = predictProcess(predictData, featureUse, model)
    
    # regression problem, need to translate to class problem, when result > 1, means activity user
    predict = predict.loc[predict["result"] >= 1]
    return predict["user_id"].unique().tolist()
    
    
def trainProcess(data, featureUse):
    '''
    train process
    :param data: train data
    :param featureUse: feature to use in train process
    :return: a trained model
    '''
    model_lgb = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=20,
                                  learning_rate=0.01, n_estimators=512, max_bin=128,
                                  min_data_in_leaf=20, min_sum_hessian_in_leaf=2)
    model_lgb.fit(data[featureUse].fillna(-1), data['Y'])
    return model_lgb
    
def predictProcess(data, featureUse, model):
    '''
    use model and data predict further data
    :param data: input data
    :param featureUse: feature to use in predict process
    :param model: a trained model
    :return: DataFrame column = (user_id, result)
    '''
    resultData = model.predict(data[featureUse].fillna(-1))
    result = pd.DataFrame(data["user_id"])
    result["result"] = resultData
    return result

def dataStatistics(trainData, featureNames, labelData = None):
    '''
    statistics data
    :param trainData:DataFrame
    :param predictData: DataFrame
    :return: DataFrame
    '''
    
    day = trainData["day"].unique()
    
    # feature_device_type and register type
    featureDeviceAndRegisterType = trainData[["user_id","device_type","register_type"]].drop_duplicates()
    
    # feature app, activity, video, behind 7
    featureStatistics7 = trainData.loc[trainData["day"] >= day[-1]].groupby("user_id") \
                                    ["total_launch", "total_activity", "total_video"].sum().reset_index()
    
    # feature app, activity, video, behind 5
    featureStatistics5 = trainData.loc[trainData["day"] >= day[-3]].groupby("user_id") \
        ["total_launch", "total_activity", "total_video"].sum().reset_index()

    # feature app, activity, video, behind 3
    featureStatistics3 = trainData.loc[trainData["day"] >= day[-5]].groupby("user_id") \
        ["total_launch", "total_activity", "total_video"].sum().reset_index()

    # feature app, activity, video, behind 1
    featureStatistics1 = trainData.loc[trainData["day"] >= day[-7]].groupby("user_id") \
        ["total_launch", "total_activity", "total_video"].sum().reset_index()
    
    # 7 days all time
    tmp = trainData.groupby("user_id")
    totalTime = (tmp["total_launch"].sum() + tmp["total_activity"].sum() + tmp["total_video"].sum())\
                        .reset_index()
    
    # feature trend 1-3
    tmp = trainData.loc[trainData["day"] <= day[2]].groupby("user_id")
    featureTrend1to3 = (tmp["total_launch"].sum() + tmp["total_activity"].sum() + tmp["total_video"].sum())\
                        .reset_index()
    featureTrend1to3[0] = featureTrend1to3[0]/totalTime[0]

    # feature trend 3-5
    tmp = trainData.loc[(trainData["day"] >= day[2]) & (trainData["day"] <= day[4])].groupby("user_id")
    featureTrend3to5 = (tmp["total_launch"].sum() + tmp["total_activity"].sum() + tmp["total_video"].sum())\
                        .reset_index()
    featureTrend3to5[0] = featureTrend3to5[0]/totalTime[0]

    # feature trend 5-7
    tmp = trainData.loc[(trainData["day"] >= day[4]) & (trainData["day"] <= day[6])].groupby("user_id")
    featureTrend5to7 = (tmp["total_launch"].sum() + tmp["total_activity"].sum() + tmp["total_video"].sum())\
                        .reset_index()
    featureTrend5to7[0] = featureTrend5to7[0]/totalTime[0]
    
    # featureTrend, see 7 days difference,
    # if day_x > day_(x+1) difference = 1, if day_x < day(x+1) difference = -1, if day_x == day_(x+1) difference = 1
    # sum one user 6 differences
    dataSum = trainData[["user_id", "day", "total_launch", "total_activity", "total_video"]]
    dataSum["total"] = trainData["total_launch"] + trainData["total_activity"] + trainData["total_video"]
    dataSum = dataSum[["user_id", "day", "total"]]
    dataSum["total_shift_1"] = dataSum["total"].shift(-1).fillna(0)
    dataSum["difference_total"] = dataSum["total_shift_1"] - dataSum["total"]
    dataSum = dataSum.drop(dataSum.loc[dataSum["day"] == day[-1]].index)
    dataSum.loc[dataSum["difference_total"] >= 0, "difference_total"] = 1
    dataSum.loc[dataSum["difference_total"] == 0, "difference_total"] = 0
    dataSum.loc[dataSum["difference_total"] < 0, "difference_total"] = -1
    featureTrend = dataSum.groupby("user_id")["difference_total"].sum().reset_index()[["user_id", "difference_total"]]
    
    # merge all feature
    featureData = featureDeviceAndRegisterType \
                    .merge(featureStatistics7, on="user_id") \
                        .merge(featureStatistics5, on="user_id") \
                            .merge(featureStatistics3, on="user_id") \
                                .merge(featureStatistics1, on="user_id") \
                                    .merge(featureTrend1to3, on="user_id") \
                                        .merge(featureTrend3to5, on="user_id") \
                                            .merge(featureTrend5to7, on="user_id") \
                                                .merge(featureTrend, on="user_id")
    
    # Y
    try:
        predictGroup = labelData.groupby("user_id")
        y = (predictGroup["total_launch"].sum()
            + predictGroup["total_activity"].sum()
            + predictGroup["total_video"].sum()).reset_index()
        y.columns = ["user_id", "Y"]
        y.loc[y["Y"] >= 1, "Y"] = 1
        # test
        y.loc[y['Y']<1, "Y"] = 0
        y.loc[((y['Y']>=1) & (y['Y']<13)), "Y"] =1
        y.loc[((y['Y'] >= 13) & (y['Y'] < 67)), "Y"] = 2
        y.loc[((y['Y'] >= 67) & (y['Y'] < 204)), "Y"] = 3
        y.loc[((y['Y'] >= 204) & (y['Y'] < 546)), "Y"] = 4
        y.loc[y['Y'] >= 546 , "Y"] = 5


        featureData = featureData.merge(y, on="user_id")
    except:
        featureData["Y"] = 0
        
    featureData.columns = featureNames
    return  featureData