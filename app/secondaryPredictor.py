# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def secondaryPredict(data = None, print_log=False):
    '''
    use rf and lr model predict active users.
    :param data: pandas DataFrame, come from finalTable.csv
    :param print_log: default False, if true, it will print log
    :return: list, prediction
    '''

    firstPrediction = predict25To28Days(data, print_log)
    secondPrediction = predict29and30days(data, print_log)
    firstPrediction.extend(secondPrediction)

    if print_log:
        print("first prediction number(25-28 days): ",
              len(firstPrediction)-len(secondPrediction))
        print("second prediction number(29-30 days): ",len(secondPrediction))
        print("prediction number(25-30 days): ",len(firstPrediction))

    return firstPrediction



def predict25To28Days(data, print_log=False):
    '''
    use RandomForestClassifier model to predict active users who's register_day
    is from 25 to 28.
    :param data: pandas DataFrame, come from finalTable.csv
    :param print_log: default False, if true, it will print log
    :return: list, prediction
    '''

    # parsing data
    trainData, validationData, testData = genData(data, "rf")
    x_train = np.array(trainData)[:, :3].tolist()
    y_train = np.array(trainData)[:, 3
              ].tolist()
    id_test = np.array(testData)[:, 0].tolist()
    data_test = np.array(testData)[:, 1:].tolist()

    # build RandomForestClassifier model
    rf = RandomForestClassifier(n_jobs=4)

    # train model
    rf.fit(x_train, y_train)

    # validate model
    if print_log:
        x_validation = np.array(validationData)[:, :3].tolist()
        y_validation = np.array(validationData)[:, 3].tolist()
        acc = rf.score(x_validation, y_validation)
        print(" rf validation acc:", acc)
        print(" rf train data number: ", len(x_train))
        print(" 25-28 days all user number:", len(id_test))

    # predict
    prediction = rf.predict(data_test)

    dfMap = {"user_id": id_test, "prediction": prediction}
    df = pd.DataFrame(dfMap)
    df = df.loc[df["prediction"] == 1]
    predictions = df["user_id"].tolist()

    return predictions





def predict29and30days(data, print_log=False):
    '''
    use LogisticRegression model to predict active users who's register_day
    is from 29 and 30.
    :param data: pandas DataFrame, come from finalTable.csv
    :param print_log: default False, if true, it will print log
    :return: list, prediction
    '''
    # parsing data
    trainData, validationData, testData = genData(data, "lr")
    x_train = np.array(trainData)[:, :2].tolist()
    y_train = np.array(trainData)[:, 2].tolist()
    id_test = np.array(testData)[:, 0].tolist()
    data_test = np.array(testData)[:, 1:].tolist()

    # build LogisticRegression model
    lr = LogisticRegression()

    # train model
    re = lr.fit(x_train, y_train)

    # validate model
    if print_log:
        x_validation = np.array(validationData)[:, :2].tolist()
        y_validation = np.array(validationData)[:, 2].tolist()
        acc = re.score(x_validation, y_validation)
        print(" lr validation acc:", acc)
        print(" lr train data number: ", len(x_train))
        print(" 29-30 days all user number:", len(id_test))
    # predict
    prediction = re.predict(data_test)

    dfMap = {"user_id": id_test, "prediction": prediction}
    df = pd.DataFrame(dfMap)
    df = df.loc[df["prediction"] == 1]
    predictions = df["user_id"].tolist()

    return predictions




def genData(data, model_type="lr", proportion=0.8):
    '''
    create train, validation and test data for LogisticRegression model.
    :param data: DataFrame, read from merged_data by pandas
    :param proportion: proportion of train data
    :return: tuple, train, validation and test data
    '''

    countAttr = ['user_id', 'total_launch', 'total_activity', 'total_video']
    registerDeviceAttr = ['user_id', 'register_type', 'device_type']

    # create data for RandomForest model
    if model_type=="rf":
        dataMoreThan10Days = data.loc[data['register_day'] <= 21]
        # near 3 days for feature from register day
        featureData = dataMoreThan10Days.loc[(dataMoreThan10Days['day']
                        >= dataMoreThan10Days['register_day'])
            &(dataMoreThan10Days['day']<=dataMoreThan10Days['register_day']+2)]
        # next 7 days for label
        labelData = dataMoreThan10Days.loc[(dataMoreThan10Days['day']
                        > dataMoreThan10Days['register_day'] + 2)
            &(dataMoreThan10Days['day']<=dataMoreThan10Days['register_day']+9)]

        # a table include user_id, register_type and device_type
        registerDeviceFeature = featureData[registerDeviceAttr].drop_duplicates()

        # create feature_3_days
        countGrouped = featureData[countAttr].groupby('user_id')
        feature3Days = (countGrouped['total_launch'].sum()
                        + countGrouped['total_activity'].sum()
                        + countGrouped['total_video'].sum()).reset_index()
        feature3Days.columns = ['user_id', 'feature_3_days']

        # create label
        labelGrouped = labelData[countAttr].groupby('user_id')
        labelCount = (labelGrouped['total_launch'].sum()
                        + labelGrouped['total_activity'].sum()
                        + labelGrouped['total_video'].sum()).reset_index()
        labelCount.columns = ['user_id', 'label']
        labelCount.loc[labelCount['label'] >= 1, "label"] = 1
        labelCount.loc[labelCount['label'] < 1, "label"] = 0

        # default merge table, key = user_id
        table = pd.merge(registerDeviceFeature, feature3Days)
        table = pd.merge(table, labelCount)
        table = table[['register_type','device_type','feature_3_days', 'label']]
        temp = np.array(table).tolist()

        # create train and validation data
        shuffle(temp)
        numOfValidationData = int(len(temp)*proportion)
        trainData = temp[:numOfValidationData]
        validationData = temp[numOfValidationData:]
        del temp

        # create test data for 25 to 28 day
        data25To28Days = data.loc[(data['register_day'] > 24)
            & (data['register_day'] <= 28)]
        registerDeviceFeature = data25To28Days[
            registerDeviceAttr].drop_duplicates()
        featureTestData = data25To28Days.loc[(data25To28Days['day']
                        >= data25To28Days['register_day'])
            & (data25To28Days['day'] <= data25To28Days['register_day'] + 2)]

        # create feature_3_days for test
        countGrouped = featureTestData[countAttr].groupby('user_id')
        feature3Days = (countGrouped['total_launch'].sum()
                        + countGrouped['total_activity'].sum()
                        + countGrouped['total_video'].sum()).reset_index()
        feature3Days.columns = ['user_id', 'feature_3_days']
        table = pd.merge(registerDeviceFeature, feature3Days)
        table = table[['user_id', 'register_type','device_type','feature_3_days']]
        testData = np.array(table).tolist()

        return trainData, validationData, testData

    # create data for LogisticRegression model
    if model_type=='lr':
        dataMoreThan6Days = data.loc[data['register_day'] < 25]
        data29and30Days = data.loc[data['register_day'] >= 29]

        newData=dataMoreThan6Days.loc[(dataMoreThan6Days['day']
                                       >dataMoreThan6Days['register_day'])
                              & (dataMoreThan6Days['day']
                                 <= dataMoreThan6Days['register_day']+7)]
        registerDevice=newData[registerDeviceAttr].drop_duplicates()

        featureData = dataMoreThan6Days.loc[dataMoreThan6Days['day']
                                == dataMoreThan6Days['register_day']]

        features = ['user_id', 'total_launch', 'total_activity', 'total_video',
                    'page_0', 'page_1', 'page_2', 'page_3', 'page_4',
                    'action_type_0', 'action_type_1', 'action_type_2',
                    'action_type_3', 'action_type_4', 'action_type_5']
        featureData = featureData[features]

        forCountTimes=newData[countAttr]
        count=(forCountTimes.groupby("user_id")["total_launch"].sum()
               +forCountTimes.groupby("user_id")["total_activity"].sum()
              +forCountTimes.groupby("user_id")["total_video"].sum()).reset_index()
        count.columns = ['user_id',"label"]
        count.loc[count['label'] >= 1, "label"] = 1
        count.loc[count['label'] < 1, "label"] = 0
        count=count[['user_id','label']]

        table=pd.merge(registerDevice,featureData)
        table = pd.merge(table, count)

        all_features = ['register_type', 'device_type',
                        'total_launch', 'total_activity', 'total_video',
                        'page_0', 'page_1', 'page_2', 'page_3','page_4',
                        'action_type_0', 'action_type_1', 'action_type_2',
                        'action_type_3', 'action_type_4', 'action_type_5',
                        'label']

        table=table[all_features]
        table = np.array(table)  # np.ndarray()
        temp= table.tolist()  # list
        # random choose data for training and validating
        shuffle(temp)
        numOfValidationData = int(len(temp) * proportion)
        trainData = temp[:numOfValidationData]
        validationData = temp[numOfValidationData:]
        del temp

        # create test data
        data29and30Days = data29and30Days[registerDeviceAttr].drop_duplicates()
        data29and30Days = np.array(data29and30Days)
        testData = data29and30Days.tolist()  # list
        return trainData, validationData, testData


if __name__=="__main__":

    path = "../src/finalTable.csv"
    data = pd.read_csv(path)
    # train, validation, test = genData(data, 'rf')
    # for i in range(5):
    #     print(train[i])
    # for i in range(5):
    #     print(test[i])
    # print("train: ", len(train))
    # print("test: ", len(test))

    prediction = secondaryPredict(data, True)

