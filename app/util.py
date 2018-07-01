# _*_ coding:utf-8 _*_
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import random
import time

def score(M, N):
    '''
    description: get your model score, higher score represent better affect
    :param M: List, refer to predictive active user id
    :param N: List, refer to real active user id
    :return: final score, higher score represent better affect,
                if M or N has no element or precision add recall equals zero ,
                return -1
    '''
    intersection = (set(M) & set(N))
    
    try:
        
        precision = len(intersection) / len(M)
        recall = len(intersection) / len(N)
        finalScore = (2 * precision * recall) / (precision + recall)
    
    except ZeroDivisionError:
        # if M or N has no element or precision add recall equals zero, return -1
        return -1
    
    return finalScore


def testScore(M, intersection, N=23722):
    '''
    description: get your model score, higher score represent better affect
    :param M: List, refer to predictive active user id
    :param N: List, refer to real active user id
    :return: final score, higher score represent better affect,
                if M or N has no element or precision add recall equals zero ,
                return -1
    '''

    try:

        precision = intersection / M
        recall = intersection / N
        finalScore = 2 * (precision * recall) / (precision + recall)

    except ZeroDivisionError:
        # if M or N has no element or precision add recall equals zero, return -1
        return -1

    return finalScore


def getInfo(m_num, F1, n_num=23722):
    '''
    description: get online information after submitting prediction file in the
    terms of F1 score, include numbers of right predictions, online precision
    and recall.
    :param m_num: refer to the number of predictive active user id
    :param F1: final score, get from online submitting
    :param n_num: refer to the number of real active user id
    :return: none
    '''
    rightPrediction = int(F1*(m_num + n_num) / 2)
    precision = F1*(m_num + n_num) / (2*m_num)
    recall = F1*(m_num + n_num) / (2*n_num)
    print("right prediction number: {0}, \nprecision: {1:.5}, \nrecall: {2:.5}"
          .format(rightPrediction, precision, recall))


def dataSpite(launchData, activityData, videoData, start, end):
    pass


def grid():
    pass


def slideWindow(blendData=None, step=2, trainSize=7, predictSize=7,
                totalDays=30, effectiveDay = 11):
    pass

def compare(file1, file2, flag=False):
    df1=pd.read_table(file1,names=["user_id"])
    df2=pd.read_table(file2,names=["user_id"])
    df3=pd.merge(df1,df2,how="inner")
    df4 = df3['user_id'].values
    print("file 1: {0}, file 2: {1}, and common: {2}".
          format(len(df1), len(df2), len(df3)))
    if flag:
        pre_len = "_" + str(len(df3))
        currentTime = time.strftime("%Y%m%d_%H_%M", time.localtime())
        with open("../result/C_result" + str(currentTime) +
                          pre_len + ".txt", "w") as file:
            for data in df4:
                file.write(str(data) + '\n')


class RnnData(object):
    def __init__(self):
        self.path = "../app_v2/src/finalSecondTable4.csv"
        self.data = pd.read_csv(self.path)
        self.features_attr = [ 'day',  'total_launch', 'total_activity',
                               'total_video','page_0', 'page_1', 'page_2',
                               'page_3', 'page_4','action_type_0',
                               'action_type_1', 'action_type_2','action_type_3',
                               'action_type_4', 'action_type_5']
        self.features_col = 15
        self.features_row = 23

        # self.path_ab = "../src/combine2.csv"
        # self.data_ab = pd.read_csv(self.path_ab)


    def get_validation_data(self):
        validation_data, validation_label = self.get_data('validation')
        return validation_data, validation_label


    def get_test_data(self):
        test_data = self.data.loc[(self.data['day'] >= (31 - self.features_row))]
        # test_data.loc[(test_data['day'] < test_data['register_day']),
        #               self.features_attr] = -1

        id_data = test_data[['user_id']].drop_duplicates()['user_id'].tolist()
        num_test_data = len(id_data)
        # z-score normalize
        # test_data = (test_data - test_data.mean())/(test_data.std())


        test_data = test_data[self.features_attr].values
        test_data = test_data[:, :self.features_col].reshape(num_test_data,
                                        self.features_col*self.features_row)
        return test_data, id_data

    def get_train_batch(self, batch_size):
        train_data, label, num_train_data = self.get_data('train')
        i = 0
        while True:
            X = train_data[i:i+batch_size]
            Y = label[i:i+batch_size]
            yield (X, Y)
            i = i + batch_size
            if i >= num_train_data - batch_size:
                i = random.randint(0, batch_size)

    def get_data(self, data_type):
        # create label data
        label_data = self.data.loc[(self.data['register_day'] <= self.features_row)&
                                   (self.data['day'] >= (self.features_row + 1))]
        label_attr = ['total_launch', 'total_activity', 'total_video']
        label_grouped = label_data[['user_id'] + label_attr].groupby('user_id')
        label_count = (label_grouped['total_launch'].sum()
                      + label_grouped['total_activity'].sum()
                      + label_grouped['total_video'].sum()).reset_index()
        label_count.columns = ['user_id', 'label']
        # print(label_count.describe([.25, .50, .55, .60, .65,
        #                             .70, .75, .80, .85, .90]))
        label_count['label'] = label_count['label'].apply(self.classify_label)


        # create train data
        train_data = self.data.loc[(self.data['register_day'] <= self.features_row)&
                                   (self.data['day'] <= self.features_row )]
        # train_data.loc[(train_data['day'] < train_data['register_day']),
        #               self.features_attr] = -1

        # z-score normalization
        # train_data = (train_data - train_data.mean())/(train_data.std())

        num_train_data = len(train_data['user_id'].drop_duplicates())
        # convert np.array [x * 17]
        train_data = train_data[self.features_attr].values
        train_data = train_data[:, :self.features_col].reshape(num_train_data,
                                        self.features_col*self.features_row)
        label = label_count['label'].tolist()

        if data_type == 'validation':
            validation_data = train_data[::50,:]
            validation_label = label[::50]
            return validation_data, validation_label
        else:
            return train_data, label, num_train_data


    def classify_label(self, x):
        '''
        divide the activity into 6 levels.
        :param x: amount of 3 kinds of behaviors in a week.
        :return: label classification
        '''
        if x < 1.0:
            return [1, 0, 0, 0, 0, 0]
        elif x < 3.0:
            return [0, 1, 0, 0, 0, 0]
        elif x < 20.0:
            return [0, 0, 1, 0, 0, 0]
        elif x < 100.0:
            return [0, 0, 0, 1, 0, 0]
        elif x < 500.0:
            return [0, 0, 0, 0, 1, 0]
        else:
            return [0, 0, 0, 0, 0, 1]



if __name__ == "__main__":
 

    print('ok')
