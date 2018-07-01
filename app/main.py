# _*_ coding:utf-8 _*_

import app.fileReader as fileReader
import time
from app.primaryPredictor import primaryPredict
from app.secondaryPredictor import secondaryPredict

def main():
    # datq reader
    data = fileReader.read_blend()
    
    # primary data predict
    primaryDataPredictResult = primaryPredict(data)
    
    print("prediction number(1-24 days):",len(primaryDataPredictResult))
    
    # secondary data predict
    secondaryDataPredictResult = secondaryPredict(data, True)
    
    # combine two results
    primaryDataPredictResult.extend(secondaryDataPredictResult)
    
    # result write
    currentTime = time.strftime("%Y%m%d_%H", time.localtime())
    with open("../result/result" +str(currentTime) + ".txt", "w") as file:
        for data in primaryDataPredictResult:
            file.write(str(data) + '\n')
    

if __name__=="__main__":
    main()
