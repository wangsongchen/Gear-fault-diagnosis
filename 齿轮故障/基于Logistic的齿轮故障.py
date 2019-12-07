# -*- coding:UTF-8 -*-
import numpy as np
import random
import math

def handleText():
    faultText1=open('E:/matlab数据/辛辛那提轴承数据/原始数据集.csv')
    faultTextSet1 = []
    for line in faultText1.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)):
            lineArr.append(float(currLine[i]))
        faultTextSet1.append(lineArr)
    faultTextSet1=np.array(faultTextSet1)
    return faultTextSet1
def handleTrain():
    faultTrain1=open('E:/matlab数据/辛辛那提轴承数据/原始测试数据集.csv')
    faultTrainSet1 = []
    for line in faultTrain1.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)):
            lineArr.append(float(currLine[i]))
        faultTrainSet1.append(lineArr)
    faultTrainSet1=np.array(faultTrainSet1)
    return faultTrainSet1
def FFTTrain():
    fftTrain1=open('E:/matlab数据/辛辛那提轴承数据/fft测试数据集.csv')
    fftTrainSet1 = []
    for line in fftTrain1.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)):
            lineArr.append(float(currLine[i]))
        fftTrainSet1.append(lineArr)
    fftTrainSet1=np.array(fftTrainSet1)
    return fftTrainSet1
def FFTText():
    fftText1=open('E:/matlab数据/辛辛那提轴承数据/fft数据集.csv')
    fftTextSet1 = []
    for line in fftText1.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)):
            lineArr.append(float(currLine[i]))
        fftTextSet1.append(lineArr)
    fftTextSet1=np.array(fftTextSet1)
    return fftTextSet1
def mathTrain():
    filename = '测试数据特征.txt'
    faultTrainSet=handleTrain()
    fftTrainSet=FFTTrain()
    for i in range(32):
        faultTrainmath = faultTrainSet[:, i]
        fftTrainSetmath = fftTrainSet[:, i]
        Sum = 0
        Sum2 = 0
        Sum4 = 0
        Sum_2 = 0
        Sum_1 = 0
        sump = 0
        sump2 = 0
        sump3 = 0
        fk2 = 0
        fk4 = 0
        fk = 0
        sums = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for j in range(20480):
            Sum = Sum + float(faultTrainmath[[j]])
        X_Average = Sum / 20480
        with open(filename, 'a') as f:
            f.write(str(X_Average))
            f.write('\n')
        for j in range(20480):
            Sum2 = Sum2 + float(faultTrainmath[[j]]) ** 2
        X_rms = math.sqrt(Sum2 / 20480)
        with open(filename, 'a') as f:
            f.write(str(X_rms))
            f.write('\n')

        for j in range(20480):
            Sum4 = Sum4 + float(faultTrainmath[[j]]) ** 4
        β = Sum4 / 20480
        with open(filename, 'a') as f:
            f.write(str(β))
            f.write('\n')

        for j in range(20480):
            Sum_2 = Sum_2 + math.sqrt(math.sqrt(faultTrainmath[[j]] ** 2))
        X_r = (Sum_2 / 20480) ** 2
        with open(filename, 'a') as f:
            f.write(str(X_r))
            f.write('\n')

        for j in range(20480):
            Sum_1 = Sum_1 + (float(faultTrainmath[[j]]) - X_Average) ** 2
            σ2 = Sum_1 / 20479
        with open(filename, 'a') as f:
            f.write(str(σ2))
            f.write('\n')
        for j in range(10240):
            sump = sump + float(fftTrainSetmath[[j]])
        p1 = sump / 10240
        with open(filename, 'a') as f:
            f.write(str(p1))
            f.write('\n')

        for j in range(10240):
            sump2 = sump2 + (float(fftTrainSetmath[[j]]) - p1) ** 2
        p2 = sump2 / 10239
        with open(filename, 'a') as f:
            f.write(str(p2))
            f.write('\n')

        for j in range(10240):
            sump3 = sump3 + (float(fftTrainSetmath[[j]]) - p1) ** 3
        p3 = sump3 / (10240 * (math.sqrt(p2)) ** 3)
        with open(filename, 'a') as f:
            f.write(str(p3))
            f.write('\n')
        for j in range(10240):
            fk4 = fk4 + float(fftTrainSetmath[[j]]) * (j+1) ** 4
            fk2 = fk2 + float(fftTrainSetmath[[j]]) * (j+1) ** 2
        p4 = math.sqrt(fk4 / (fk2 ))
        with open(filename, 'a') as f:
            f.write(str(p4))
            f.write('\n')
        for j in range(10240):
            fk = fk + (j+1) * float(fftTrainSetmath[[j]])
            sums = sums + float(fftTrainSetmath[[j]])
        p5 = fk / sums
        with open(filename, 'a') as f:
            f.write(str(p5))
            f.write('\n')
        for j in range(10240):
            sum1 = sum1 + float(fftTrainSetmath[[j]]) * (j+1 - p5) ** 2
        p6 = math.sqrt(sum1 / 10240)
        with open(filename, 'a') as f:
            f.write(str(p6))
            f.write('\n')
        for j in range(10240):
            sum2 = sum2 + (j+1) * (j+1) * float(fftTrainSetmath[[j]])
            sum3 = sum3 + float(fftTrainSetmath[[j]])
            sum4 = sum4 + float(fftTrainSetmath[[j]]) * (j+1) ** 4
        p7 = sum2 / math.sqrt(sum3 * sum4 )
        with open(filename, 'a') as f:
            f.write(str(p7))
            f.write('\n')
        if i < 16:
            with open(filename, 'a') as f:
                f.write('+1')
                f.write('\n')
        else:
            with open(filename, 'a') as f:
                f.write('-1')
                f.write('\n')
def TrainwriteNp():
    b = np.loadtxt('测试数据特征.txt')

    b=np.array(b).reshape(32,13)
    #print(b)
    #np.savetxt("E:/matlab数据/辛辛那提轴承数据/b.txt", b, fmt='%f', delimiter=',')

    return b

def mathText():
    faultTextSet=handleText()
    fftTextSet=FFTText()
    filename='数据特征.txt'

    for i in range(160):
        faultTextmath=faultTextSet[:,i]
        fftTextSetmath=fftTextSet[:,i]
        Sum = 0
        Sum2 = 0
        Sum4 = 0
        Sum_2=0
        Sum_1=0
        sump=0
        sump2=0
        sump3=0
        fk2=0
        fk4=0
        fk=0
        sums=0
        sum1=0
        sum2=0
        sum3=0
        sum4=0
        for j in range(20480):

            Sum=Sum+float(faultTextmath[[j]])
        X_Average=Sum/20480
        with open(filename,'a') as f:
            f.write(str(X_Average))
            f.write('\n')
        for j in range(20480):
            Sum2=Sum2+float(faultTextmath[[j]])**2
        X_rms=math.sqrt(Sum2/20480)
        with open(filename, 'a') as f:
            f.write(str(X_rms))
            f.write('\n')

        for j in range(20480):
            Sum4=Sum4+float(faultTextmath[[j]])**4
        β=Sum4/20480
        with open(filename, 'a') as f:
            f.write(str(β))
            f.write('\n')

        for j in range(20480):
            Sum_2=Sum_2+math.sqrt(math.sqrt(faultTextmath[[j]]**2))
        X_r=(Sum_2/20480)**2
        with open(filename, 'a') as f:
            f.write(str(X_r))
            f.write('\n')

        for j in range(20480):
            Sum_1=Sum_1+(float(faultTextmath[[j]])-X_Average)**2
        σ2=Sum_1/20479
        with open(filename, 'a') as f:
            f.write(str(σ2))
            f.write('\n')
        for j in range(10240):
            sump =sump + float(fftTextSetmath[[j]])
        p1 = sump / 10240
        with open(filename, 'a') as f:
            f.write(str(p1))
            f.write('\n')

        for j in range(20480):
            sump2 = sump2+(float(fftTextSetmath[[j]])-p1)**2
        p2=sump2/10239
        with open(filename, 'a') as f:
            f.write(str(p2))
            f.write('\n')

        for j in range(10240):
            sump3=sump3+(float(fftTextSetmath[[j]])-p1)**3
        p3=sump3/(10240*(math.sqrt(p2))**3)
        with open(filename, 'a') as f:
            f.write(str(p3))
            f.write('\n')
        for j in range(10240):
            fk4=fk4+float(fftTextSetmath[[j]])*((j+1)**4)
            fk2=fk2+float(fftTextSetmath[[j]])*((j+1)**2)
        p4=math.sqrt(fk4/(fk2))
        with open(filename, 'a') as f:
            f.write(str(p4))
            f.write('\n')
        for j in range(10240):
            fk=fk+(j+1)*float(fftTextSetmath[[j]])
            sums=sums+float(fftTextSetmath[[j]])
        p5=fk/sums
        with open(filename, 'a') as f:
            f.write(str(p5))
            f.write('\n')
        for j in range(10240):
            sum1=sum1+float(fftTextSetmath[[j]])*((j-p5)**2)
        p6=math.sqrt(sum1/10240)
        with open(filename, 'a') as f:
            f.write(str(p6))
            f.write('\n')
        for j in range(10240):
            sum2=sum2+(j+1)*(j+1)*float(fftTextSetmath[[j]])
            sum3=sum3+float(fftTextSetmath[[j]])
            sum4=sum4+float(fftTextSetmath[[j]])*((j+1)**4)
        p7=sum2/math.sqrt(sum3*sum4)
        with open(filename, 'a') as f:
            f.write(str(p7))
            f.write('\n')
        if i<80:
            with open(filename, 'a') as f:
                f.write('+1')
                f.write('\n')
        else:
            with open(filename, 'a') as f:
                f.write('-1')
                f.write('\n')


def writeNp():
    a = np.loadtxt('数据特征.txt')

    a=np.array(a).reshape(160,13)



    #np.savetxt("E:/matlab数据/辛辛那提轴承数据/a.txt", a, fmt='%f', delimiter=',')
    return a
def Test():
    a=writeNp()
    b=TrainwriteNp()

    trainingSet=a[:,0:12]
    trainingLabels=a[:,[12]]
    trainingLabels = np.array(trainingLabels).flatten()


    trainWeights = gradAscent(trainingSet, trainingLabels)

    errorCount = 0;numTestVec = 32
    for i in range(32):
        textSet=b[i,0:12]
        textLable=b[i,[12]]
        textLable=np.array(textLable).flatten()

        if int(classifyVector(np.array(textSet), trainWeights[:, 0])) != int(textLable[0]):
            errorCount += 1
        print(textSet,classifyVector(np.array(textSet), trainWeights[:, 0]),int(textLable[0]))



    errorRate = (float(errorCount) / numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)
    return trainWeights, errorRate


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    #print(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


if __name__ == '__main__':
    handleText()
    mathText()
    writeNp()
    handleTrain()
    mathTrain()
    TrainwriteNp()
    Test()