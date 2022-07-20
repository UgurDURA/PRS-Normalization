from statistics import variance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import scipy.integrate

df = pd.read_csv("sample.txt", sep=",")

# print(df)

scores = df.loc[:,'PGS000018']

# print(scores)

scores_mean = scores.mean()
scores_Std = scores.std()

# print("Mean of the scores: ", scores_mean)
# print("Standart Deviation of the scores: ", scores_Std)


Z_Scores_Data = np.empty(shape=[0])


for i in range(scores.size):

    result = (scores[i]-scores_mean)/scores_Std
    result = round(result, 2)
    print("No: ",i, " Z-Score : ", result)
    Z_Scores_Data = np.append(Z_Scores_Data,result)


Z_Scores = pd.DataFrame(Z_Scores_Data, columns= ["Z-Scores"])
# print(Z_Scores)

concResult = df.merge(Z_Scores, how='outer', left_index=True, right_index=True )

# print(concResult)


# ax = concResult.plot(x='sample', y='PGS000018', kind='kde', figsize=(10, 6))
# # get the x axis values corresponding to this slice (See beneath the plot)
# arr = ax.get_children()[0]._x
# # take the first and last element of this array to constitute the xticks and 
# # also rotate the ticklabels to avoid overlapping
# plt.xticks(np.linspace(arr[0], arr[-1]), rotation=90)
# plt.show()


###########################################################################################################################################################
                                                                 #Z-Table Generation
###########################################################################################################################################################


x = []
y = []
i = -4

while i<4:
    i += 0.01
    x.append(round(i,3))

e = 2.718281828459
pi = 3.14159265


c = 1/((2*pi)**0.5)

for a in x:

    expo = (-a**2)/2
    distro = c*(e**expo)
    y.append(distro)

# plt.plot(x,y)
# plt.show()

negative_infinity = -float('inf')

z = 0 

def f(x):
    c = 1/((2*pi)**0.5)
    exponent = (-x**2)/2
    standart_normal_curve = c*(e**exponent)
    return standart_normal_curve

probability, error = scipy.integrate.quad(f, negative_infinity, z)

# print(round(probability, 5))

z_table = []

for row in x:
    probability, error = scipy.integrate.quad(f, negative_infinity, row)
    z_table.append([row, round(probability, 5)])
    print(row, round(probability, 5))


def getMean(sample):
    sampleSize = len(sample)
    sumTotal = 0

    for row in sample:
        sumTotal = row + sumTotal
    mean = sumTotal/sampleSize
    return mean

def getPopulationSD(sample):
    sampleSize = len(sample)
    sumOfSquares = 0
    mean = getMean(sample)

    for row in sample:
        deviationScore = row - mean
        sumOfSquares = sumOfSquares + deviationScore**2
    variance = sumOfSquares/sampleSize
    populationSD = variance**0.5
    return populationSD

def getSampleSD(sample):
    sampleSize = len(sample)
    sumOfSquares = 0
    mean = getMean(sample)

    for row in sample:
        deviationScore = row - mean
        sumOfSquares = sumOfSquares + deviationScore**2
    variance = sumOfSquares/(sampleSize-1)
    sampleSD = variance**0.5
    return sampleSD

def getPopulationStandartError(sample):
    sampleSize = len(sample)
    populationSD = getPopulationSD(sample)

    SEM = populationSD/sampleSize**0.5
    return SEM

def getSampleStandartError(sample):
    sampleSize = len(sample)
    sampleSD = getSampleSD(sample)

    SEM = sampleSD / sampleSize**0.5
    return SEM

def getZscore(sample, population):
    sampleMean = getMean(sample)
    populationMean = getMean(sample)
    populationMean = getMean (population)
    SEM = getSampleStandartError(sample)
    Z = (sampleMean - populationMean) / SEM
    return Z



## Population Evaluation
def getProbabilityFromZ(Z):
    z = round(float(z),2)
    probability = 0

    if z < -4:
        return 0.000
    elif z > 4:
        return 0.000
    
    else:
        for row in z_table:
            if z == float(row[0]):
                probability = float(row[1])
            
    return probability


## Sample Evaluation

def getProbabilityFromZ_Sample(PGS, sampleID, Z):
    percentile = 0

    for row in z_table:
        if float(row[0] == Z):
            probability = row[1]
            percentile = round(float(probability)*100,2)
            print(sampleID, " has probability of ", percentile, "% to have ", PGS )
            break
    return percentile


Percentiles= np.empty(shape=[0])

for i in range(concResult['Z-Scores'].size):
    Percentiles = np.append(Percentiles, getProbabilityFromZ_Sample(concResult.columns[1], concResult["sample"].values[i], concResult["Z-Scores"].values[i]))


# print(Percentiles)

Percentiles = pd.DataFrame(Percentiles, columns= ["Probability"])

PercentileResults = concResult.merge(Percentiles, how='outer', left_index=True, right_index=True )

print(PercentileResults)
    

 

