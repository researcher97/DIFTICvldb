import csv
import time

import pandas as pd
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'

time_startpre= time.time()

from tensorflow import keras
import numpy as np
import pandas as pd
#load model
import csv
model = keras.models.load_model('models/pima.h5')
#model = keras.models.load_model('models/regression_house_model.h5')
#model = keras.models.load_model('models/housepredictionClassifier_model.h5')
#model = keras.models.load_model('models/bankcustomer_model.h5')
#model = keras.models.load_model('models/germanCredit_model.h5')

mu = [0.75, 0.90, 0.95, 0.99]
bowtie = ['>=','<=','>','<','==','!=']
l = len(model.layers)
outputConstraint=[] #output_constraints

w = np.array([])
Weight = []
Bias = []
Gamma = []
N = []
Gamma_tr =[]
X =[]
dense= []
for idx in range(len(model.layers)):
  if ("dense" in model.get_layer(index=idx).name):
    dense.append(idx)
l = len(dense)
for i in dense:
    w = model.layers[i].get_weights()[0]
    Weight.append(w)
    b = model.layers[i].get_weights()[1]
    Bias.append(b)
    a = model.layers[i].get_config()['activation']
    N.append(a)
    w_tr = np.transpose(w)
    A = np.matmul(w,w_tr)
    A_inv = np.linalg.inv(A)
    B = np.matmul(w_tr,A_inv)
    Gamma.append(B)
    B_tr = np.transpose(B)
    Gamma_tr.append(B_tr)

for i in range(l):
    for j in range(len(mu)):
        for k in range(len(bowtie)):
            M = np.matmul((Gamma_tr[i]*mu[j]), -(Bias[i]))
            print("X_", i + 1, "output_constraint:",bowtie[k], mu[j])
            print(M)

def beta(N,outputConstraint,l,i):
    p =0
    N0 = N[0]
    if l == 1:
        if (N0 == 'linear'):
            M = np.matmul((Gamma_tr[i] * outputConstraint), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], outputConstraint)
            print(M)

        if (N0 == 'sigmoid'):
            M = np.matmul((Gamma_tr[i] * np.log(outputConstraint / (1 - outputConstraint))), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], outputConstraint)
            print(M)

        if (N0 == 'tanh'):
            n_tanh = abs((outputConstraint - 1) / (outputConstraint + 1))
            M = np.matmul((Gamma_tr[i] * (0.5 * np.log(n_tanh))), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], outputConstraint)
            print(M)

        if (N0 == 'relu'):
            M1 = np.matmul((Gamma_tr[i] * outputConstraint), -(Bias[i]))
            M2 = np.matmul(Gamma_tr[i], -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], outputConstraint)
            print(M1)
            print(M2)
            M = M2
        return M
    else:
        N0 = [N[0]]
        N1 = N[1:]

        l = len(N1)
        featureConstraint2 = beta(N1,outputConstraint,l,i)
        i = i - 1
        outputConstraint =featureConstraint2

        while (i >=0):
            featureConstraint1 = beta(N0, outputConstraint, 1, i)
            i = i - 1
            outputConstraint =featureConstraint1
        return outputConstraint
outputConstraint=mu[2]
i=l-1
featureConstraint = beta(N,outputConstraint,l,i)

file = 'testData/pimatest.csv'

with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    names =reader.fieldnames
    print(reader.fieldnames)
    names = names[:-1]

    features = np.array([])
    for i in names:
        features = np.append(features, i)

if(featureConstraint.size == features.size):
    Featdictionary = dict(zip(features, featureConstraint))
    print(Featdictionary)
    for key, value in Featdictionary.items():
        print(key)

else:
    feature_counter = 1
    positive = 0
    negative = 0
    feature_count = np.array([])
    for i in featureConstraint:
        print("feature_counter",feature_counter, '>=', "{0:.2f}".format(i) )
        feature_count = np.append(feature_count,feature_counter)
        feature_counter = feature_counter + 1

    Featdictionary = dict(zip(feature_count, featureConstraint))

WPdict = Featdictionary

print(WPdict)

file = 'testData/pimatest.csv'
#file = 'testData/HousingData_test.csv'
#file = 'testData/housepricedata_test.csv'
#file = 'testData/bankcustomer_test.csv'
#file = 'testData/german_test.csv'
with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    fields =reader.fieldnames
    print(reader.fieldnames)
    features = fields[:-1]
    features = fields
    print(features)
    output = fields[-1]
    print(output)

df = pd.read_csv(file, sep=',')
print(df.values)

rslt_df = df

print(rslt_df.size)
#drop last column
rslt_df = rslt_df.iloc[:,:-1]
#drop first column
#rslt_df = df.iloc[: , 1:]
print(rslt_df.size)

print(rslt_df)

for key, value in WPdict.items():
    print(key, '->', value)
    #bowtieition
    rslt_df[key] = rslt_df[key].map(lambda x: "Y" if x >= value else "N")
     #print(key, '->', value)
    print(rslt_df)

#Column-wise FeatureConstraints violation count
FeatureConstraints_violation_count =rslt_df.apply(lambda x: x.str.contains("N")).sum()

FeatureConstraints_violation_count =rslt_df.apply(lambda x: x.str.contains("N")).mean()

global_viol = FeatureConstraints_violation_count.to_dict()

for key, value in global_viol.items():
    print(key, '->', value)

print("FeatureConstraints_violation_count Table:")


WP_satified_count=rslt_df[rslt_df.apply(lambda x: x.str.contains('Y'))].count()
#
sortedDict = sorted(global_viol.items(), key=lambda x:x[1])

print(sortedDict)

for key, value in sortedDict:
     print(key, '->', value)

for key, value in sortedDict:
     print(key)

elapsed_timePerm = time.time() - time_startpre

print(elapsed_timePerm)
