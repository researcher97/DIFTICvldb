import csv
import time

#import featureConstraints
import pandas as pd
import featureImpGlobal
from collections import Counter
pd.options.mode.chained_assignment = None
import time
time_startpre= time.time()
from tensorflow import keras
import numpy as np
import pandas as pd
#load model
import csv
model = keras.models.load_model('modelsNew/pima_model.h5')
#model = keras.models.load_model('models/regression_house_model.h5')
#model = keras.models.load_model('models/housepredictionClassifier_model.h5')
#model = keras.models.load_model('models/bankcustomer_model.h5')
#model = keras.models.load_model('models/germanCredit_model.h5')

mu = [0.75, 0.90, 0.95, 0.99]
bowtie = ['>=','<=','>','<','==','!=']
l = len(model.layers)
Q=[] #output_constraints

w = np.array([])
Weight = [] #Storing Weights
Bias = [] #Storing Biases
Gamma = [] #Storing Weights
N = [] #Activation function list
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

def beta(N,Q,l,i):
    p =0
    N0 = N[0]
    if l == 1:
        if (N0 == 'linear'):
            M = np.matmul((Gamma_tr[i] * Q), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], Q)

        if (N0 == 'sigmoid'):
            M = np.matmul((Gamma_tr[i] * np.log(Q / (1 - Q))), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], Q)


        if (N0 == 'tanh'):
            n_tanh = abs((Q - 1) / (Q + 1))
            M = np.matmul((Gamma_tr[i] * (0.5 * np.log(n_tanh))), -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], Q)

        if (N0 == 'relu'):
            M1 = np.matmul((Gamma_tr[i] * Q), -(Bias[i]))
            M2 = np.matmul(Gamma_tr[i], -(Bias[i]))
            print("X_", i + 1, "output_constraint:", bowtie[0], Q)
            M = M2
        return M
    else:
        N0 = [N[0]]
        N1 = N[1:]

        l = len(N1)
        Featdictionary2 = beta(N1,Q,l,i)
        i = i - 1
        Q =Featdictionary2

        while (i >=0):
            Featdictionary1 = beta(N0, Q, 1, i)
            i = i - 1
            Q =Featdictionary1
        return Q

Q=mu[2]
i=l-1
Featdictionary = beta(N,Q,l,i)

file = 'testData/pimatest.csv'

with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    names =reader.fieldnames
    print(reader.fieldnames)
    names = names[:-1]

    features = np.array([])
    for i in names:
        features = np.append(features, i)

if(Featdictionary.size == features.size):
    print("True")
    WPdictionary = dict(zip(features, Featdictionary))
    print(WPdictionary)
    for key, value in WPdictionary.items():
        print(key)

else:
    feature_counter = 1
    positive = 0
    negative = 0
    feature_count = np.array([])
    for i in Featdictionary:
        print("feature_counter",feature_counter, '>=', "{0:.2f}".format(i) )
        feature_count = np.append(feature_count,feature_counter)
        feature_counter = feature_counter + 1

    WPdictionary = dict(zip(feature_count, Featdictionary))
    print(WPdictionary)

WPdict = WPdictionary
g_viol = featureImpGlobal.global_viol

print(g_viol)

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
    #features = fields
    print(features)
    output = fields[-1]
    print(output)

df = pd.read_csv(file, sep=',')
df = df.iloc[:,:-1]
print(df.values)
print("Size:")
print(df.shape[0])

cols = 'top3Featdictionary'
lst = []
dfFeatdictionary = pd.DataFrame(lst, columns=[cols])

for i in range(df.shape[0]):
    df2 = df.iloc[i]
    print(df.iloc[i])

    # for i in range(len(features)):
    #     print(df.iloc[0][i])

    Coln =[]
    Colv = []
    for (colname,colval) in df2.iteritems():
         print(colname, colval)
         Coln.append(colname)
         Colv.append(colval)
    print(Coln)
    print(Colv)
    dfdict = dict(zip(Coln, Colv))
    print(dfdict)
    viol = 0
    sat = 0
    violList = []
    scoreList = []
    for dfdict_values, WPdict_values in zip(dfdict.items(), WPdict.items()):
        #print(WPdict_values[0])
        #print(dfdict_values[0])
        if dfdict_values >= WPdict_values:
            sat = 1
            viol = 0
            print('Satisfied')
            violList.append(viol)
            #violDict.append(dict(zip(Coln, Colv)))
        else:
            print('Not Satisfied')
            viol = 1
            violList.append(viol)
    #violDict = dict(zip(WPdict_values[0], viol))
        score = (sat+viol)/len(features)
        print(score)
        scoreList.append(score)

    print(sat)
    print(viol)
    print(violList)
    #violDict = {}
    violDict = dict(zip(features, violList))
    scoreDict = dict(zip(features, scoreList))
    print(scoreDict)
    print(g_viol)
    Cdict = Counter(scoreDict) + Counter(g_viol)
    print(Cdict)


    for key, value in scoreDict.items():
        print(key, '->', value)

    sortedDict = sorted(scoreDict.items(), key=lambda x:x[1])

    print(sortedDict)
    impFeatFeatdictionary = []
    for k in sortedDict:
    #print(key, '->', value)
        impFeatFeatdictionary.append(k[0])

    #How many features n=3?
    impFeatures = impFeatFeatdictionary[:3]
    #print(list2[::-1])
    print(impFeatures)
    top3Featdictionary=[]
    # for index, tuple in enumerate(impFeatures):
    #         top3Featdictionary.append(tuple)
    # print(top3Featdictionary)
    add = {'top3Featdictionary': impFeatures}
    dfFeatdictionary= dfFeatdictionary.append(add, ignore_index=True)

    print(dfFeatdictionary)


#Convert to csv file
elapsed_timePerm = time.time() - time_startpre

print(elapsed_timePerm)

# #Rows-wise recall
# dfFeatdictionary[top3Featdictionary] =dfFeatdictionary[top3Featdictionary].map(lambda x: "Y" if "Glucose" in x else "N")
#
# recall =dfFeatdictionary.apply(lambda x: x.str.contains("Y")).sum()
#
# print(recall)
