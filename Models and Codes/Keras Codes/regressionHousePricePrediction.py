import numpy as np
import pandas as pd
from tensorflow import keras

# load the dataset

dfN = pd.read_csv('housingData/HousingData.csv')
print(dfN.head())
#
print(dfN.isnull().sum())
#
df_miss=dfN[dfN.isna().any(axis=1)]
#print(df[df.isna().any(axis=1)].to_string())
df_miss.to_csv("housingData/HousingDataMissingValues.csv", sep='\t', encoding='utf-8')

# Fill missing values
df_fillna = dfN.fillna(dfN.mean())

print(df_fillna.isnull().sum())
df_fillna.to_csv("housingData/HousingDataFillMissingValues.csv", sep='\t', encoding='utf-8')
#print(df.head())
#


dataset_x = df_fillna.iloc[:,0:13] #independent columns
dataset_y = df_fillna.iloc[:,-1] #target column i.e price range

print(dataset_x)
features = list(dataset_x.columns.values)
print(features)


#features.remove('DiabetesPedigreeFunction')
print(dataset_x)


from sklearn.model_selection import train_test_split
# split the dataset into training and test datasets.
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.20)



from sklearn.preprocessing import MinMaxScaler
# apply min-max scaling
mms = MinMaxScaler()
mms.fit(training_dataset_x)
training_dataset_x = mms.transform(training_dataset_x)
test_dataset_x = mms.transform(test_dataset_x)

#print(training_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# create keras model and add the layers
model = Sequential(name='BostonHousingPrices')
model.add(Dense(13, input_dim = training_dataset_x.shape[1], activation='relu'))
model.add(Dense(13, activation='relu'))

model.add(Dense(1, activation='linear'))



#
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# fit the model and assign it to 'hist' variable
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)
model.save('regression_house_model1.h5')



# testing the model with using evaluate method
test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]} ---> {test_result[i]}')

"""

0.63796   0.00   8.140  0  0.5380  6.0960  84.50  4.4619   4  307.0  21.00 380.02  10.26 -->  18.20
1.05393   0.00   8.140  0  0.5380  5.9350  29.30  4.4986   4  307.0  21.00 386.85   6.58 --> 23.10

0.01432 100.00   1.320  0  0.4110  6.8160  40.50  8.3248   5  256.0  15.10 392.90   3.95  31.60
 0.15445  25.00   5.130  0  0.4530  6.1450  29.20  7.8148   8  284.0  19.70 390.68   6.86  23.30
 0.10328  25.00   5.130  0  0.4530  5.9270  47.20  6.9320   8  284.0  19.70 396.90   9.22  19.60
"""
#print(test_dataset_x)
""""
TEST
0.06905	0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	396.9	12.7154321 - >36.2
0.08829	12.5	7.87	0.06995884773662552	0.524	6.0120000000000005	66.6	5.5605	5	311	15.2	395.6	12.43 ->	22.9
0.17004	12.5	7.87	0.06995884774	0.524	6.004	85.9	6.5921	5	311	15.2	386.71	17.1	-> 18.9
0.6379600000000001	0	8.14	0.06995884773662552	0.5379999999999999	6.096	84.5	4.4619	4	307	21	380.02	10.26->	18.2
0.06417	0	5.96	0	0.499	5.933	68.2	3.3603	5	279	19.2	396.9	12.715432098765433 ->	18.9
"""

# creating simple data group for the estimation part then predicting the results
predict_data = np.array([ 0.06905,0,2.18,0,0.458,7.147,54.2,6.0622,3,222,18.7,396.9,0])
#predict_data = np.array([ 0.08829,12.5,7.87,0.0,0.524,6.0120000000000005,66.6,5.5605,5,311,15.2,395.6,12.43])
#predict_data = np.array([ 0.17004,12.5,7.87,0,0.524,6.004,85.9,6.5921,5,311,15.2,386.71,17.1])
#predict_data = np.array([0.6379600000000001,0,8.14,0,0.5379999999999999,6.096,84.5,4.4619,4,307,21,380.02,10.26])
#predict_data = np.array([0.06417,0,5.96,0,0.499,5.933,68.2,3.3603,5,279,19.2,396.9,0])
predict_data = mms.transform(predict_data.reshape(1, -1))
predict_result = model.predict(predict_data)
print(f'Predicted result: {predict_result[0, 0]}')
