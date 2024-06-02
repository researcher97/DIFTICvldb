import numpy as np
import pandas as pd


import pandas as pd
import pandas as panda
import shap

from SHAPPerm import shap_values

data = pd.read_csv("bank_customer_survey.csv")
data.shape

data.head()

f = list(data.columns.values)

data.isnull().sum().sum()

data['y'].value_counts(normalize=True)

df_cf0 = data[data['y'] == 0]
df_cf1 = data[data['y'] == 1]
print(df_cf0.shape, df_cf1.shape)

#take sample of df0 as per 1
df_cf0 = df_cf0.sample(df_cf1.shape[0], random_state=10)
print(df_cf0.shape, df_cf1.shape)

df_new = pd.concat([df_cf0, df_cf1])
df_new.shape

df_new.head()

#Encoding month with month numbers
#df_new.month.unique()
month_code = {'may':5, 'mar':3, 'jun':6, 'feb':2, 'jul':7, 'aug':8,
              'apr':4, 'jan':1, 'nov':11,'dec':12, 'sep':9, 'oct':10}

df_new['month'] = df_new['month'].map(month_code)

# Encode education
edc_code = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}
df_new['education'] = df_new['education'].map(edc_code)

#Encode P_outcome
p_out_code = {'unknown':0, 'failure':1, 'other':2, 'success':3}
df_new['poutcome'] = df_new['poutcome'].map(p_out_code)

df_new = pd.get_dummies(df_new, drop_first=True)


#look at data
print(df_new.head())

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X = df_new.drop('y', axis=1)
Y = df_new['y']

print(X.columns.values)


print(X)


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_traindf = x_train[:].values
x_testdf = x_test[:].values
y_traindf = y_train[:].values
y_testdf = y_test[:].values

#num of input features
n_features = x_train.shape[1]

#print(len(n_features))

#Define Model
model = Sequential()
model.add(Dense(10, activation="relu", input_shape=(n_features,)))
model.add(Dense(28, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

#compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#train
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, shuffle=False)
model.save('bankcustomer_mod.h5')
# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss: %.3f' % loss)
print('Test Accuracy: %.3f' % acc)

x_test.to_csv('bankcustomer_test_df.csv', index=False)
y_test.to_csv('bankcustomer_Ytest_df.csv', index=False)


print('Test Loss: %.3f' % loss)
print('Test Accuracy: %.3f' % acc)
