import numpy as np
import pandas as pd 
import shap

from SHAPPerm import shap_values


def normalize(df):
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result

from pandas.api.types import is_string_dtype

data = pd.read_csv('german_credit_data.csv',index_col=0,sep=',')
labels = data.columns

print(labels)


for col in labels:
    if is_string_dtype(data[col]):
        if col == 'Risk':

            data[col] = pd.factorize(data[col])[0]
            continue

        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)

        data.drop(col, axis=1, inplace=True)
    else:
        data[col] = normalize(data[col])



data_train = data.iloc[:800]
data_valid = data.iloc[800:]



x_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]
x_val = data_valid.iloc[:,:-1]
y_val = data_valid.iloc[:,-1]


from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import Dense, Dropout

sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)

model = Sequential()
model.add(Dense(units=50, activation='tanh', input_dim=23, kernel_initializer='glorot_normal', bias_initializer='zeros'))#, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=22, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values), epochs=30, batch_size=128)

model.save('germanCredit_mod.h5')
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print('Test Loss: %.3f' % loss)
print('Test Accuracy: %.3f' % acc)

x_val.to_csv('german_df_test.csv', index=False)


y_pred = model.predict_classes(x_val.values)

y = pd.DataFrame(y_val.values)
