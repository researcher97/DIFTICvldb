import pandas as pd
df = pd.read_csv('housingData/housepricedata.csv')
df

print(df)
features = list(df.columns.values)
features.remove('BedroomAbvGr')
print(features)

dataset = df.values
dataset
X_ = dataset[:,0:10]
Y = dataset[:,10]
#preprocessing

print(df)
features = list(df.columns.values)
print(features)
#features.remove('LSTAT')

#features.remove('DiabetesPedigreeFunction')


import xgboost
import shap
#
#train an XGBoost model
X, y = X_,Y
# model = xgboost.XGBRegressor().fit(X, y)
#


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.2)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#model
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(12, activation='relu', input_shape=(10,)),
    Dense(10, activation='relu'),
    #Dense(10, activation='relu'),
    Dense(1, activation='sigmoid'),
])


model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['acc'])


hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

model.save('housepredictionClassifier_model1.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
#

#shap.plots.bar(shap_values)

print("Test loss:", score[0])
print("Test accuracy:", score[1])


predictions = (model.predict(X) > 0.5).astype(int)
