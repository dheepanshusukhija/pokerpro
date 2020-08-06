from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd, os
import pickle
feature_names = list()
for index in range(1, 6):
    feature_names.extend(["Suit"+str(index), "Rank"+str(index)])

feature_names.append('class')

scaler = StandardScaler()
testing_input_file=os.path.abspath('test1.csv')
test_data = pd.read_csv(testing_input_file, names=feature_names)
test_y = np.array(test_data['class'])
test_x = np.array(test_data.drop('class', 1))
scaler.fit(test_set)
test_set = np.empty(test_x.shape, dtype = float)
for index in range(len(test_x)):
    test_set[index] = test_x[index].astype(float)
data_test = scaler.transform(test_set)   
model = pickle.load(open('model.pk1','rb'))
c=model.predict(data_test)
print(c[0])
