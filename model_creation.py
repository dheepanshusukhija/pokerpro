from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd, os
import pickle
import json
from sklearn.externals import joblib






feature_names = list()
for index in range(1, 6):
    feature_names.extend(["Suit"+str(index), "Rank"+str(index)])

feature_names.append('class')


training_input_file = os.path.abspath('train.csv')

np.random.seed(666)     

class myConfigs:
    features = 0
    classes = 0


config = myConfigs()




train_data = pd.read_csv(training_input_file, names=feature_names)

config.features = len(train_data.columns) - 1
config.classes = len(set(train_data['class']))


train_data = train_data.sample(frac=1).reset_index(drop=True)


train_y = np.array(train_data['class'])
train_x = np.array(train_data.drop('class', 1))


scaler = StandardScaler()

train_set = np.empty(train_x.shape, dtype = float)


for index in range(len(train_x)):
    train_set[index] = train_x[index].astype(float)



scaler.fit(train_set)
scalefit = "scaler.save"
joblib.dump(scaler, scalefit) 

data_train = scaler.transform(train_set)
l=[2,7,3,13,4,1,4,3,3,7]
n=np.array(l).reshape(1,10)
test_data=pd.DataFrame(n)
test_data.columns=["Suit1",  "Rank1",  "Suit2",  "Rank2",  "Suit3",  "Rank3",  "Suit4",  "Rank4",  "Suit5",  "Rank5"]
test_x = np.array(test_data)
test_set = np.empty(test_x.shape, dtype = float)
for index in range(len(test_x)):
    test_set[index] = test_x[index].astype(float)
data_test = scaler.transform(test_set)
classifier = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (32, 16), activation = 'tanh', learning_rate_init = 0.02, max_iter = 2000)
classifier.fit(data_train, train_y)
pickle.dump(classifier,open('model.pk1','wb'))
model = pickle.load(open('model.pk1','rb'))
c=model.predict(data_test)
print(model.predict(data_test))
