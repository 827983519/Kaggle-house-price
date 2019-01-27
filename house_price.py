import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor 

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


remove_feature = ['MSSubClass','Alley','GrLivArea','SaleType']

data.drop(data[(data["GrLivArea"]>4000)&(data["SalePrice"]<300000)].index,inplace=True)

output = data.SalePrice

X = data.copy()
X = X.drop(['Id','SalePrice'],axis=1)
#X = X.drop(remove_feature,axis=1)

X["LotAreaCut"] = pd.qcut(X.LotArea,10)
X['LotFrontage'] = X.groupby(["LotAreaCut",'Neighborhood'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
X['LotFrontage'] = X.groupby("LotAreaCut")['LotFrontage'].transform(lambda x:x.fillna(x.median()))

a = X.isna().sum()
print(a[a>0])
n_feature = []
o_feature = []

for i in range(len(X.columns)):
    if X.isna().sum().values[i] != 0:
        if X.dtypes[i] != 'O':
            n_feature.append(X.columns[i])
        else:
            o_feature.append(X.columns[i])
print(n_feature)
print(o_feature)

#X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)

X[o_feature] = X[o_feature].fillna('None')
my_imputer = SimpleImputer( strategy='median')
X[n_feature] = my_imputer.fit_transform(X[n_feature])

Y = test.copy()
Y = Y.drop('Id',axis=1)
#Y['GarageYrBlt'] = Y['GarageYrBlt'].fillna(0)

n_feature = []
o_feature = []
for i in range(len(Y.columns)):
    if Y.isna().sum().values[i] != 0:
        if Y.dtypes[i] != 'O':
            n_feature.append(Y.columns[i])
        else:
            o_feature.append(Y.columns[i])

Y[o_feature] = Y[o_feature].fillna('None')
my_imputer = SimpleImputer(strategy='median')
Y[n_feature] = my_imputer.fit_transform(Y[n_feature])

dict_vec = DictVectorizer(sparse=False)
train_data = dict_vec.fit_transform(X.drop('LotAreaCut',axis=1).to_dict('record'))
test_data = dict_vec.transform(Y.to_dict('record'))

model =  RandomForestClassifier(n_estimators=400)
model.fit(train_data,output)

model2 = GradientBoostingRegressor(n_estimators=400)
model2.fit(train_data,output)

model3 = SVC(gamma='auto')
model3.fit(train_data,output)

a = model.predict(test_data)
b = model2.predict(test_data)
c = model3.predict(test_data)

file = open('output.csv','w')
file_svm = open('svm.csv','w')
file_gra = open('gra.csv','w')

#file.write('Id,SalePrice\n')

#file_svm.write('Id,SalePrice\n')

file_gra.write('Id,SalePrice\n')

for i in range(len(test.Id)):
    ''' 
    file.write(str(test.Id[i]))
    file.write(',')
    file.write(str(a[i]))
    file.write('\n')
    '''
    '''
    file_svm.write(str(test.Id[i]))
    file_svm.write(',')
    file_svm.write(str(c[i]))
    file_svm.write('\n')
    ''' 
    file_gra.write(str(test.Id[i]))
    file_gra.write(',')
    file_gra.write(str(b[i]))
    file_gra.write('\n')

file.close()
file_svm.close()
file_gra.close()



