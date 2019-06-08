import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
from collections import Counter
import random


from fuzzylogic import *

def choose_cols(data,cols):
	return [[v for  k,v in row.items() if k in cols] for row in  data]

data_fid = 'data/csv/all_data.csv'
data = load_data( data_fid )
random.shuffle(data)

params = {
		'cond':get_condition_params(),
		'crit':get_criticality_params()
		}

p = params['cond']  # working just with condition model

p['var_map'] = json.load(open(p['var_map'], 'r'))
if 'val_map' in p: p['val_map'] = json.load(open(p['val_map'], 'r'))

X = np.array(choose_cols(data,p['var_map'].values()))
X = X.astype(np.float)
y = np.array(choose_cols(data,['PCSDesc']))
y = y.flatten()
y = np.array([p['val_map'][e] for e in y])

m = int(len(X)*0.75)

#classifier = MultinomialNB() # 1.9205653
#classifier = linear_model.BayesianRidge() # 1.34634878
classifier = svm.SVC() #1.1418
#classifier = svm.SVR() #1.1409
#classifier = linear_model.SGDClassifier() #0.98 - 2.2 (big range)
#classifier = GaussianNB() # 0.93
#classifier.fit(X[:m],y[:m]) 
classifier.fit(X,y) 

#pred = classifier.predict(X[m:])
pred = classifier.predict(X)

#dif = np.absolute(pred-y[m:])
dif = np.absolute(pred-y)
loss = np.mean(dif)
print(loss)
print(Counter(pred),Counter(y[m:]))

# SAVE DATA TO OUT FILE
cols = ['PCSDesc']
out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
for i,row in enumerate(out_data):
	row['PCSDesc'] = y[i]
	row['PCSDesc_prediction'] = pred[i]

save_data('out/model_out_data.csv',out_data)

