# needs ucimlrepo

from ucimlrepo import fetch_ucirepo

# fetch dataset
census_income = fetch_ucirepo(id=20)

# data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)

X.head()

X.dtypes

def get_options(data_frame, column_name):
  seen = []
  for val in data_frame[column_name]:
    if val not in seen:
      seen.append(val)
  return seen

def str_to_num(data_frame, column_name):
  key = {}
  count = 0
  for tipe in get_options(data_frame, column_name):
    data_frame[column_name].replace(tipe, count, inplace=True)
    key[tipe] = count
    count +=1

  print(key)
  return data_frame

for col_name in X.columns:
  if X[col_name].dtype == 'object':
    str_to_num(X, col_name)

y.head()

get_options(y, 'income')

y['income'].replace('<=50K.', '<=50K', inplace=True)
y['income'].replace('>50K.', '>50K', inplace=True)

str_to_num(y, 'income')

# Predicting target variable using a RandomForestClassifier from scikit-learn, with all the other numerical features in the dataset as features.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_train, data_test, labels_train, labels_test = train_test_split(X, y)
clf = RandomForestClassifier()
clf.fit(data_train, labels_train)
clf.score(data_test, labels_test)

clf2 = RandomForestClassifier(criterion='entropy')
clf2.fit(data_train, labels_train)
clf2.score(data_test, labels_test)

clf3 = RandomForestClassifier(criterion='log_loss')
clf3.fit(data_train, labels_train)
clf3.score(data_test, labels_test)
# find relative importance of features
features = clf.feature_importances_

print('Feature Importances for Forest\n')

for idx in range(len(features)):
  print(f'  {X.columns[idx]}: {features[idx]}\n')

#additional analysis
import matplotlib.pyplot as plt
import seaborn as sns

C = X
C['income'] = y

corr = C.corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=True)
plt.title('Correlation')
plt.show()

for col in X.columns:
    X[col] = X[col] / (max(X[col]))

X.head()

from sklearn.neighbors import KNeighborsClassifier

nbrs = KNeighborsClassifier(n_neighbors=29).fit(data_train, labels_train)
nbrs.score(data_test, labels_test)
"""
Conclusions:
I was suprised that the decision tree classifier worked better than K nearest neighbors on this dataset, 
but because I imagined there would be lots of people in similar situations within the same income 'bracket'.
Instead the only 'heavily' correlated feature to ones income was marital status, and even this wasn't 
correlated that much. This is also cool because we just talked about this in my 109 lecture and to see it 
confirmed with data off the internet is validating. Finally, I think a lot of people make very big 
generalizations and assumptions about others based on their race, gender, age, job, ect... But (at least 
going off this dataset), these do not always decide someones fate (to make over or under 50k per year). 
Computers can make those assumptions sometimes with cool math and stuff and yeild 85% accuracy, but I sure 
couldn't from this data set. #dont judge a book by its cover!"""
