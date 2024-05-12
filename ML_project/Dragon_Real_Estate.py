from sysconfig import get_path
import pandas as pd 

housing = pd.read_csv("data.csv")
housing.head()

housing.info()

housing['CHAS'].value_counts()

housing.describe()
#get_path().run_line_magic('matplotlib', 'inline')

#for ploting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize =(20,15))

# ##Train-Test Spliting 

#for leaenig purpose
import numpy as np
np.random.seed(42)
def split_train_test(data,test_ratio):
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing,0.2)

print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size =0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]


strat_test_set.info()


strat_test_set['CHAS'].value_counts()

#95/7

housing = strat_train_set.copy()


# ## Looking for correlation

corr_matrix= housing.corr()
corr_matrix['MEDV'].sort_values(ascending= False)

from pandas.plotting import scatter_matrix
attributes =[ "MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize =(12,8))


housing.plot(kind="scatter",x="RM", y="MEDV", alpha=0.8)
# ## Atrribute combination

housing['TAXRM']= housing['TAX']/housing['RM']

housing.head()




corr_matrix= housing.corr()
corr_matrix['MEDV'].sort_values(ascending= False)


housing.plot(kind="scatter",x="TAXRM", y="MEDV", alpha=0.8)




housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing attribute



a= housing.dropna(subset=['RM']) #option1
a.shape




housing.drop("RM",axis=1).shape #option2




median = housing["RM"].median()




housing['RM'].fillna(median)




housing.shape



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)




imputer.statistics_




x= imputer.transform(housing)



housing_tr = pd.DataFrame(x, columns = housing.columns)



housing_tr.describe()


# ## Scikit-learn Design

# ## Creating a Pipeline



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([ ('imputer',  SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler())
                       ])


housing_num_tr = my_pipeline.fit_transform(housing)




housing_num_tr.shape


# ## Selecting a desired model for dragon real estates


from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)



some_data= housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)



model.predict(prepared_data)




prepared_data[0]


# ## Evalution the model


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)



rmse


# ## Using netter evaluation technique - cross validation 

# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv= 10)
rmse_scores = np.sqrt(-scores)



rmse_scores




def print_scores(scores):
    print("scores:",scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    


print_scores(rmse_scores)


# ## convert this note book into python file and run pipeline using vs
# ## Saving the model



from joblib import dump, load
dump(model,'project.joblib')


# ## Testing the model on data 



x_test = strat_test_set.drop('MEDV',axis =1)
y_test = strat_test_set['MEDV'].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictios = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictios)
final_rmse = np.sqrt(final_mse)
print(final_predictios, list(y_test))




print(final_rmse)


# ## Model Usage

from joblib import dump, load
import numpy as np 
model= load('project.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

print(model.predict(features))






