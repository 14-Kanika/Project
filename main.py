import pandas as pd
import numpy as np
AdClickData=pd.read_csv("/content/Ad click data.csv", encoding='latin')
print('Shape before deleting duplicate values:', AdClickData.shape)

# Removing duplicate rows if any
AdClickData=AdClickData.drop_duplicates()
print('Shape After deleting duplicate values:', AdClickData.shape)

# Printing sample data
# Start observing the Quantitative/Categorical/Qualitative variables
AdClickData.head(10)
%matplotlib inline
# Creating Bar chart as the Target variable is Categorical
GroupedData=AdClickData.groupby('Clicked').size()
GroupedData.plot(kind='bar', figsize=(4,3))

AdClickData.info()

AdClickData.describe(include='all')
AdClickData.nunique()

UselessColumns = ["VistID", "Country_Name" , "Year"]
AdClickData = AdClickData.drop(UselessColumns,axis=1)
AdClickData.head()

def PlotBarCharts(inpData, colsToPlot):
    %matplotlib inline

    import matplotlib.pyplot as plt

    # Generating multiple subplots
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(40,6))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])
PlotBarCharts(inpData=AdClickData, colsToPlot=["Ad_Topic","City_code", "Male",
                                               "Time_Period", "Weekday","Month"])

AdClickData.hist(["Time_Spent", "Age", "Avg_Income", "Internet_Usage"], figsize=(18,10))

AdClickData.isnull().sum()
ContinuousColsList=["Time_Spent", "Age", "Avg_Income", "Internet_Usage"]

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(ContinuousColsList), figsize=(18,5))

# Creating box plots for each continuous predictor against the Target Variable "Clicked"
for PredictorCol , i in zip(ContinuousColsList, range(len(ContinuousColsList))):
    AdClickData.boxplot(column=PredictorCol, by='Clicked', figsize=(5,5), vert=True, ax=PlotCanvas[i])


def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]

    print('##### ANOVA Results ##### \n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])

    return(SelectedPredictors)

ContinuousVariables=["Time_Spent", "Age", "Avg_Income", "Internet_Usage"]
FunctionAnova(inpData=AdClickData, TargetVariable='Clicked', ContinuousPredictorList=ContinuousVariables)

CrossTabResult=pd.crosstab(index=AdClickData['Male'], columns=AdClickData['Clicked'])
CrossTabResult

def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]

    for predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)

        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])

    return(SelectedPredictors)


CategoricalVariables=["Ad_Topic","City_code", "Male",
                     "Time_Period", "Weekday","Month"]

# Calling the function
FunctionChisq(inpData=AdClickData,
              TargetVariable='Clicked',
              CategoricalVariablesList= CategoricalVariables)

SelectedColumns=["Time_Spent", "Age", "Avg_Income", "Internet_Usage",
                "Ad_Topic", "City_code", "Male", "Time_Period"]

# Selecting final columns
DataForML=AdClickData[SelectedColumns]
DataForML.head()

DataForML.to_pickle('DataForML.pkl')

DataForML['Male'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML.head()

DataForML_Numeric=pd.get_dummies(DataForML)
DataForML_Numeric['Clicked']=AdClickData['Clicked']
DataForML_Numeric.head()

DataForML_Numeric.columns

TargetVariable='Clicked'
Predictors=['Time_Spent', 'Age', 'Avg_Income', 'Internet_Usage', 'Male',
       'Ad_Topic_product_1', 'Ad_Topic_product_10', 'Ad_Topic_product_11',
       'Ad_Topic_product_12', 'Ad_Topic_product_13', 'Ad_Topic_product_14',
       'Ad_Topic_product_15', 'Ad_Topic_product_16', 'Ad_Topic_product_17',
       'Ad_Topic_product_18', 'Ad_Topic_product_19', 'Ad_Topic_product_2',
       'Ad_Topic_product_20', 'Ad_Topic_product_21', 'Ad_Topic_product_22',
       'Ad_Topic_product_23', 'Ad_Topic_product_24', 'Ad_Topic_product_25',
       'Ad_Topic_product_26', 'Ad_Topic_product_27', 'Ad_Topic_product_28',
       'Ad_Topic_product_29', 'Ad_Topic_product_3', 'Ad_Topic_product_30',
       'Ad_Topic_product_4', 'Ad_Topic_product_5', 'Ad_Topic_product_6',
       'Ad_Topic_product_7', 'Ad_Topic_product_8', 'Ad_Topic_product_9',
       'City_code_City_1', 'City_code_City_2', 'City_code_City_3',
       'City_code_City_4', 'City_code_City_5', 'City_code_City_6',
       'City_code_City_7', 'City_code_City_8', 'City_code_City_9',
       'Time_Period_Early-Morning', 'Time_Period_Evening',
       'Time_Period_Mid-Night', 'Time_Period_Morning', 'Time_Period_Night',
       'Time_Period_Noon']
X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
PredictorScaler=MinMaxScaler()
PredictorScalerFit=PredictorScaler.fit(X)
X=PredictorScalerFit.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1,penalty='l2', solver='newton-cg')
LOG=clf.fit(X_train,y_train)
prediction=LOG.predict(X_test)


from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))


from sklearn.model_selection import cross_val_score
Accuracy_Values=cross_val_score(LOG, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))
