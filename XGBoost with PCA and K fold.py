import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from IPython.core.display import display, HTML
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


#Read the dataset, which is csv file
all_data=pd.read_csv('./hhsu/Previous versions/Grifols_6.28.20.csv')

# Pre-process the '% Missed Ppx Doses' column by replacing '%' and 'NA' with ''
# and then converting it to float data type
all_data['% Missed Ppx Doses']=all_data['% Missed Ppx Doses'].str.replace('%','')
all_data['% Missed Ppx Doses']=all_data['% Missed Ppx Doses'].str.replace('NA','')
all_data['% Missed Ppx Doses']=all_data['% Missed Ppx Doses'].astype('float')  

# Separate out different categories of columns from the dataset
democa=all_data[['MOI - Blunt', 'MOI - Penetrating', 'MOI - Burn', 'Transfer?', 'Gender', 'Race', 'Ethnicity',
            'Pre-injury Anticoags', 'Initial Dispo - ICU', 'Initial Dispo - IMU', 'Initial Dispo - Floor',
            "PH T'fusion", "In-hosp T'fusion (24hrs)", "Any T'fusion (PH-24hrs)",'PH TXA', 'In-hosp TXA',
            'Any TXA', 'Death', 'REBOA?']]# Demographic categorical data
demooutcome=all_data[['VTE']]# Outcome labels
demonu=all_data[['Age', 'Height (cm)', 'Weight (kg)', 'BMI', 'Time to 1st Ppx (hrs)', 
                 '# Potential Ppx Doses', '# Ppx Doses Given', '# Missed Ppx Doses', '% Missed Ppx Doses', 
                 'Calc LOS', 'ICU LOS', '# Vent Days', 'AIS-Head', 'AIS-Face', 'AIS-Chest', 
                 'AIS-Abdomen', 'AIS-Extremity', 'AIS-External', 'ISS']]
vsca=all_data[['PH FAST', 'Arrival FAST']]# Vital signs categorical data
vsnu=all_data[['PH SBP', 'PH DBP', 'PH MAP (est.)', 'PH HR', 'PH SI', 'PH PP', 'PH GCS-T', 'PH GCS-E', 
               'PH GCS-V', 'PH GCS-M', 'Arrival SBP', 'Arrival DBP', 'Arrival MAP', 'Arrival HR', 
               'Arrival SI', 'Arrival PP', 'Arrival GCS-T', 'Arrival GCS-E', 'Arrival GCS-V', 
               'Arrival GCS-M', 'Change SBP', 'Change DBP', 'Change MAP (est.)', 'Change HR', 'Change SI', 
               'Change PP', 'Change GCS-T', 'Change GCS-E', 'Change GCS-M', 'Change GCS-V']]# Vital signs numerical data
soc_lab=all_data[['Arrival ACT ', 'Arrival Split Point', 'Arrival  R-time', 'Arrival K-time', 
                  'Arrival Alpha Angle', 'Arrival Max Amp', 'Arrival G-value', 'Arrival Ly30', 
                  'Arrival Creatinine', 'Arrival Platelet', 'Arrival Base Value', 'D1 Highest Creatinine', 
                  'D1 First Platelet', 'D2 Highest Creatinine', 'D2 First Platelet', 'D3 Highest Creatinine', 
                  'D3 First Platelet', ' D4 Highest Creatinine', 'D4 First Platelet', 'D5 Highest Creatinine', 
                  'D5 First Platelet', 'D6 Highest Creatinine', 'D6 First Platelet', 'D7 Highest Creatinine', 
                  'D7 First Platelet', 'D8 Highest Creatinine', 'D8 First Platelet', 'D9 Highest Creatinine', 
                  'D9 First Platelet', 'D10 Highest Creatinine', 'D10 First Platelet', 'D0-10 Highest Creatinine', 
                  'D0-10 Lowest Creatinine', 'Diff. H., L. Creat.', 'D0-D10 Highest Plt', 'D0-D10 Lowest Plt', 
                  'Diff. H.,  L. Plt.', "All anti-Xa's.1", "All anti-Xa's.2", "All anti-Xa's.3", 
                  "All anti-Xa's.4", "All anti-Xa's.5", "All anti-Xa's.6", 'Highest anti-Xa', 'Lowest anti-Xa', 
                  'Diff. H., L. anti-Xa', 'All antithrombins.1', 'All antithrombins.2', 'All fibrinogens .1', 
                  'All fibrinogens .2']]# Lab results data

# Convert categorical data to dummy/indicator variables
democa=pd.get_dummies(democa)
vsca=pd.get_dummies(vsca)

# Concatenate all the pre-processed dataframes along columns to form the final dataset
all_concat_data=pd.concat([democa,demonu,vsca,vsnu,soc_lab],axis=1)

# Define a function to run xgboost model and return the results
def xgbresult_kfold(X_train,y_train,X_test,y_test):
    print('xgboost')
    params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'verbosity':0,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4}
    dtrain = xgb.DMatrix(data=X_train,label=y_train)
    num_rounds = 500
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round= num_rounds)
    
    # Predicting on the test set
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)

    # Calculate accuracy
    true_positives = 0
    false_positives = 0
    for i in range(len(y_test.values)):
        if (ans[i]>0.5)*1 == y_test.values[i]:
            true_positives += 1
        else:
            false_positives += 1

    print("Accuracy: %.2f %% " % (100 * true_positives / (true_positives + false_positives)))



col=['MOI - Blunt_N','MOI - Blunt_Y','MOI - Penetrating_N','MOI - Penetrating_Y','MOI - Burn_N','MOI - Burn_Y','Transfer?_No',
 'Transfer?_Unknown','Transfer?_Yes','Gender_F','Gender_M','Race_Asian','Race_Black','Race_White',
 'Ethnicity_Hispanic','Ethnicity_Not Hispanic','Pre-injury Anticoags_No/unknown','Pre-injury Anticoags_Yes','Initial Dispo - ICU_N',
 'Initial Dispo - ICU_Y','Initial Dispo - IMU_N','Initial Dispo - IMU_Y','Initial Dispo - Floor_N','Initial Dispo - Floor_Y',
 "PH T'fusion_No","PH T'fusion_Yes","In-hosp T'fusion (24hrs)_No","In-hosp T'fusion (24hrs)_Yes","Any T'fusion (PH-24hrs)_No",
 "Any T'fusion (PH-24hrs)_Yes",'PH TXA_N','PH TXA_Y','In-hosp TXA_N','In-hosp TXA_Y','Any TXA_N','Any TXA_Y','Death_No',
 'Death_Yes','REBOA?_No','REBOA?_Yes','Age','Height (cm)','Weight (kg)','BMI','Time to 1st Ppx (hrs)','# Potential Ppx Doses',
 '# Ppx Doses Given','# Missed Ppx Doses','% Missed Ppx Doses','Calc LOS','ICU LOS','# Vent Days',
 'AIS-Head','AIS-Face','AIS-Chest','AIS-Abdomen','AIS-Extremity','AIS-External','ISS','PH FAST_INDETERMINATE','PH FAST_NEGATIVE',
 'PH FAST_Negative','PH FAST_Not done','PH FAST_POSITIVE','PH FAST_Positive','Arrival FAST_INDETERMINATE','Arrival FAST_Indeterminate',
 'Arrival FAST_NEGATIVE','Arrival FAST_Negative','Arrival FAST_Not done','Arrival FAST_POSITIVE','Arrival FAST_Positive','PH SBP','PH DBP',
 'PH MAP (est.)','PH HR','PH SI','PH PP','PH GCS-T','PH GCS-E','PH GCS-V','PH GCS-M','Arrival SBP','Arrival DBP','Arrival MAP',
 'Arrival HR','Arrival SI','Arrival PP','Arrival GCS-T','Arrival GCS-E','Arrival GCS-V','Arrival GCS-M','Change SBP','Change DBP',
 'Change MAP (est.)','Change HR','Change SI','Change PP','Change GCS-T','Change GCS-E','Change GCS-M','Change GCS-V',
 'Arrival ACT ','Arrival Split Point','Arrival  R-time','Arrival K-time','Arrival Alpha Angle',
 'Arrival Max Amp','Arrival G-value','Arrival Ly30','Arrival Creatinine','Arrival Platelet','Arrival Base Value',
 'D1 Highest Creatinine','D1 First Platelet','D2 Highest Creatinine','D2 First Platelet','D3 Highest Creatinine','D3 First Platelet',
 ' D4 Highest Creatinine','D4 First Platelet','D5 Highest Creatinine','D5 First Platelet','D6 Highest Creatinine','D6 First Platelet',
 'D7 Highest Creatinine','D7 First Platelet','D8 Highest Creatinine','D8 First Platelet','D9 Highest Creatinine',
 'D9 First Platelet','D10 Highest Creatinine','D10 First Platelet','D0-10 Highest Creatinine','D0-10 Lowest Creatinine',
 'Diff. H., L. Creat.','D0-D10 Highest Plt','D0-D10 Lowest Plt','Diff. H.,  L. Plt.',
 "All anti-Xa's.1","All anti-Xa's.2","All anti-Xa's.3","All anti-Xa's.4","All anti-Xa's.5","All anti-Xa's.6",'Highest anti-Xa','Lowest anti-Xa',
 'Diff. H., L. anti-Xa','All antithrombins.1','All antithrombins.2','All fibrinogens .1','All fibrinogens .2']

# K fold split dataset to 10 small datset
kf = KFold(n_splits=10, random_state=None, shuffle=False)
x=all_concat_data
y=(demooutcome['VTE'] == 'Y')
pca = PCA(n_components=20)
for train_index, test_index in kf.split(x):
    # Split dataset
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #Transfer nan to mean in feature for specific column
    X_train[col]=X_train[col].fillna(X_train[col].mean()[0])
    X_test[col]=X_test[col].fillna(X_test[col].mean()[0])
    
     # Applying PCA
    Xpca_train = pca.fit_transform(X_train)
    Xpca_test = pca.transform(X_test)
    xgbresult_kfold(Xpca_train,y_train,Xpca_test,y_test)