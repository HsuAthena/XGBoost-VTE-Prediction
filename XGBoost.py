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
def xgbresult(table,label):
    print('xgboost')
    # Set hyperparameters for the XGBoost model
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
    plst = params.items()

    # Split data into training and testing sets
    Xc_train,  Xc_test, yc_train, yc_test = train_test_split(table, label, stratify=label,test_size=0.2,random_state=100, shuffle=True)

    # Convert data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(data=Xc_train,label=yc_train)
   
    # Train the XGBoost model
    num_rounds = 500
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round= num_rounds)
    
    # Make predictions on the test set
    dtest = xgb.DMatrix(Xc_test)
    ans = model.predict(dtest)

    # calculate the accuracy
    # Counters for true positives and false positives
    true_positives = 0
    false_positives = 0
    for i in range(len(yc_test.values)):
        if (ans[i]>0.5)*1 == yc_test.values[i]:
            true_positives += 1
        else:
            false_positives += 1

    print("Accuracy: %.2f %% " % (100 * true_positives / (true_positives + false_positives)))

    # Plot feature importance and ROC curve
    plot_importance(model)
    plt.show()
    fpr, tpr, _ = roc_curve(yc_test.values.astype(int), ans)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    return model

# Get the XGBoost results
table=all_concat_data
label=(demooutcome['VTE'] == 'Y')
model=xgbresult(table,label)

# Extract and print feature importance
demo_vs_soclabreport=pd.DataFrame(model.get_fscore().keys(),columns=['variable'])
demo_vs_soclabreport['variable_importance']=model.get_fscore().values()
demo_vs_soclabreport=demo_vs_soclabreport.sort_values(by='variable_importance',ascending=False)
print(demo_vs_soclabreport)