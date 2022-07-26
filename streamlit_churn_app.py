import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Churn Prediction')
st.write('Churn Prediction is used by businesses to predict customer churn in advanced so that marketing or customer service actions can be taken to prevent the customer from churning. Use our Churn Prediction app to see where customers would churn based on your data.')
st.header('Upload Data')
fu = st.file_uploader('Upload a CSV')
if fu is not None:
    df = pd.read_csv(fu)
    st.write('Display the dataframe.')
    st.dataframe(df)

# Check the Null Values
st.subheader('Exploratory Data Analysis')

st.write('Check the number of null values in the data.')
st.write(df.isna().sum())

# Display a frequency distribution for churn
# plt.figure(figsize=(5,5))
# ax = sns.countplot(x=df.iloc[:,-1], palette="Blues", linewidth=1)
# plt.savefig('freq_dist.png')
# plt.show()
# st.pyplot(ax)

# Select columns with less than 10 unique values.
#lessthan10 = {df.iloc[:, df.apply(lambda x: x.nunique()) <= 10]}
lessthan10 = df.loc[:, df.nunique() <= 10]

#Create a function to generate boxplots
#plots = {1:[111], 2:[121,122], 3: [131, 132, 133], 4: [221, 222,223, 224], 5: [231, 232, 233, 234, 235], 6: [231, 232, 233, 234, 235, 236], 7:[ }

plots = {}
y = 0
coords = [0, 1]

for x in range(1, lessthan10.shape[1]+1):
    ndx = ((y // 3) + 1) 
    h = ndx * 100
    t = (coords[0] + coords[1]) * 10
    
    plots[x] = [h+t+i for i in range(1,x+1)]
    if coords[1] == 3:
        coords[0] +=1
        coords[1] = 1
    else:
        coords[1] +=1
    y+=1
    

# Create a function to generate countplots:

def countplot(x, y, df):
    
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize=(7*columns, 7*rows))
    
    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df, palette='Blues', alpha=0.8, linewidth=0.4, edgecolor='black')
        ax.set_title(j)
        
    return plt.show() 


# Generate countplots for various features
st.write('Plot Churn vs Non-Churn for Various Features')
cp = countplot(df.iloc[:,-1], lessthan10.columns, df)
plt.tight_layout()

st.pyplot(cp)


# One-Hot-Encoding for identified columns
st.subheader('One Hot Encoding')
st.write('We apply One Hot Encoding to Categorical Features.')
df_d = pd.get_dummies(df, columns=lessthan10.columns)
st.dataframe(df_d)

# Min-Max-Scaling for identified columns.
st.subheader('Min-Max Scaling')
st.write('We apply Min-Max Scaling to the Numerical Data.')

from sklearn.preprocessing import MinMaxScaler
columns = df_d.columns
mms = MinMaxScaler()
df_scaled = mms.fit_transform(df_d)
df_scaled_columns = pd.DataFrame(df_scaled, columns=columns)

st.dataframe(df_scaled_columns)

# Show correlation plot for correlation of Churn with each of the remaining features.
plt.figure(figsize=(16,10))
# st.pyplot(df_scaled_columns.corr()[df_scaled_columns.iloc[:,-1:]].sort_values(axis=1,ascending=False).plot(kind='bar', figsize=(20,5)))


# Train-Test-Split
st.subheader('Train-Val-Test Split')
st.write('We conduct a 80-10-10 Train-Val-Test split.')
# Drop last column
X = df_scaled_columns.iloc[:,:-1]
y = df_scaled_columns.iloc[:,-1]

from sklearn.model_selection import train_test_split

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


#Model Evaluation Metrics
st.subheader('Model Evaluation Metrics')
st.markdown('For performance assessment of the chosen models, various metrics are used: **Feature**: Indicates the top features used by the model to generate the predictions **Confusion matrix**: Shows a grid of true and false predictions compared to the actual values **Accuracy score**: Shows the overall accuracy of the model for training set and test set **ROC Curve**: Shows the diagnostic ability of a model by bringing together true positive rate (TPR) and false positive rate (FPR) for different thresholds of class predictions (e.g. thresholds of 10%, 50% or 90% resulting to a prediction of churn) **AUC (for ROC)**: Measures the overall separability between classes of the model related to the ROC curve **Precision-Recall-Curve**: Shows the diagnostic ability by comparing false positive rate (FPR) and false negative rate (FNR) for different thresholds of class predictions. It is suitable for data sets with high class imbalances (negative values overrepresented) as it focuses on precision and recall, which are not dependent on the number of true negatives and thereby excludes the imbalance **F1 Score**: Builds the harmonic mean of precision and recall and thereby measures the compromise between both. **AUC (for PRC)**: Measures the overall separability between classes of the model related to the Precision-Recall curve')

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, plot_confusion_matrix, precision_score, recall_score

# Define a function that plots the feature weights for a classifier.
def feature_weights(X_df, classifier, classifier_name):
    weights = pd.Series(classifier.coef_[0], index=X_df.columns.values).sort_values(ascending=False)
    
    top_weights_selected = weights[:10]
    plt.figure(figsize=(7,6))
    plt.tick_params(labelsize=10)#plt.xlabel(fontsize=10)
    plt.title(f'{classifier_name} - Top 10 Features')
    top_weights_selected.plot(kind="bar")
    
    bottom_weights_selected = weights[-10:]
    plt.figure(figsize=(7,6))
    plt.tick_params(labelsize=10)#plt.xlabel(fontsize=10)
    plt.title(f'{classifier_name} - Bottom 10 Features')
    bottom_weights_selected.plot(kind="bar")
    
    return print("")

# Define a function that plots the confusion matrix for a classifier and the train and test accuracy
def confusion_matrix_plot(X_train, y_train, X_test, y_test, classifier, y_pred, classifier_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Churn", "Churn"], cmap=plt.cm.Blues, normalize=None, ax=ax)
    ax.set_title(f'{classifier_name} - Confusion Matrix')
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["No Churn", "Churn"], cmap=plt.cm.Blues, normalize='true', ax=ax)
    ax.set_title(f'{classifier_name} - Confusion Matrix (norm.)')
    plt.show()
    
    print(f'Accuracy Score Test: {accuracy_score(y_test, y_pred)}')
    print(f'Accuracy Score Train: {classifier.score(X_train, y_train)} (as comparison)')
    return print("")

# Define a function that plots the ROC curve and the AUC score
def roc_curve_auc_score(X_test, y_test, y_pred_probabilities, classifier_name):
    
    y_pred_prob = y_pred_probabilities[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'{classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.show()
    
    return print(f'AUC Score (ROC): {roc_auc_score(y_test, y_pred_prob)}\n')

# Define a function that plots the precision-recall-curve and the F1 score and AUC score
def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
    
    y_pred_prob = y_pred_probabilities[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    plt.plot(recall, precision, label=f'{classifier_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{classifier_name} - Precision-Recall Curve')
    plt.show()
    
    f1_score_result, auc_score_result = f1_score(y_test, y_pred), auc(recall, precision)
    
    return print(f'F1 Score: {f1_score_result} \nAUC Score (PR): {auc_score_result}\n')


st.header('Model Selection, Training, Prediction and Assessment')
st.write('In the beginning we will test out several models and measure their performance by several metrics. Those models will be optimized in a later step by tuning their hyperparameters. The models used include: K Nearest Neighbors - fast, simple and instance-based Logistic Regression - fast and linear model Random Forest - slower but accurate ensemble model based on decision trees Support Vector Machines - slower but accurate model used here in the non-linear form')

st.subheader('K Nearest Neighbors')

from sklearn.neighbors import KNeighborsClassifier

# Instanciate and train the KNN classifier based on the training set.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions (classes and probabilities) with the trained classifier on the test set.
y_pred_knn = knn.predict(X_val)
y_pred_knn_prob = knn.predict_proba(X_val)

st.write('KNN Prediction')
st.dataframe(y_pred_knn)
st.write('KNN Probability')
st.dataframe(y_pred_knn_prob)

# Plot model evaluations.
st.set_option('deprecation.showPyplotGlobalUse', False)
# knn_cm = confusion_matrix_plot(X_train, y_train, X_val, y_val, knn, y_pred_knn, 'KNN')
st.pyplot(confusion_matrix_plot(X_train, y_train, X_val, y_val, knn, y_pred_knn, 'KNN'))
st.pyplot(roc_curve_auc_score(X_val, y_val, y_pred_knn_prob, 'KNN'))
st.pyplot(precision_recall_curve_and_scores(X_val, y_val, y_pred_knn, y_pred_knn_prob, 'KNN'))

st.subheader('Logistic Regression')

from sklearn.linear_model import LogisticRegression

# Instanciate and train the logistic regression model based on the training set.
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions (classes and probabilities) with the trained model on the test set.
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_prob = logreg.predict_proba(X_test)

st.write('Logistic Regression Prediction')
st.dataframe(y_pred_logreg)
st.write('Logistic Regression Probability')
st.dataframe(y_pred_logreg_prob)

# Plot model evaluations.
feature_weights(X, logreg, 'Log. Regression')
st.pyplot(confusion_matrix_plot(X_train, y_train, X_val, y_val, logreg, y_pred_logreg, 'Log. Regression'))
st.pyplot(roc_curve_auc_score(X_val, y_val, y_pred_logreg_prob, 'Log. Regression'))
st.pyplot(precision_recall_curve_and_scores(X_val, y_val, y_pred_logreg, y_pred_logreg_prob, 'Log. Regression'))

st.subheader('Random Forest')

from sklearn.ensemble import RandomForestClassifier

# Instanciate and train the random forest model based on the training set.
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions (classes and probabilities) with the trained model on the test set.
y_pred_rf = rf.predict(X_test)
y_pred_rf_prob = rf.predict_proba(X_test)

st.write('Random Forest Prediction')
st.dataframe(y_pred_rf)
st.write('Random Forest Probability')
st.dataframe(y_pred_rf_prob)

# Plot model evaluations.
st.pyplot(confusion_matrix_plot(X_train, y_train, X_val, y_val, rf, y_pred_rf, 'Random Forest'))
st.pyplot(roc_curve_auc_score(X_val, y_val, y_pred_rf_prob, 'Random Forest'))
st.pyplot(precision_recall_curve_and_scores(X_val, y_val, y_pred_rf, y_pred_rf_prob, 'Random Forest'))

st.subheader('Support Vector Machine')

from sklearn.svm import SVC

# Instanciate and train the SVM model on the training set.
support_vector_m = SVC(kernel='rbf', probability=True) 
support_vector_m.fit(X_train,y_train)

# Make predictions (classes and probabilities) with the trained model on the test set.
y_pred_svm = support_vector_m.predict(X_val)
y_pred_svm_prob = support_vector_m.predict_proba(X_val)

st.write('SVM Prediction')
st.dataframe(y_pred_svm)
st.write('SVM Probability')
st.dataframe(y_pred_svm_prob)

# Plot model evaluations.
st.pyplot(confusion_matrix_plot(X_train, y_train, X_test, y_test, support_vector_m, y_pred_svm, 'SVM'))
st.pyplot(roc_curve_auc_score(X_val, y_val, y_pred_svm_prob, 'SVM'))
st.pyplot(precision_recall_curve_and_scores(X_val, y_val, y_pred_svm, y_pred_svm_prob, 'SVM'))

st.header('Conclusion')
st.write('Pick the right model based on your chose evaluation metric. Run hyper-parameter optimization. Deploy the model.')
st.write('For a more in-depth Churn Prediction project, feel free to reach out to the Data Analytics Organization at MDI Novare.')