import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Step 1: Load data
data = pd.read_csv("loanapprovalprediction.csv")
data.columns = data.columns.str.strip()
print(data.shape)
print(data.head(10))

data = data.drop(['loan_id'], axis=1)
print(data.head(10))

label_encoder = preprocessing.LabelEncoder()
data['education'] = label_encoder.fit_transform(data['education'])
data['self_employed'] = label_encoder.fit_transform(data['self_employed'])
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])

print(data.head(10))

X = data.drop(['loan_status'], axis=1)
feature_names = X.columns
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

rf= RandomForestClassifier (n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

print(feat_importance_df)

plt.figure(figsize=(8,6))
plt.barh(feat_importance_df['Feature'], feat_importance_df['Importance'], color='teal')
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
rf.pred=rf.predict(X_test)  
print("Accuracy:", 100*metrics.accuracy_score(y_test, rf.pred)) 

lr=LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred=lr.predict(X_test)
print("Accuracy:", 100*metrics.accuracy_score(y_test, lr_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test, rf.pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred=xgb.predict(X_test)
print("Accuracy:", 100*metrics.accuracy_score(y_test, xgb_pred))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Cross-Validation R² Scores:", cv_scores)
print("Mean R² Score:", cv_scores.mean())