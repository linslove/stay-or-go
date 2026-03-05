import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

df = pd.read_csv("HR-Employee-Attrition.csv")

df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], inplace=True)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=82, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=200, class_weight={0: 1, 1: 10}, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.20).astype(int)

accuracy = accuracy_score(y_test, y_pred)
recall   = recall_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)

print("\nModel trained successfully!")
print(f"Accuracy : {accuracy*100:.1f}%")
print(f"Recall   : {recall*100:.1f}%")
print(f"F1 Score : {f1*100:.1f}%")

pickle.dump(model,              open("model.pkl",    "wb"))
pickle.dump(scaler,             open("scaler.pkl",   "wb"))
pickle.dump(label_encoders,     open("encoders.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

metrics = {
    "accuracy": accuracy, "recall": recall, "f1": f1,
    "report": classification_report(y_test, y_pred,
                target_names=["Stayed", "Left"], output_dict=True)
}
pickle.dump(metrics, open("metrics.pkl", "wb"))

print("\nFiles saved: model.pkl, scaler.pkl, encoders.pkl, features.pkl, metrics.pkl")