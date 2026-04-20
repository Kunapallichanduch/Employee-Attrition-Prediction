import pandas as pd

df = pd.read_csv("employee_attrition_dataset.csv")

# Simple rule-based tree logic
predictions = []

for _, row in df.iterrows():
    if row["Job_Satisfaction"] <= 2:
        predictions.append("Yes")
    elif row["Overtime"] == "Yes" and row["Years_At_Company"] < 3:
        predictions.append("Yes")
    else:
        predictions.append("No")

df["Predicted_Attrition"] = predictions

accuracy = (df["Attrition"] == df["Predicted_Attrition"]).mean() * 100

print("Decision Tree Accuracy:", round(accuracy,2), "%")
print(df[["Employee_Name","Attrition","Predicted_Attrition"]].head(10))
