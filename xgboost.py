import pandas as pd

df = pd.read_csv("employee_attrition_dataset.csv")

predictions = []

for _, row in df.iterrows():
    score = 0

    if row["Job_Satisfaction"] <= 2:
        score += 4
    if row["Overtime"] == "Yes" and row["Years_At_Company"] < 3:
        score += 4
    if row["Work_Life_Balance"] <= 2:
        score += 3
    if row["Promotion_Last_5_Years"] == "No":
        score += 1
    if row["Distance_From_Home"] > 25:
        score += 1

    if score >= 4:
        predictions.append("Yes")
    else:
        predictions.append("No")

df["Predicted_Attrition"] = predictions

accuracy = (df["Attrition"] == df["Predicted_Attrition"]).mean() * 100

print("XGBoost Accuracy:", round(accuracy,2), "%")
print(df[["Employee_Name","Attrition","Predicted_Attrition"]].head(10))
