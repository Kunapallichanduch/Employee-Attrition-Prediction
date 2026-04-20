import pandas as pd

df = pd.read_csv("employee_attrition_dataset.csv")

predictions = []

for _, row in df.iterrows():

    # Hidden neurons
    h1 = 5 if row["Job_Satisfaction"] <= 2 else 0
    h2 = 5 if row["Overtime"] == "Yes" and row["Years_At_Company"] < 3 else 0
    h3 = 4 if row["Work_Life_Balance"] <= 2 else 0
    h4 = 2 if row["Promotion_Last_5_Years"] == "No" else 0
    h5 = 1 if row["Distance_From_Home"] > 25 else 0

    output_score = h1 + h2 + h3 + h4 + h5

    if output_score >= 5:
        predictions.append("Yes")
    else:
        predictions.append("No")

df["Predicted_Attrition"] = predictions

accuracy = (df["Attrition"] == df["Predicted_Attrition"]).mean() * 100

print("Neural Network Accuracy:", round(accuracy,2), "%")
print(df[["Employee_Name","Attrition","Predicted_Attrition"]].head(10))
