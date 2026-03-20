import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.style.use("seaborn-v0_8-darkgrid")

# -------------------------------
# LOAD DATA
# -------------------------------
student_df = pd.read_excel("student_marks.xlsx")
retail_df = pd.read_excel("retail_store.xlsx")

print("\n✅ DATA LOADED")

# -------------------------------
# CLEANING
# -------------------------------
student_df.fillna(student_df.mean(numeric_only=True), inplace=True)
student_df.drop_duplicates(inplace=True)

# -------------------------------
# STUDENT ANALYSIS
# -------------------------------
student_df["Total"] = student_df.sum(axis=1)
student_df["Average"] = student_df.mean(axis=1)

top_students = student_df.sort_values(by="Average", ascending=False).head(3)
print("\nTop Students:\n", top_students)

# -------------------------------
# CORRELATION
# -------------------------------
sns.heatmap(student_df.corr(numeric_only=True), annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# REGRESSION
# -------------------------------
X = student_df[["Maths"]]
y = student_df["Science"]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

print("\nR2 Score:", round(r2_score(y, pred), 3))

plt.scatter(X, y)
plt.plot(X, pred, color="red")
plt.title("Maths vs Science")
plt.show()

# -------------------------------
# RETAIL ANALYSIS
# -------------------------------
retail_df["Revenue"] = retail_df["Quantity"] * retail_df["Unit_Price"]

total = retail_df["Revenue"].sum()
print("\nTotal Revenue:", total)

top_products = retail_df.groupby("Product_ID")["Revenue"].sum().sort_values(ascending=False)
top3 = top_products.head(3)

# FIXED seaborn warning
sns.barplot(x=top3.index, y=top3.values, hue=top3.index, palette="coolwarm", legend=False)
plt.title("Top Products")
plt.show()

# -------------------------------
# MONTHLY TREND
# -------------------------------
retail_df["Month"] = pd.to_datetime(retail_df["Date"]).dt.to_period("M")
monthly = retail_df.groupby("Month")["Revenue"].sum()

monthly.plot(marker='o')
plt.title("Monthly Sales")
plt.show()

# -------------------------------
# FINAL INSIGHT
# -------------------------------
print("\n📌 INSIGHTS:")
print("- Strong relationship between Maths and Science")
print("- Few products generate most revenue")
print("- Sales fluctuate monthly")
