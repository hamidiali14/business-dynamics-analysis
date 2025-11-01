import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# === Load dataset ===
file_path = r"C:\Users\lenovo\Downloads\bds2022_st_sec.csv"
df = pd.read_csv(file_path)

# Preview
print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nShape of dataset:", df.shape)

# STEP 2: Explore the dataset
print("\n--- Dataset Shape ---")
print(df.shape)  # (rows, columns)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))

# STEP 3: Data Cleaning

# Replace 'D' (suppressed values) with NaN
df_clean = df.replace('D', np.nan)

# Convert numeric-looking columns from object to numeric
for col in df_clean.columns:
    if col not in ['year', 'st', 'sector']:  # keep identifiers as is
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("\n--- Cleaned Data Types ---")
print(df_clean.dtypes)

print("\n--- Sample After Cleaning ---")
print(df_clean.head())

# STEP 4: Aggregate by year (across all states and sectors)
yearly = df_clean.groupby("year")[["firms", "estabs", "emp"]].sum().reset_index()

print("\n--- Yearly Aggregated Data ---")
print(yearly.head())

# STEP 5: Plot trends
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(yearly["year"], yearly["firms"], label="Firms", marker="o")
plt.plot(yearly["year"], yearly["estabs"], label="Establishments", marker="o")
plt.plot(yearly["year"], yearly["emp"]/1000, label="Employment (in 1000s)", marker="o")  # divide for readability

plt.title("Business Dynamics in the U.S. (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# STEP 6: Compute Growth Rates
yearly["firm_growth"] = yearly["firms"].pct_change() * 100
yearly["estab_growth"] = yearly["estabs"].pct_change() * 100
yearly["emp_growth"] = yearly["emp"].pct_change() * 100

print("\n--- Growth Rates (first 10 rows) ---")
print(yearly[["year", "firm_growth", "estab_growth", "emp_growth"]].head(10))

# STEP 7: Aggregate Net Job Creation
net_jobs = df_clean.groupby("year")["net_job_creation"].sum().reset_index()

# Plot growth rates
plt.figure(figsize=(12,6))
plt.plot(yearly["year"], yearly["firm_growth"], label="Firm Growth (%)", marker="o")
plt.plot(yearly["year"], yearly["estab_growth"], label="Establishment Growth (%)", marker="o")
plt.plot(yearly["year"], yearly["emp_growth"], label="Employment Growth (%)", marker="o")
plt.axhline(0, color="black", linewidth=1)
plt.title("Year-over-Year Growth Rates (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Plot net job creation
plt.figure(figsize=(12,6))
plt.plot(net_jobs["year"], net_jobs["net_job_creation"], label="Net Job Creation", color="green", marker="o")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Net Job Creation in the U.S. (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Jobs Created (Net)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# STEP 8: Sector-Level Analysis

# Average employment by sector (over all years)
sector_emp = df_clean.groupby("sector")["emp"].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=sector_emp.head(10), x="emp", y="sector", palette="viridis")
plt.title("Top 10 Sectors by Average Employment (1978–2022)")
plt.xlabel("Average Employment")
plt.ylabel("Sector")
plt.show()

# Net job creation by sector (total over all years)
sector_jobs = df_clean.groupby("sector")["net_job_creation"].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=sector_jobs.head(10), x="net_job_creation", y="sector", palette="coolwarm")
plt.title("Top 10 Sectors by Net Job Creation (1978–2022)")
plt.xlabel("Net Job Creation")
plt.ylabel("Sector")
plt.show()

# Bottom 10 (declining sectors)
plt.figure(figsize=(12,6))
sns.barplot(data=sector_jobs.tail(10), x="net_job_creation", y="sector", palette="coolwarm")
plt.title("Bottom 10 Sectors by Net Job Creation (1978–2022)")
plt.xlabel("Net Job Creation")
plt.ylabel("Sector")
plt.show()

# Employment over time (all sectors, total)
emp_over_time = df_clean.groupby("year")["emp"].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=emp_over_time, x="year", y="emp", marker="o")
plt.title("Total Employment in the US (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Total Employment")
plt.grid(True)
plt.show()

# Net job creation over time (all sectors, total)
jobs_over_time = df_clean.groupby("year")["net_job_creation"].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=jobs_over_time, x="year", y="net_job_creation", marker="o", color="green")
plt.title("Net Job Creation in the US (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Net Job Creation")
plt.axhline(0, color="red", linestyle="--")  # reference line at 0
plt.grid(True)
plt.show()

# Firm births & deaths over time
firm_births = df_clean.groupby("year")["estabs_entry"].sum().reset_index()
firm_deaths = df_clean.groupby("year")["estabs_exit"].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=firm_births, x="year", y="estabs_entry", label="Firm Births", color="blue")
sns.lineplot(data=firm_deaths, x="year", y="estabs_exit", label="Firm Deaths", color="orange")
plt.title("Firm Births vs Deaths in the US (1978–2022)")
plt.xlabel("Year")
plt.ylabel("Number of Firms")
plt.legend()
plt.grid(True)
plt.show()

# --- Employment by State (Latest Year) ---
latest_year = df_clean["year"].max()
emp_by_state = df_clean[df_clean["year"] == latest_year].groupby("st")["emp"].sum().reset_index()

plt.figure(figsize=(14,6))
sns.barplot(data=emp_by_state.sort_values("emp", ascending=False).head(10),
            x="st", y="emp", palette="coolwarm")
plt.title(f"Top 10 States by Employment in {latest_year}")
plt.xlabel("State Code")
plt.ylabel("Employment")
plt.show()


# --- Job Creation Rate by State (Latest Year) ---
job_creation_by_state = df_clean[df_clean["year"] == latest_year].groupby("st")["net_job_creation_rate"].mean().reset_index()

plt.figure(figsize=(14,6))
sns.barplot(data=job_creation_by_state.sort_values("net_job_creation_rate", ascending=False).head(10),
            x="st", y="net_job_creation_rate", palette="viridis")
plt.title(f"Top 10 States by Net Job Creation Rate in {latest_year}")
plt.xlabel("State Code")
plt.ylabel("Net Job Creation Rate")
plt.show()


# --- Firm Births vs Deaths by State (Latest Year) ---
firm_births_state = df_clean[df_clean["year"] == latest_year].groupby("st")["estabs_entry"].sum().reset_index()
firm_deaths_state = df_clean[df_clean["year"] == latest_year].groupby("st")["estabs_exit"].sum().reset_index()

plt.figure(figsize=(14,6))
sns.scatterplot(data=firm_births_state.merge(firm_deaths_state, on="st"),
                x="estabs_entry", y="estabs_exit", hue="st", palette="tab20", legend=False)
plt.title(f"Firm Births vs Firm Deaths by State ({latest_year})")
plt.xlabel("Firm Births")
plt.ylabel("Firm Deaths")
plt.grid(True)
plt.show()

# --- Correlation Matrix ---
numeric_cols = df_clean.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix of Key Business Dynamics Variables")
plt.show()


# --- Focused correlations ---
print("\nCorrelation of Employment (emp) with key factors:")
print(correlation_matrix["emp"][["estabs_entry", "estabs_exit", "job_creation", "job_destruction", "net_job_creation"]])

# --- Statistical Test: Does job creation significantly impact employment? ---
from scipy.stats import pearsonr

emp = df_clean["emp"].dropna()
job_creation = df_clean["job_creation"].dropna()

corr_coef, p_value = pearsonr(emp, job_creation)
print(f"\nPearson correlation between Employment & Job Creation: {corr_coef:.3f}, p-value={p_value:.3e}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Features and target ---
features = ["job_creation", "job_destruction", "estabs_entry", "estabs_exit"]
X = df_clean[features]
y = df_clean["emp"]

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- Regression Results ---")
print("R² (explained variance):", round(r2, 3))
print("Mean Squared Error:", round(mse, 2))
print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)

import matplotlib.pyplot as plt
import seaborn as sns

# --- Residuals ---
residuals = y_test - y_pred

# Scatter plot: predicted vs actual
plt.figure(figsize=(6,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Employment")
plt.ylabel("Predicted Employment")
plt.title("Predicted vs Actual Employment")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # 45-degree line
plt.show()

# Residuals distribution
plt.figure(figsize=(6,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.show()

# Aggregate employment by year
employment_trend = df_clean.groupby("year")["emp"].sum().reset_index()

print("\n--- Employment by Year ---")
print(employment_trend.head())

# Plot employment trend
plt.figure(figsize=(10,6))
sns.lineplot(data=employment_trend, x="year", y="emp", marker="o")
plt.title("Total Employment Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Employment")
plt.grid(True)
plt.show()


