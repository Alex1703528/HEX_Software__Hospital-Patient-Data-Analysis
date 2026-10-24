# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Loading Data
df = pd.read_csv('healthcare_dataset.csv')
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Cleaning & Feature Engineering
# Standardizing dates
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Calculating Length of Stay (in days)
df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Filtering out erroneous negative billing amounts
initial_count = len(df)
df_clean = df[df['Billing Amount'] >= 0].copy()
print(f"Filtered {initial_count - len(df_clean)} records with negative billing amounts.")

# 3. Exploratory Data Analysis & Visualizations

# Chart 1: Age Distribution by Gender
plt.figure(figsize=(9, 5))
sns.histplot(data=df_clean, x='Age', hue='Gender', bins=25, kde=True, palette={'Male': '#2b5c8f', 'Female': '#d95f02'})
plt.title('Patient Age Distribution by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Patient Age')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig('HEX_Project_2.1_age_distribution.png', dpi=300)
plt.close()

# Chart 2: Admission Types Breakdown
plt.figure(figsize=(8, 5))
admission_counts = df_clean['Admission Type'].value_counts()
sns.barplot(x=admission_counts.index, y=admission_counts.values, palette='Blues_r')
plt.title('Patient Volume by Admission Type', fontsize=14, fontweight='bold')
plt.xlabel('Admission Type')
plt.ylabel('Total Admissions')
for i, v in enumerate(admission_counts.values):
    plt.text(i, v + 200, f"{v:,}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('HEX_Project_2.2_admission_types.png', dpi=300)
plt.close()

# Chart 3: Average Length of Stay by Medical Condition
plt.figure(figsize=(10, 5))
los_by_condition = df_clean.groupby('Medical Condition')['Length_of_Stay'].mean().sort_values()
sns.barplot(x=los_by_condition.values, y=los_by_condition.index, palette='crest')
plt.title('Average Length of Stay by Medical Condition (Days)', fontsize=14, fontweight='bold')
plt.xlabel('Average Days in Hospital')
plt.ylabel('Medical Condition')
plt.xlim(10, 18)
for i, v in enumerate(los_by_condition.values):
    plt.text(v + 0.1, i, f"{v:.1f} days", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('HEX_Project_2.3_length_of_stay.png', dpi=300)
plt.close()

# Chart 4: Billing Amount Distribution across Medical Conditions
plt.figure(figsize=(11, 6))
sns.boxplot(data=df_clean, x='Medical Condition', y='Billing Amount', palette='Set3')
plt.title('Billing Amount Range by Medical Condition', fontsize=14, fontweight='bold')
plt.xlabel('Medical Condition')
plt.ylabel('Billing Amount ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('HEX_Project_2.4_billing_by_condition.png', dpi=300)
plt.close()

# 4. Export Statistical Summary Table
summary_metrics = df_clean.groupby('Medical Condition').agg(
    Patient_Count=('Name', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Stay_Days=('Length_of_Stay', 'mean'),
    Avg_Billing=('Billing Amount', 'mean'),
    Median_Billing=('Billing Amount', 'median')
).round(2)

print("\n--- Summary Performance by Condition ---")
print(summary_metrics)
summary_metrics.to_csv('hospital_kpi_summary.csv')