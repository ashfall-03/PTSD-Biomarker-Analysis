import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("ptsd_study_data.csv")

# 1. Data Cleaning: Focus on biological integrity
# Dropping subjects with missing biomarker data
df_clean = df.dropna(subset=['cortisol_level', 'trauma_score'])

# 2. Descriptive Statistics
print("Summary Statistics for Cortisol Levels:")
print(df_clean.groupby('group_type')['cortisol_level'].describe())

# 3. Visualization: The 'Core' of Neuro-Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='group_type', y='cortisol_level', data=df_clean, palette='Set2')
plt.title('Cortisol Reactivity: Trauma-Exposed vs. Control Groups')
plt.ylabel('Cortisol Concentration (ng/mL)')
plt.xlabel('Subject Group')

# 4. Statistical Hypothesis Testing
trauma_group = df_clean[df_clean['group_type'] == 'Trauma']['cortisol_level']
control_group = df_clean[df_clean['group_type'] == 'Control']['cortisol_level']

t_stat, p_val = stats.ttest_ind(trauma_group, control_group)
print(f"\nT-test Result: p-value = {p_val:.4f}")

plt.show()

# 5. Correlation Analysis
# We calculate the correlation matrix for our numerical variables
corr_matrix = df_clean[['cortisol_level', 'trauma_score']].corr()

# 6. Heatmap Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix: Stress Biomarkers vs. Trauma Scores')

# 7. Regression Plot (Visualizing the trend line)
plt.figure(figsize=(10, 6))
sns.regplot(x='trauma_score', y='cortisol_level', data=df_clean, 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Linear Regression: Trauma Score as a Predictor of Cortisol Blunting')
plt.xlabel('Trauma Score (Severity)')
plt.ylabel('Baseline Cortisol (ng/mL)')

plt.show()
