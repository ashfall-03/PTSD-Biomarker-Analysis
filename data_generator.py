import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# 1. Create demographics
n_subjects = 200
groups = ['Control'] * 100 + ['Trauma'] * 100

# 2. Simulate Cortisol Levels (ng/mL) 
# Trauma groups often show "blunted" cortisol reactivity
control_cortisol = np.random.normal(loc=15, scale=3, size=100)
trauma_cortisol = np.random.normal(loc=10, scale=4, size=100)

# 3. Simulate Trauma Scores (0-100 scale)
control_scores = np.random.randint(0, 30, size=100)
trauma_scores = np.random.randint(40, 95, size=100)

# 4. Assemble the DataFrame
data = {
    'subject_id': range(1, n_subjects + 1),
    'group_type': groups,
    'cortisol_level': np.concatenate([control_cortisol, trauma_cortisol]),
    'trauma_score': np.concatenate([control_scores, trauma_scores])
}

df = pd.DataFrame(data)

# 5. Save to CSV
df.to_excel("ptsd_study_data.xlsx", index=False) # Saving as Excel for your tracker
df.to_csv("ptsd_study_data.csv", index=False)
print("Dataset 'ptsd_study_data.csv' has been generated successfully!")