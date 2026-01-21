import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

# -------------------------
# Candidate Info
# -------------------------
age = np.random.randint(20, 45, N)
gender = np.random.choice(["Male", "Female"], N, p=[0.6, 0.4])
location = np.random.choice(["Germany", "France", "Italy", "Spain", "Poland"], N)

# -------------------------
# Lead / Enquiry Info
# -------------------------
lead_source = np.random.choice(
    ["Website", "Referral", "LinkedIn", "Email Campaign", "Walk-in"],
    N,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

interest_level = np.random.randint(1, 6, N)  # 1 (low) to 5 (high)
follow_ups = np.random.randint(0, 8, N)

lead_score = (interest_level * 2) + follow_ups + np.random.normal(0, 1, N)
lead_converted = (lead_score > 8).astype(int)

# -------------------------
# Training Info
# -------------------------
course = np.random.choice(
    ["Data Science", "ML Engineering", "HR Analytics", "Python", "Business Analytics"],
    N
)

course_level = np.random.choice(["Beginner", "Intermediate", "Advanced"], N)
duration_weeks = np.random.randint(4, 24, N)

# -------------------------
# Candidate Performance
# -------------------------
attendance = np.random.randint(60, 100, N)
test_score = np.random.randint(40, 100, N)
project_marks = np.random.randint(45, 100, N)

performance_score = (
    0.4 * test_score +
    0.4 * project_marks +
    0.2 * attendance
)

# -------------------------
# PIP & Top Performer Flags
# -------------------------
pip_flag = (performance_score < 60).astype(int)
top_performer_flag = (performance_score > 85).astype(int)

# -------------------------
# Joining & Employment Info
# -------------------------
joined = ((lead_converted == 1) & (performance_score > 55)).astype(int)

employment_type = np.where(
    (performance_score > 80) & (joined == 1),
    "Permanent",
    "Contract"
)

# -------------------------
# Salary Components (EUR)
# -------------------------
basic_salary = np.where(
    employment_type == "Permanent",
    np.random.randint(2500, 4000, N),
    np.random.randint(1800, 3000, N)
)

hra = basic_salary * np.random.uniform(0.18, 0.25, N)
ta = basic_salary * np.random.uniform(0.08, 0.12, N)
pf = basic_salary * np.random.uniform(0.10, 0.12, N)

ctc = basic_salary + hra + ta + pf

# -------------------------
# Final Dataset
# -------------------------
df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Location": location,
    "Lead_Source": lead_source,
    "Interest_Level": interest_level,
    "Follow_Ups": follow_ups,
    "Lead_Converted": lead_converted,
    "Course": course,
    "Course_Level": course_level,
    "Course_Duration_Weeks": duration_weeks,
    "Attendance": attendance,
    "Test_Score": test_score,
    "Project_Marks": project_marks,
    "Performance_Score": performance_score.round(2),
    "PIP_Flag": pip_flag,
    "Top_Performer_Flag": top_performer_flag,
    "Joined": joined,
    "Employment_Type": employment_type,
    "Basic_Salary": basic_salary,
    "HRA": hra.round(2),
    "TA": ta.round(2),
    "PF": pf.round(2),
    "CTC": ctc.round(2)
})

# Save to CSV
df.to_csv("euromax_hr_synthetic_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())