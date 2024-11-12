#This was my Ai/Ml project about a students next day planner according to his/her current day schedule.....
#I got my idea about the demo project through this project only.


import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tkinter import messagebox
import tkinter as tk


data = pd.read_csv("C:\\Users\\ASUS\\Desktop\\Elements of AiML\\Untitled form (Responses) - Form responses 1.csv")

# Check the initial data types
# print("Initial data types:")
# print(data.dtypes)


def map_time_to_hours(value):
    if value == 'Less than 1 hour':
        return 0.5
    elif value == '1-2 hours':
        return 1.5
    elif value == '3-4 hours':
        return 3.5
    elif value == '5-6 hours':
        return 5.5
    elif value == 'More than 6 hours':
        return 7.5  
    else:
        return 0  


data['study_time_today'] = data['How many hours did you study today?'].apply(map_time_to_hours)
data['sleep_time_today'] = data['How many hours of sleep did you get last night?'].apply(map_time_to_hours)
data['social_media_time_today'] = data['How many hours did you spend on social media today?'].apply(map_time_to_hours)
data['physical_activity_time_today'] = data['How many hours did you spend on physical activity/exercise today?'].apply(map_time_to_hours)
data['class_time_today'] = data['How many hours did you spend attending classes today (including lectures and practicals)?'].apply(map_time_to_hours)
data['extracurricular_time_today'] = data['How much time did you spend on extracurricular activities (clubs, sports, etc.)?'].apply(map_time_to_hours)
data['relaxing_time_today'] = data['How many hours did you spend relaxing or on hobbies (reading, games, music, etc.)?'].apply(map_time_to_hours)


data['gender'] = data['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})


data['productive_today'] = pd.to_numeric(data['How productive do you feel today on a scale of 1 to 5?'], errors='coerce')
data['productive_today'] = data['productive_today'].fillna(data['productive_today'].mean())  


data['stressed_today'] = pd.to_numeric(data['How Stressed do you feel today on a scale of 1 to 5?'], errors='coerce')
data['stressed_today'] = data['stressed_today'].fillna(data['stressed_today'].mean())  


data.fillna(0, inplace=True)


X = data[['study_time_today', 'sleep_time_today', 'social_media_time_today',
           'physical_activity_time_today', 'class_time_today', 
           'extracurricular_time_today', 'relaxing_time_today', 
           'gender', 'productive_today', 'stressed_today']]
y = data['next_day_schedule']  

model = DecisionTreeClassifier()


kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')


print("Accuracy for each fold: ", scores)


print(f'Average accuracy across all folds: {scores.mean() * 100:.2f}%')


def recommend_next_day_routine(study_time, sleep_time, social_media_time, physical_activity_time, stress_level, productivity_level, gender):
    
    input_data = pd.DataFrame({
        'study_time_today': [study_time],
        'sleep_time_today': [sleep_time],
        'social_media_time_today': [social_media_time],
        'physical_activity_time_today': [physical_activity_time],
        'class_time_today': [0],  
        'extracurricular_time_today': [0],
        'relaxing_time_today': [0],
        'gender': [gender],
        'productive_today': [productivity_level],
        'stressed_today': [stress_level]
    })
    
    
    model.fit(X, y)
    
    
    predicted_routine = model.predict(input_data)
    return predicted_routine[0]


next_day_schedule = recommend_next_day_routine(1.5, 6, 2, 0.5, 3, 4, 0)
print(f"Recommended next day's schedule: {next_day_schedule}")


def calculate_routine():
    try:
        study_time = float(study_time_entry.get())
        sleep_time = float(sleep_time_entry.get())
        social_media_time = float(social_media_time_entry.get())
        physical_activity_time = float(physical_activity_time_entry.get())
        stress_level = float(stress_level_entry.get())
        productivity_level = float(productivity_level_entry.get())
        gender = gender_var.get()

        
        next_day_routine = recommend_next_day_routine(study_time, sleep_time, social_media_time, physical_activity_time, stress_level, productivity_level, gender)

        
        messagebox.showinfo("Next Day Routine", f"Recommended next day's schedule: {next_day_routine}")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers.")


root = tk.Tk()
root.title("Next Day Routine Predictor")


tk.Label(root, text="Study Time Today (hours)").grid(row=0)
study_time_entry = tk.Entry(root)
study_time_entry.grid(row=0, column=1)

tk.Label(root, text="Sleep Time Today (hours)").grid(row=1)
sleep_time_entry = tk.Entry(root)
sleep_time_entry.grid(row=1, column=1)

tk.Label(root, text="Social Media Time Today (hours)").grid(row=2)
social_media_time_entry = tk.Entry(root)
social_media_time_entry.grid(row=2, column=1)

tk.Label(root, text="Physical Activity Time Today (hours)").grid(row=3)
physical_activity_time_entry = tk.Entry(root)
physical_activity_time_entry.grid(row=3, column=1)

tk.Label(root, text="Stress Level (1-5)").grid(row=4)
stress_level_entry = tk.Entry(root)
stress_level_entry.grid(row=4, column=1)

tk.Label(root, text="Productivity Level (1-5)").grid(row=5)
productivity_level_entry = tk.Entry(root)
productivity_level_entry.grid(row=5, column=1)

tk.Label(root, text="Gender (0: Male, 1: Female, 2: Other)").grid(row=6)
gender_var = tk.IntVar(value=0)
gender_entry = tk.Entry(root, textvariable=gender_var)
gender_entry.grid(row=6, column=1)


calculate_button = tk.Button(root, text="Calculate Next Day Routine", command=calculate_routine)
calculate_button.grid(row=7, column=1)


root.mainloop()



