# generate_data.py
from pymongo import MongoClient
from faker import Faker
import random
import numpy as np

# ---------------------------------
# 1️⃣ CONNECT TO MONGODB
# ---------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["e_learning"]

# Clean old data
for col in ["learners", "teachers", "subjects", "marks", "specialties", "quizzes", "answers"]:
    db[col].delete_many({})

fake = Faker()

# ---------------------------------
# 2️⃣ SPECIALTIES
# ---------------------------------
specialties = [
    {"id_sp": 1, "name": "Computer Science"},
    {"id_sp": 2, "name": "Mathematics"},
    {"id_sp": 3, "name": "Physics"},
    {"id_sp": 4, "name": "Biology"}
]
db.specialties.insert_many(specialties)

# ---------------------------------
# 3️⃣ TEACHERS
# ---------------------------------
teachers = []
for i in range(50):
    teachers.append({
        "id_t": i + 1,
        "name": fake.name(),
        "specialty_id": random.choice(specialties)["id_sp"]
    })
db.teachers.insert_many(teachers)

# ---------------------------------
# 4️⃣ LEARNERS
# ---------------------------------
learners = []
for i in range(2500):
    learners.append({
        "id_l": i + 1,
        "name": fake.name(),
        "specialty_id": random.choice(specialties)["id_sp"]
    })
db.learners.insert_many(learners)

# ---------------------------------
# 5️⃣ SUBJECTS
# ---------------------------------
subjects = [
    {"id_s": 1, "name": "Mathematics"},
    {"id_s": 2, "name": "Physics"},
    {"id_s": 3, "name": "Biology"},
    {"id_s": 4, "name": "Computer Science"}
]
db.subjects.insert_many(subjects)

# ---------------------------------
# 6️⃣ QUIZZES
# ---------------------------------
quizzes = []
quiz_id = 1
for subj in subjects:
    for _ in range(10):  # 10 quizzes per subject
        quizzes.append({
            "id_q": quiz_id,
            "title": f"Quiz {quiz_id} - {subj['name']}",
            "subject_id": subj["id_s"]
        })
        quiz_id += 1
db.quizzes.insert_many(quizzes)

# ---------------------------------
# 7️⃣ ANSWERS (simulate learner performance)
# ---------------------------------
answers = []
for learner in learners:
    n_quizzes = random.randint(5, 15)  # each learner takes 5–15 quizzes
    taken = random.sample(quizzes, n_quizzes)

    for q in taken:
        correct_rate = random.uniform(0.3, 1.0)  # how many correct answers (0.3 = weak, 1.0 = perfect)
        score = round(correct_rate * 20, 2)
        answers.append({
            "id_l": learner["id_l"],
            "id_q": q["id_q"],
            "subject_id": q["subject_id"],
            "score": score,
            "correct_rate": correct_rate
        })

db.answers.insert_many(answers)

# ---------------------------------
# 8️⃣ MARKS (average by subject)
# ---------------------------------
marks = []
for learner in learners:
    for subj in subjects:
        subj_quizzes = [a for a in answers if a["id_l"] == learner["id_l"] and a["subject_id"] == subj["id_s"]]
        if subj_quizzes:
            avg_mark = np.mean([a["score"] for a in subj_quizzes])
        else:
            avg_mark = random.uniform(6, 15)  # fallback

        marks.append({
            "id_l": learner["id_l"],
            "id_s": subj["id_s"],
            "mark": round(avg_mark, 2)
        })
db.marks.insert_many(marks)

print("✅ Database filled with realistic data:")
print(f"   • {len(learners)} learners")
print(f"   • {len(teachers)} teachers")
print(f"   • {len(quizzes)} quizzes")
print(f"   • {len(answers)} answers")
print(f"   • {len(marks)} marks")
print("\n🎯 Ready for model training (Ann_trainmodel.py)")

