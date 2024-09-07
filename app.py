import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load and preprocess data
def load_data():
    df = pd.read_csv("stud_training.csv")
    df.replace({'Courses': {'BBA- Bachelor of Business Administration': 0,
                             'BEM- Bachelor of Event Management': 1,
                             'Integrated Law Course- BA + LL.B': 2,
                             'BJMC- Bachelor of Journalism and Mass Communication': 3,
                             'BFD- Bachelor of Fashion Designing': 4,
                             'BBS- Bachelor of Business Studies': 5,
                             'BTTM- Bachelor of Travel and Tourism Management': 6,
                             'BVA- Bachelor of Visual Arts': 7,
                             'BA in History': 8,
                             'B.Arch- Bachelor of Architecture': 9,
                             'BCA- Bachelor of Computer Applications': 10,
                             'B.Sc.- Information Technology': 11,
                             'B.Sc- Nursing': 12,
                             'BPharma- Bachelor of Pharmacy': 13,
                             'BDS- Bachelor of Dental Surgery': 14,
                             'Animation, Graphics and Multimedia': 15,
                             'B.Sc- Applied Geology': 16,
                             'B.Sc.- Physics': 17,
                             'B.Sc. Chemistry': 18,
                             'B.Sc. Mathematics': 19,
                             'B.Tech.-Civil Engineering': 20,
                             'B.Tech.-Computer Science and Engineering': 21,
                             'B.Tech.-Electrical and Electronics Engineering': 22,
                             'B.Tech.-Electronics and Communication Engineering': 23,
                             'B.Tech.-Mechanical Engineering': 24,
                             'B.Com- Bachelor of Commerce': 25,
                             'BA in Economics': 26,
                             'CA- Chartered Accountancy': 27,
                             'CS- Company Secretary': 28,
                             'Diploma in Dramatic Arts': 29,
                             'MBBS': 30,
                             'Civil Services': 31,
                             'BA in English': 32,
                             'BA in Hindi': 33,
                             'B.Ed.': 34
                            }}, inplace=True)
    
    l1 = ['Drawing', 'Dancing', 'Singing', 'Sports', 'Video Game', 'Acting', 'Travelling',
          'Gardening', 'Animals', 'Photography', 'Teaching', 'Exercise', 'Coding',
          'Electricity Components', 'Mechanic Parts', 'Computer Parts', 'Researching',
          'Architecture', 'Historic Collection', 'Botany', 'Zoology', 'Physics',
          'Accounting', 'Economics', 'Sociology', 'Geography', 'Psychology', 'History',
          'Science', 'Business Education', 'Chemistry', 'Mathematics', 'Biology', 'Makeup',
          'Designing', 'Content writing', 'Crafting', 'Literature', 'Reading', 'Cartooning',
          'Debating', 'Astrology', 'Hindi', 'French', 'English', 'Other Language',
          'Solving Puzzles', 'Gymnastics', 'Yoga', 'Engineering', 'Doctor', 'Pharmacist',
          'Cycling', 'Knitting', 'Director', 'Journalism', 'Business', 'Listening Music']
    
    X = df[l1]
    y = df[['Courses']]
    np.ravel(y)
    
    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(X, np.ravel(y))
    
    clf4 = RandomForestClassifier()
    clf4.fit(X, np.ravel(y))
    
    gnb = GaussianNB()
    gnb.fit(X, np.ravel(y))
    
    return l1, clf3, clf4, gnb

l1, clf3, clf4, gnb = load_data()

# Streamlit UI
st.title("Course Prediction Based on Interests")

interests = []
for i in range(1, 6):
    interest = st.text_input(f"Interest {i}", "")
    if interest:
        interests.append(interest)

def predict_course(interests, clf):
    l2 = [0] * len(l1)
    for z in interests:
        if z in l1:
            l2[l1.index(z)] = 1
    inputtest = [l2]
    predict = clf.predict(inputtest)
    predicted = predict[0]
    return Course[predicted]

if st.button('Predict Course'):
    if interests:
        predicted_course_dt = predict_course(interests, clf3)
        predicted_course_rf = predict_course(interests, clf4)
        predicted_course_nb = predict_course(interests, gnb)
        
        st.write(f"Predicted Course (Decision Tree): {predicted_course_dt}")
        st.write(f"Predicted Course (Random Forest): {predicted_course_rf}")
        st.write(f"Predicted Course (Naive Bayes): {predicted_course_nb}")
    else:
        st.write("Please enter your interests.")
