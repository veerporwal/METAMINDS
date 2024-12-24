import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("face-attendence-with-real-time-firebase-adminsdk-fw8ua-7cd123b24c.json")
firebase_admin.initialize_app(cred,{'databaseURL': 'https://face-attendence-with-real-time-default-rtdb.firebaseio.com/'})

ref = db.reference('students')

data = {
    '21019C04061':
    {
        'Name' : 'Veer Porwal',
        'Branch' : 'CSE',
        'semester' : '6',
        'Last_attendance' : '2024-03-01 00:27:12',
        "total_attendance" : 8,
        "starting_year" : "2021",
        "current_year" : '3'
    },
    "963852":
    {
        'Name' : 'Elon Musk',
        'Branch' : 'CSE',
        'semester' : '6',
        'Last_attendance' : '2024-03-01 00:29:13',
        "total_attendance" : 1,
        "starting_year" : "2021",
        "current_year" : "3"
    }
}
for key, value in data.items():
    ref.child(key).set(value)