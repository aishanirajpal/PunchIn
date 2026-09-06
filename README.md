# PunchIn
PunchIn — Face Recognition Attendance System

A face recognition–based attendance management system built for educational institutions using Python, Streamlit, OpenCV, MTCNN, LBPH, and SQLite.

Live Demo: PunchIn Attendance System

Overview

PunchIn automates student attendance using facial recognition. Students can be registered, their face data captured and used to train an LBPH recognition model. The system then identifies students through a webcam and records their attendance with date, time, and approximate location.

Features
Student registration and management
Webcam-based face data collection
MTCNN face detection
LBPH face recognition
Check-in and check-out attendance
Duplicate attendance prevention
IP-based location tracking
Attendance filtering and CSV export
Daily, weekly, monthly, and custom attendance analytics
Timetable management
SQLite database
Database backup and reset
Technology Stack
Technology	Purpose
Python	Application development
Streamlit	Web interface
OpenCV	Computer vision and LBPH
MTCNN	Face detection
SQLite	Database
Pandas	Data processing
Matplotlib	Data visualization
Geocoder	Location information
Architecture
Webcam
   |
   v
MTCNN Face Detection
   |
   v
LBPH Face Recognition
   |
   v
Student Database
   |
   v
Attendance Record
   |
   +--> Date / Time
   +--> Check-in / Check-out
   +--> Location
Project Structure
PunchIn/
├── app.py
├── attendance-system-demo.py
├── requirements.txt
├── runtime.txt
├── dataset/
├── trainer/
└── attendance.db

dataset/, trainer/, and attendance.db are generated during application use and should not contain real student data in a public repository.
