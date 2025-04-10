import streamlit as st
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import time
import os
from mtcnn import MTCNN
from PIL import Image
import pandas as pd
import geocoder  # For location tracking
import matplotlib.pyplot as plt
import io
import base64

# Setup page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="✅",
    layout="wide"
)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'attendance.db')
dataset_path = os.path.join(BASE_DIR, 'dataset')
model_path = os.path.join(BASE_DIR, 'trainer/trainer.yml')

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, 'dataset'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'trainer'), exist_ok=True)

# Database setup function
def setup_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        roll TEXT UNIQUE NOT NULL,
        department TEXT,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY,
        student_id INTEGER,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        check_type TEXT DEFAULT 'in',
        location TEXT,
        latitude REAL,
        longitude REAL,
        FOREIGN KEY (student_id) REFERENCES students(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        code TEXT UNIQUE NOT NULL,
        instructor TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS timetable (
        id INTEGER PRIMARY KEY,
        course_id INTEGER,
        day TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        room TEXT,
        FOREIGN KEY (course_id) REFERENCES courses(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Get student name from ID
def get_student_name(student_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    return "Unknown"

# Sidebar Menu
def sidebar_menu():
    st.sidebar.title("Smart Attendance")
    
    # User info section
    st.sidebar.subheader("Navigation")
    
    # Menu options
    menu = st.sidebar.radio(
        "Select Option:",
        ["Home", "Student Management", "Attendance Records", "Timetable", "Teachers", "Attendance Statistics", "Settings"]
    )
    
    # Display current date and time
    st.sidebar.markdown("---")
    st.sidebar.info(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    st.sidebar.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    return menu

# Home Page
def home_page():
    st.title("Smart Attendance System")
    st.write("Welcome to the facial recognition-based attendance system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mark Attendance")
        if st.button("👉 Check In", use_container_width=True):
            mark_attendance_with_location("in")
            
    with col2:
        st.subheader("Mark Departure")
        if st.button("👋 Check Out", use_container_width=True):
            mark_attendance_with_location("out")
    
    # System overview
    st.markdown("---")
    st.subheader("System Overview")
    
    # Get stats from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]
    
    # Total attendance records
    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_records = cursor.fetchone()[0]
    
    # Today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today,))
    today_attendance = cursor.fetchone()[0]
    
    conn.close()
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", total_students)
    col2.metric("Total Attendance Records", total_records)
    col3.metric("Today's Attendance", today_attendance, f"{100 * today_attendance / max(1, total_students):.1f}%")
    
    # Display recent activity
    st.markdown("---")
    st.subheader("Recent Activity")
    display_recent_activity()

# Mark attendance with location
def mark_attendance_with_location(check_type):
    # Initialize detector
    detector = MTCNN()
    
    # Try to load recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    except:
        st.error("Face recognition model not found! Please train the system first.")
        return
    
    # Get location
    try:
        g = geocoder.ip('me')
        location = f"{g.city}, {g.state}, {g.country}"
        latitude, longitude = g.latlng
    except:
        location = "Location unavailable"
        latitude, longitude = 0, 0
        
    # Camera section
    st.subheader(f"Face Recognition for {check_type.capitalize()} Attendance")
    st.write(f"📍 Your location: {location}")
    
    # Camera placeholder
    camera_col, info_col = st.columns([3, 1])
    camera_placeholder = camera_col.empty()
    status_placeholder = info_col.empty()
    progress_bar = st.progress(0)
    result_placeholder = st.empty()
    
    status_placeholder.info("Starting camera...")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        result_placeholder.error("Cannot open camera. Please check your webcam connection.")
        return
    
    # Face recognition loop
    max_frames = 30  # Process 30 frames for recognition
    frames = 0
    recognized_ids = {}  # Store recognized IDs and their counts
    
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            result_placeholder.error("Failed to grab frame")
            break
            
        # Display camera feed
        camera_placeholder.image(frame, channels="BGR", caption="Camera Feed", use_column_width=True)
        
        # Process for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(rgb_frame)
        
        if faces:
            status_text = f"Detected {len(faces)} faces"
        else:
            status_text = "No faces detected"
        
        status_placeholder.info(status_text)
        
        for face in faces:
            if face['confidence'] < 0.9:  # Filter low confidence detections
                continue
                
            x, y, w, h = face['box']
            # Ensure positive values
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            
            # Extract face region
            face_img = gray_frame[y:y+h, x:x+w]
            
            try:
                # Resize face for recognition if needed
                if face_img.size == 0:
                    continue
                    
                face_img = cv2.resize(face_img, (100, 100))
                
                # Recognize the face
                student_id, confidence = recognizer.predict(face_img)
                
                # Lower confidence means better match in LBPH
                if confidence < 80:  # Threshold for good recognition
                    # Count recognitions of this ID
                    if student_id in recognized_ids:
                        recognized_ids[student_id] += 1
                    else:
                        recognized_ids[student_id] = 1
                        
                    # Show recognition info
                    student_name = get_student_name(student_id)
                    status_placeholder.success(f"Recognized: {student_name}\nConfidence: {100-confidence:.1f}%")
            except Exception as e:
                status_placeholder.warning(f"Recognition error: {str(e)}")
        
        frames += 1
        progress_bar.progress(frames / max_frames)
        time.sleep(0.1)  # Short delay between frames
    
    cap.release()
    
    # Determine the most recognized face
    best_match = None
    max_count = 0
    
    for student_id, count in recognized_ids.items():
        if count > max_count:
            max_count = count
            best_match = student_id
    
    # Clear progress bar
    progress_bar.empty()
    
    # Process recognition result
    if best_match and max_count > 10:  # Require at least 10 successful recognitions
        # Get student name
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, roll, department FROM students WHERE id = ?", (best_match,))
        result = cursor.fetchone()
        
        if result:
            student_name, roll, department = result
            
            # Record attendance with location
            now = datetime.now()
            date_string = now.strftime("%Y-%m-%d")
            time_string = now.strftime("%H:%M:%S")
            
            # Check if already marked for this type today
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND check_type = ?
            """, (best_match, date_string, check_type))
            
            already_marked = cursor.fetchone() is not None
            
            if not already_marked:
                cursor.execute("""
                    INSERT INTO attendance 
                    (student_id, date, time, check_type, location, latitude, longitude) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (best_match, date_string, time_string, check_type, 
                     location, latitude, longitude))
                conn.commit()
                
                # Show success message
                result_placeholder.success(f"""
                ✅ {check_type.upper()} marked successfully!
                
                **Student**: {student_name}
                **Roll**: {roll}
                **Department**: {department}
                **Time**: {time_string}
                **Location**: {location}
                """)
            else:
                result_placeholder.warning(f"⚠️ {student_name} already marked {check_type} today.")
        else:
            result_placeholder.error("Student not found in database.")
            
        conn.close()
    else:
        result_placeholder.error("❌ No face recognized clearly. Please try again with better lighting and positioning.")

# Register new student
def register_student():
    st.subheader("Register New Student")
    
    # Form for student details
    with st.form("student_registration"):
        name = st.text_input("Full Name")
        roll = st.text_input("Roll Number")
        department = st.text_input("Department")
        
        submit_button = st.form_submit_button("Register")
        
    if submit_button:
        if not name or not roll or not department:
            st.error("Please fill all the fields")
            return
            
        # Check if roll number already exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM students WHERE roll = ?", (roll,))
        if cursor.fetchone():
            st.error(f"Roll number {roll} already exists!")
            conn.close()
            return
            
        # Insert new student
        try:
            cursor.execute("INSERT INTO students (name, roll, department) VALUES (?, ?, ?)", 
                         (name, roll, department))
            conn.commit()
            student_id = cursor.lastrowid
            conn.close()
            
            st.success(f"Student {name} registered successfully with ID: {student_id}")
            
            # Option to capture face data
            if st.button("Capture Face Data"):
                capture_face_data(student_id, name)
                
        except Exception as e:
            st.error(f"Error registering student: {str(e)}")
            conn.close()

# Capture facial data for training
def capture_face_data(student_id, name):
    st.subheader(f"Capturing Face Data for {name}")
    st.write("Please look at the camera and ensure good lighting")
    
    # Initialize detector
    detector = MTCNN()
    
    # Create directory for this person
    student_dir = os.path.join(dataset_path, f"student_{student_id}")
    os.makedirs(student_dir, exist_ok=True)
    
    # Camera placeholder
    col1, col2 = st.columns([3, 1])
    camera_placeholder = col1.empty()
    status_placeholder = col2.empty()
    progress_bar = st.progress(0)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot open camera")
        return
        
    # Capture parameters
    max_images = 30
    count = 0
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break
            
        # Display camera feed
        camera_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        # Detect faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(rgb_frame)
        
        if faces:
            status_placeholder.info(f"Detected {len(faces)} faces")
            
            for face in faces:
                if face['confidence'] < 0.9:  # Filter low confidence detections
                    continue
                    
                x, y, w, h = face['box']
                # Ensure positive values
                x, y = max(0, x), max(0, y)
                w, h = max(1, w), max(1, h)
                
                # Extract face region
                face_img = gray_frame[y:y+h, x:x+w]
                
                if face_img.size > 0:
                    # Save the captured face
                    img_name = os.path.join(student_dir, f"student_{student_id}_{count}.jpg")
                    cv2.imwrite(img_name, face_img)
                    
                    count += 1
                    status_placeholder.success(f"Captured image {count}/{max_images}")
                    progress_bar.progress(count / max_images)
                    time.sleep(0.2)  # Delay to avoid too many similar frames
                    break  # Only process the first valid face
        else:
            status_placeholder.warning("No face detected. Please adjust position.")
            
        time.sleep(0.1)
        
        if count >= max_images:
            break
    
    cap.release()
    
    if count >= max_images:
        st.success(f"Successfully captured {count} face samples")
        if st.button("Train Model"):
            train_model()
    else:
        st.warning(f"Only captured {count}/{max_images} images. Consider recapturing.")

# Train the face recognition model
def train_model():
    st.subheader("Training Face Recognition Model")
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Collecting face samples...")
    
    # Collect face samples and IDs
    faces = []
    ids = []
    
    # Walk through all face samples in the dataset directory
    total_dirs = len([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    current_dir = 0
    
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if not os.path.isdir(person_path):
            continue
            
        current_dir += 1
        progress_bar.progress(current_dir / (total_dirs * 2))  # First half is for collection
        
        # Extract student_id from directory name
        try:
            student_id = int(person_dir.split("_")[1])
        except:
            continue
            
        progress_text.text(f"Processing images for student ID: {student_id}")
        
        for img_file in os.listdir(person_path):
            if not (img_file.endswith(".jpg") or img_file.endswith(".png")):
                continue
                
            img_path = os.path.join(person_path, img_file)
            
            # Read the image and convert to grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and img.size > 0:
                # Add face and ID to arrays
                faces.append(img)
                ids.append(student_id)
    
    if not faces or not ids:
        st.error("No face samples found. Please register students and capture face data first.")
        return False
    
    progress_text.text("Training the model...")
    
    # Create and train the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    # Save the model
    progress_text.text("Saving the model...")
    progress_bar.progress(0.9)  # 90% complete
    
    recognizer.write(model_path)
    
    progress_bar.progress(1.0)  # 100% complete
    progress_text.text("Training complete!")
    
    st.success(f"Model trained successfully with {len(faces)} face samples from {len(set(ids))} students")
    return True

# Display recent activity
def display_recent_activity():
    conn = sqlite3.connect(db_path)
    query = """
        SELECT s.name, s.roll, a.date, a.time, a.check_type, a.location
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.id DESC
        LIMIT 10
    """
    recent_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not recent_df.empty:
        # Format check_type to be more readable
        recent_df['check_type'] = recent_df['check_type'].apply(lambda x: "Check In" if x == "in" else "Check Out")
        
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent activity")

# Student Management Page
def student_management():
    st.title("Student Management")
    
    tabs = st.tabs(["Register Student", "View Students", "Capture Face Data", "Train Model"])
    
    with tabs[0]:
        register_student()
        
    with tabs[1]:
        st.subheader("Registered Students")
        conn = sqlite3.connect(db_path)
        students_df = pd.read_sql_query("SELECT id, name, roll, department, created_date FROM students ORDER BY id", conn)
        conn.close()
        
        if not students_df.empty:
            st.dataframe(students_df, use_container_width=True)
            
            # Option to delete a student
            student_id = st.number_input("Student ID to delete", min_value=1, step=1)
            if st.button("Delete Student"):
                delete_student(student_id)
        else:
            st.info("No students registered yet")
    
    with tabs[2]:
        st.subheader("Capture Face Data")
        conn = sqlite3.connect(db_path)
        students = pd.read_sql_query("SELECT id, name FROM students ORDER BY name", conn)
        conn.close()
        
        if not students.empty:
            selected_student = st.selectbox("Select Student", students['name'].tolist())
            student_id = students.loc[students['name'] == selected_student, 'id'].values[0]
            
            if st.button("Start Capture"):
                capture_face_data(student_id, selected_student)
        else:
            st.info("No students registered yet")
    
    with tabs[3]:
        st.subheader("Train Recognition Model")
        if st.button("Train Model", key="train_model_btn"):
            train_model()

# Delete a student
def delete_student(student_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if student exists
    cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    result = cursor.fetchone()
    
    if result:
        student_name = result[0]
        
        # Delete attendance records
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        
        # Delete student
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        
        conn.commit()
        
        # Delete face data directory
        student_dir = os.path.join(dataset_path, f"student_{student_id}")
        if os.path.exists(student_dir):
            for file in os.listdir(student_dir):
                os.remove(os.path.join(student_dir, file))
            os.rmdir(student_dir)
            
        st.success(f"Student {student_name} (ID: {student_id}) deleted successfully")
    else:
        st.error(f"No student found with ID: {student_id}")
        
    conn.close()

# Attendance Records Page
def attendance_records():
    st.title("Attendance Records")
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        filter_date = st.date_input("Select Date", datetime.now())
    with col2:
        check_type = st.selectbox("Check Type", ["All", "Check In", "Check Out"])
    
    # Convert date to string format
    date_str = filter_date.strftime("%Y-%m-%d")
    
    # Build query based on filters
    query = """
        SELECT s.name, s.roll, s.department, a.date, a.time, a.check_type, a.location
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
    """
    params = [date_str]
    
    if check_type != "All":
        check_type_value = "in" if check_type == "Check In" else "out"
        query += " AND a.check_type = ?"
        params.append(check_type_value)
        
    query += " ORDER BY a.time"
    
    # Get attendance records
    conn = sqlite3.connect(db_path)
    attendance_df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Display attendance summary
    if not attendance_df.empty:
        # Format the check_type column
        attendance_df['check_type'] = attendance_df['check_type'].apply(lambda x: "Check In" if x == "in" else "Check Out")
        
        # Display summary stats
        total_records = len(attendance_df)
        unique_students = attendance_df['name'].nunique()
        
        col1, col2 = st.columns(2)
        col1.metric("Total Records", total_records)
        col2.metric("Unique Students", unique_students)
        
        # Display full data
        st.subheader("Attendance Details")
        st.dataframe(attendance_df, use_container_width=True)
        
        # Export options
        if st.button("Export to CSV"):
            csv = attendance_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="attendance_{date_str}.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info(f"No attendance records found for {date_str}")

# Timetable Page
def timetable_page():
    st.title("Class Timetable")
    
    # Check if there are courses in the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM courses")
    course_count = cursor.fetchone()[0]
    
    if course_count == 0:
        # Insert sample courses and timetable if none exist
        st.info("No courses found. Adding sample timetable...")
        
        # Sample courses
        courses = [
            ("Mathematics", "MATH101", "Dr. Smith"),
            ("Physics", "PHYS101", "Prof. Johnson"),
            ("Computer Science", "CS101", "Dr. Williams"),
            ("English", "ENG101", "Ms. Brown"),
            ("Chemistry", "CHEM101", "Dr. Davis")
        ]
        
        cursor.executemany("INSERT INTO courses (name, code, instructor) VALUES (?, ?, ?)", courses)
        
        # Get course IDs
        cursor.execute("SELECT id, name FROM courses")
        course_map = {name: id for id, name in cursor.fetchall()}
        
        # Sample timetable
        timetable_entries = [
            (course_map["Mathematics"], "Monday", "09:00", "10:30", "Room 101"),
            (course_map["Physics"], "Monday", "11:00", "12:30", "Room 102"),
            (course_map["Computer Science"], "Monday", "14:00", "15:30", "Lab 201"),
            (course_map["Chemistry"], "Tuesday", "09:00", "10:30", "Room 103"),
            (course_map["English"], "Tuesday", "11:00", "12:30", "Room 104"),
            (course_map["Mathematics"], "Wednesday", "09:00", "10:30", "Room 101"),
            (course_map["Physics"], "Wednesday", "14:00", "15:30", "Lab 202"),
            (course_map["Computer Science"], "Thursday", "11:00", "12:30", "Lab 201"),
            (course_map["English"], "Thursday", "14:00", "15:30", "Room 104"),
            (course_map["Chemistry"], "Friday", "09:00", "10:30", "Room 103")
        ]
        
        cursor.executemany("""
            INSERT INTO timetable (course_id, day, start_time, end_time, room) 
            VALUES (?, ?, ?, ?, ?)
        """, timetable_entries)
        
        conn.commit()
    
    # Get timetable data
    timetable_query = """
        SELECT 
            t.day, 
            c.name AS course_name, 
            c.code AS course_code,
            c.instructor,
            t.start_time, 
            t.end_time, 
            t.room
        FROM timetable t
        JOIN courses c ON t.course_id = c.id
        ORDER BY 
            CASE 
                WHEN t.day = 'Monday' THEN 1
                WHEN t.day = 'Tuesday' THEN 2
                WHEN t.day = 'Wednesday' THEN 3
                WHEN t.day = 'Thursday' THEN 4
                WHEN t.day = 'Friday' THEN 5
                ELSE 6
            END,
            t.start_time
    """
    
    timetable_df = pd.read_sql_query(timetable_query, conn)
    conn.close()
    
    # Format timetable for display
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    # Highlight today's schedule
    today = datetime.now().strftime('%A')
    if today in days:
        st.subheader(f"Today's Schedule ({today})")
        today_df = timetable_df[timetable_df['day'] == today]
        
        if not today_df.empty:
            for _, row in today_df.iterrows():
                with st.expander(f"{row['start_time']} - {row['end_time']}: {row['course_name']} ({row['course_code']})"):
                    st.write(f"**Instructor:** {row['instructor']}")
                    st.write(f"**Room:** {row['room']}")
        else:
            st.info("No classes scheduled for today")
    
    # Full week timetable
    st.subheader("Weekly Schedule")
    
    for day in days:
        with st.expander(day):
            day_df = timetable_df[timetable_df['day'] == day]
            
            if not day_df.empty:
                for _, row in day_df.iterrows():
                    st.markdown(f"""
                    **{row['start_time']} - {row['end_time']}**: {row['course_name']} ({row['course_code']})  
                    *Instructor:* {row['instructor']}  
                    *Room:* {row['room']}
                    """)
                    st.markdown("---")
            else:
                st.info(f"No classes scheduled for {day}")

# Teachers Page
def teachers_page():
    st.title("Teachers Information")
    
    # Get teachers from courses table
    conn = sqlite3.connect(db_path)
    teachers_query = """
        SELECT DISTINCT instructor, 
            (SELECT GROUP_CONCAT(name, ', ') FROM courses WHERE instructor = c.instructor) AS courses
        FROM courses c
        ORDER BY instructor
    """
    
    teachers_df = pd.read_sql_query(teachers_query, conn)
    conn.close()
    
    if not teachers_df.empty:
        # Display teachers
        for _, row in teachers_df.iterrows():
            with st.expander(row['instructor']):
                st.write(f"**Courses:** {row['courses']}")
                
                # Add sample contact information
                name_parts = row['instructor'].replace("Dr. ", "").replace("Prof. ", "").split()
                email = f"{name_parts[-1].lower()}@university.edu"
                
                st.write(f"**Email:** {email}")
                st.write("**Office Hours:** Monday & Wednesday, 2:00 PM - 4:00 PM")
                st.write("**Office Location:** Faculty Building, Room 304")
                
                # Contact form placeholder
                st.subheader("Quick Contact")
                with st.form(f"contact_{row['instructor'].replace(' ', '_')}"):
                    message = st.text_area("Message")
                    submit = st.form_submit_button("Send Message")
                
                if submit:
                    st.success(f"Message sent to {row['instructor']}!")
    else:
        st.info("No teachers found in the system")

# Attendance Statistics Page
def attendance_statistics():
    st.title("Attendance Statistics")
    
    # Period selection
    period = st.radio("Select Period", ["Today", "This Week", "This Month", "Custom"])
    
    # Get date range based on period
    today = datetime.now().date()
    if period == "Today":
        start_date = today
        end_date = today
    elif period == "This Week":
        # Start from Monday of current week
        start_date = today - pd.Timedelta(days=today.weekday())
        end_date = today
    elif period == "This Month":
        start_date = today.replace(day=1)
        end_date = today
    else:  # Custom
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", today - pd.Timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", today)
            
    # Convert to string format for SQL
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get attendance data
    attendance_query = """
        SELECT 
            s.name, 
            s.roll,
            s.department,
            COUNT(DISTINCT a.date) as days_present,
            (SELECT COUNT(DISTINCT date) FROM attendance 
             WHERE date BETWEEN ? AND ?) as total_days,
            ROUND(COUNT(DISTINCT a.date) * 100.0 / 
                  (SELECT COUNT(DISTINCT date) FROM attendance 
                   WHERE date BETWEEN ? AND ?), 2) as attendance_percentage
        FROM 
            students s
        LEFT JOIN 
            attendance a ON s.id = a.student_id AND a.date BETWEEN ? AND ? AND a.check_type = 'in'
        GROUP BY 
            s.id
        ORDER BY 
            attendance_percentage DESC
    """
    
    attendance_df = pd.read_sql_query(
        attendance_query, 
        conn, 
        params=[start_date_str, end_date_str, start_date_str, end_date_str, start_date_str, end_date_str]
    )
    
    # Get daily attendance counts
    daily_query = """
        SELECT 
            a.date, 
            COUNT(DISTINCT a.student_id) as student_count,
            (SELECT COUNT(*) FROM students) as total_students,
            ROUND(COUNT(DISTINCT a.student_id) * 100.0 / (SELECT COUNT(*) FROM students), 2) as percentage
        FROM 
            attendance a
        WHERE 
            a.date BETWEEN ? AND ? AND a.check_type = 'in'
        GROUP BY 
            a.date
        ORDER BY 
            a.date
    """
    
    daily_df = pd.read_sql_query(daily_query, conn, params=[start_date_str, end_date_str])
    
    conn.close()
    
    # Display attendance percentage by student
    st.subheader("Student Attendance Percentages")
    
    if not attendance_df.empty:
        # Add color coding based on attendance percentage
        def color_percentage(val):
            if val < 75:
                return 'background-color: #ffcccc'
            elif val > 90:
                return 'background-color: #ccffcc'
            else:
                return ''
            
        # Format the DataFrame for display
        styled_df = attendance_df.style.applymap(
            color_percentage, 
            subset=['attendance_percentage']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Create attendance chart
        st.subheader("Attendance Distribution")
        fig, ax = plt.subplots()
        bins = [0, 25, 50, 75, 90, 100]
        labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-100%']
        attendance_df['attendance_range'] = pd.cut(attendance_df['attendance_percentage'], bins=bins, labels=labels, right=True)
        attendance_counts = attendance_df['attendance_range'].value_counts().sort_index()
        
        ax.bar(attendance_counts.index, attendance_counts.values, color='skyblue')
        ax.set_xlabel('Attendance Range')
        ax.set_ylabel('Number of Students')
        ax.set_title('Distribution of Student Attendance')
        
        st.pyplot(fig)
        
        # Daily attendance chart
        if not daily_df.empty:
            st.subheader("Daily Attendance Trend")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(daily_df['date'], daily_df['percentage'], marker='o', linestyle='-', color='blue')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Attendance Percentage')
            ax2.set_title('Daily Attendance Percentage')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add target line at 75%
            ax2.axhline(y=75, color='r', linestyle='--', alpha=0.7)
            ax2.text(daily_df['date'].iloc[0], 76, 'Target (75%)', color='r')
            
            # Rotate date labels for better visibility
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig2)
    else:
        st.info(f"No attendance data available for the selected period ({start_date_str} to {end_date_str})")

# Settings Page
def settings_page():
    st.title("System Settings")
    
    tabs = st.tabs(["General Settings", "Database", "About"])
    
    with tabs[0]:
        st.subheader("General Settings")
        
        # Attendance thresholds
        st.write("**Attendance Thresholds**")
        min_attendance = st.slider("Minimum Required Attendance (%)", 0, 100, 75)
        st.info(f"Students below {min_attendance}% attendance will be marked as at risk")
        
        # Face recognition settings
        st.write("**Face Recognition Settings**")
        confidence_threshold = st.slider("Recognition Confidence Threshold", 50, 100, 80)
        st.info(f"Lower values are more lenient but may cause false positives")
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
            
    with tabs[1]:
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Backup Database"):
                # Code for database backup
                backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"attendance_backup_{backup_time}.db"
                
                # Simple file copy as backup
                import shutil
                try:
                    shutil.copy2(db_path, os.path.join(BASE_DIR, backup_file))
                    st.success(f"Database backed up successfully to {backup_file}")
                except Exception as e:
                    st.error(f"Backup failed: {str(e)}")
        
        with col2:
            if st.button("Reset Database", help="Warning: This will delete all data!"):
                # Add a confirmation
                st.warning("⚠️ This will delete all attendance records, students, and courses!")
                confirm = st.checkbox("I understand and want to proceed")
                
                if confirm and st.button("Confirm Reset"):
                    # Simply delete the file and recreate empty tables
                    try:
                        if os.path.exists(db_path):
                            os.remove(db_path)
                        setup_database()
                        st.success("Database reset successfully")
                    except Exception as e:
                        st.error(f"Reset failed: {str(e)}")
    
    with tabs[2]:
        st.subheader("About the System")
        
        st.markdown("""
        ### Smart Attendance System
        
        **Version:** 1.0.0
        
        This system is designed to automate attendance tracking using facial recognition technology.
        
        **Features:**
        - Face recognition for attendance marking
        - Student management
        - Attendance tracking and statistics
        - Timetable integration
        - Location tracking
        
        **Technologies Used:**
        - Python
        - Streamlit
        - OpenCV
        - MTCNN
        - SQLite
        - Pandas
        - Matplotlib
        
        **Developer:** Your Name
        """)

# Main function
def main():
    # Setup database if it doesn't exist
    setup_database()
    
    # Sidebar menu for navigation
    menu = sidebar_menu()
    
    # Display the selected page
    if menu == "Home":
        home_page()
    elif menu == "Student Management":
        student_management()
    elif menu == "Attendance Records":
        attendance_records()
    elif menu == "Timetable":
        timetable_page()
    elif menu == "Teachers":
        teachers_page()
    elif menu == "Attendance Statistics":
        attendance_statistics()
    elif menu == "Settings":
        settings_page()

# Run the application
if __name__ == "__main__":
    main()
        