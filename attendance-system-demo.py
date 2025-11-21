# Smart Attendance System Demo
# Basic implementation combining MTCNN for detection and LBPH for recognition

import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import time
from mtcnn import MTCNN
import tensorflow as tf

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'attendance.db')
dataset_path = os.path.join(BASE_DIR, 'dataset')
model_path = os.path.join(BASE_DIR, 'trainer/trainer.yml')

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, 'dataset'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'trainer'), exist_ok=True)

# Initialize face detector
detector = MTCNN()

# Database setup
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
        FOREIGN KEY (student_id) REFERENCES students(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Check if the camera is working properly
def check_camera():
    print("\n[INFO] Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read camera frame")
        cap.release()
        return False
    
    print("[INFO] Camera is working properly")
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(2000)  # Display frame for 2 seconds
    cv2.destroyAllWindows()
    cap.release()
    return True

# Register a new student
def register_student():
    name = input("Enter Name: ")
    roll = input("Enter Roll Number: ")
    dept = input("Enter Department: ")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO students (name, roll, department) VALUES (?, ?, ?)", 
                      (name, roll, dept))
        student_id = cursor.lastrowid
        conn.commit()
        print(f"\n[INFO] Student registered with ID: {student_id}")
        
        capture_face_data(student_id, name)
        
    except sqlite3.IntegrityError:
        print("\n[ERROR] Roll number already exists!")
    finally:
        conn.close()

# Capture facial data for training
def capture_face_data(student_id, name):
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")
    
    # Create directory for this person
    student_dir = os.path.join(dataset_path, f"student_{student_id}")
    os.makedirs(student_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    max_images = 30
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(frame)
        
        for face in faces:
            if face['confidence'] < 0.9:  # Filter low confidence detections
                continue
                
            x, y, w, h = face['box']
            # Ensure positive values
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            
            # Save the captured face
            img_name = os.path.join(student_dir, f"student_{student_id}_{count}.jpg")
            cv2.imwrite(img_name, face_img)
            
            # Display the frame with rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Images Captured: {count+1}/{max_images}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            count += 1
        
        cv2.imshow("Face Capture", frame)
        k = cv2.waitKey(100)
        if k == 27:  # ESC key to stop
            break
            
    print(f"\n[INFO] {count} face samples collected")
    cap.release()
    cv2.destroyAllWindows()

# Train the face recognition model
def train_model():
    print("\n[INFO] Training faces. This may take a few minutes...")
    faces = []
    ids = []
    
    # Walk through all the face samples in dataset directory
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                
                # Extract student_id from directory name
                student_id = int(os.path.basename(root).split("_")[1])
                
                # Read the image and convert to grayscale
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
                # Add face and ID to arrays
                faces.append(img)
                ids.append(student_id)
    
    if not faces or not ids:
        print("[ERROR] No face samples found. Please register students first.")
        return False
    
    # Create and train the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    # Save the model
    recognizer.write(model_path)
    print(f"\n[INFO] Model trained and saved at {model_path}")
    return True

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

# Check if attendance was already marked today
def already_marked_today(student_id):
    today = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", 
                  (student_id, today))
    result = cursor.fetchone()
    conn.close()
    
    return result is not None

# Mark attendance in the database
def mark_attendance(student_id):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)", 
                  (student_id, date_string, time_string))
    conn.commit()
    conn.close()

# Start face recognition and attendance marking
def start_recognition():
    print("\n[INFO] Starting face recognition. Press 'q' to quit...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("[ERROR] Model file not found. Please train the model first.")
        return
    
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    cap = cv2.VideoCapture(0)
    marked_students = set()  # To avoid duplicate notifications
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(frame)
        
        for face in faces:
            if face['confidence'] < 0.9:  # Filter low confidence detections
                continue
                
            x, y, w, h = face['box']
            # Ensure positive values
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            
            try:
                # Recognize the face
                student_id, confidence = recognizer.predict(face_img)
                
                # Lower confidence means better match in LBPH
                if confidence < 70:  # Threshold for good recognition
                    student_name = get_student_name(student_id)
                    confidence_display = f"{100 - min(confidence, 100):.1f}%"
                    color = (0, 255, 0)  # Green
                    
                    # Mark attendance if not already marked
                    if student_id not in marked_students and not already_marked_today(student_id):
                        mark_attendance(student_id)
                        marked_students.add(student_id)
                        print(f"[INFO] Attendance marked for {student_name}")
                else:
                    student_name = "Unknown"
                    confidence_display = ""
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle around face and display name
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, student_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                if confidence_display:
                    cv2.putText(frame, confidence_display, (x, y+h+30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            except Exception as e:
                print(f"[ERROR] Recognition error: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Display the date and time
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        cv2.putText(frame, f"Date: {date_string}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Time: {time_string}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Generate attendance report
def generate_report():
    date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.name, s.roll, s.department, a.time 
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time
    """, (date,))
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print(f"\n[INFO] No attendance records found for {date}")
        return
    
    print(f"\n--- Attendance Report for {date} ---")
    print(f"Total present: {len(records)}")
    print("\nName\t\tRoll\t\tDepartment\t\tTime")
    print("-" * 70)
    
    for record in records:
        name, roll, dept, time = record
        print(f"{name}\t\t{roll}\t\t{dept}\t\t{time}")

# Main menu
def main_menu():
    setup_database()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n===== Smart Attendance System =====")
        print("1. Check Camera")
        print("2. Register New Student")
        print("3. Train Model")
        print("4. Start Attendance")
        print("5. Generate Report")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            check_camera()
        elif choice == '2':
            register_student()
        elif choice == '3':
            train_model()
        elif choice == '4':
            start_recognition()
        elif choice == '5':
            generate_report()
        elif choice == '0':
            print("\nThank you for using Smart Attendance System!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu()
