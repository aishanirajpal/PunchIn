# PunchIn

Smart Attendance System built with Streamlit, OpenCV, TensorFlow, and SQLite. The web UI lets students register, capture training samples, and mark check-in/out via photo uploads right from the browser—so it works both locally and on Streamlit Cloud.

## Features
- Browser-based face capture using `st.camera_input`
- LBPH facial recognition with configurable confidence threshold
- Student registration and dataset management
- Attendance logging with geolocation and stats dashboards
- Timetable and teacher info pages backed by SQLite

## Running Locally
1. **Install Python 3.11** (matches the Streamlit Cloud runtime).
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Open the provided URL in your browser. The site will request camera access for attendance and data capture.

## Deploying on Streamlit Cloud
Streamlit Cloud needs both Python wheels and system libraries:

- `runtime.txt` pins the runtime to `python-3.11.9` so OpenCV wheels install.
- `packages.txt` lists `libgl1` to satisfy OpenCV’s `libGL.so.1` dependency.
- `requirements.txt` already contains the Python packages.

Commit/push those files to your repo, then deploy via Streamlit Cloud’s dashboard. The build log should show the runtime pin, apt install of `libgl1`, and pip install of the requirements.

## Camera Behavior
- **Browser capture**: `st.camera_input` handles camera access, so no direct `cv2.VideoCapture` calls are needed. This works locally and in the cloud.
- **Permissions**: Users must allow camera access in their browser; otherwise attendance capture will prompt again.

## Training the Model
1. Navigate to `Student Management → Register Student` to add a student record.
2. Use `Capture Face Data` to collect 30 samples (progress shown in UI).
3. Press `Train Model` to retrain LBPH and store the model at `trainer/trainer.yml`.

## Project Layout
```
.
├── app.py                # Streamlit application
├── attendance.db         # SQLite database (auto-created)
├── dataset/              # Captured face samples per student
├── trainer/trainer.yml   # LBPH face recognizer model
├── requirements.txt      # Python dependencies
├── runtime.txt           # Streamlit Cloud Python version
├── packages.txt          # System packages for deployment
└── README.md
```

## Notes
- `.gitignore` excludes large datasets and local DB files; adjust if you need to track them.
- Streamlit warnings about deprecated parameters (e.g., `use_column_width`) have been addressed; keep dependencies updated to stay compatible.
