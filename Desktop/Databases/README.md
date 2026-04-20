# MediTrack AI / QuickBite Healthcare Analytics

A Streamlit-based medical operations dashboard and appointment booking app with an AI chatbot overlay.

## Project Overview

This repository includes a healthcare analytics workspace built with Python and Streamlit. It supports:
- interactive analytics dashboards
- booking appointments
- clinic and doctor performance insights
- a floating AI chatbot for in-app guidance

## Main files

- `quickbite/quickapp.py` — main Streamlit application and UI logic
- `quickbite/meditrackdata.py` — SQLite data access, queries, and helper functions
- `app.py` — alternate application/script entry point
- `data.py` — data utility helpers
- `generate_data.py` — sample data generation utilities
- `quickbite.db`, `swiftride.db` — local SQLite databases (ignored from Git)

## Setup

1. Create and activate a virtual environment:
   ```bash
   cd /home/khan/Desktop/Databases
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   cd quickbite
   streamlit run quickapp.py
   ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

- Use sidebar filters to change city, department, and date range.
- Open the chatbot via the floating 💬 button.
- Type messages into the chatbot and press Send.
- In the Book Appointment section, enter an existing patient name exactly or add a new patient in Manage Records.
- Confirm appointments and review the booked appointments table.

## GitHub setup

To save this project to GitHub with all files and documentation:

```bash
cd /home/khan/Desktop/Databases
git add .
git commit -m "Add MediTrack AI project, README, and Git ignore rules"
git push origin main
```

If your main branch is different, replace `main` with your branch name.

## Notes

- Do not commit `venv/` or local database files.
- If you need a full database reset, delete `quickbite.db` and `swiftride.db`, then recreate as needed.
