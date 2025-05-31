# 🏋️‍♂️ Exercise Monitoring System

The **Exercise Monitoring System** is an AI-powered web application built with **Streamlit**, **MongoDB**, and **MediaPipe**. Designed for fitness trainers and trainees, this system enables remote workout monitoring through real-time **pose detection**, group management, and task tracking.

Whether it's **Push-ups**, **Curls**, or other exercises, the system ensures accurate monitoring using camera-based posture recognition. Trainers can assign tasks to multiple groups, while trainees can perform exercises with live posture validation.

---

## 🛠️ Tools & Technologies Used

| Technology       | Purpose                                            |
|------------------|----------------------------------------------------|
| **Streamlit**    | Rapid UI development for Python web apps          |
| **MongoDB**      | NoSQL database to manage user, task, and group data|
| **MediaPipe**    | Real-time pose detection and body movement analysis|
| **PyCam / OpenCV**| Webcam integration and camera capture             |
| **Python**       | Core programming language for backend and logic    |

---

## 🔑 Roles & Features

### 👤 **User Roles**

- **Trainer**
- **Trainee**

### 🧑‍🏫 **Trainer Functionalities**
- Register and log in as trainer
- Create and manage workout **groups**
- Assign tasks like **Push-ups**, **Curls**, etc.
- Track trainees’ task completions
- View group-wise performance

### 🧑‍💼 **Trainee Functionalities**
- Register and log in as trainee
- Join existing groups
- View and perform assigned tasks
- Start task to activate webcam and record posture
- Submit performance for trainer review

### 🔐 **Authentication & User Info**
- All user credentials stored securely in MongoDB
- Role-based login flow
- Option to **change password**
- Separate dashboards for trainers and trainees

---

## 🧱 MongoDB Database Schema

| Collection Name | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `user_info`      | Stores all registered users (ID, password, role)                  |
| `trainer`        | Stores trainer-specific info such as groups, assigned tasks       |
| `trainee`        | Stores trainee info including group ID and task progress          |
| `groups`         | Contains group names, member list, and assigned task info         |

---

## 🖼️ Application Overview

| Page Name          | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `Register`         | Sign-up page for both trainers and trainees                  |
| `Login`            | Login portal with secure authentication                      |
| `Dashboard`        | Shows role-specific features and profile                     |
| `Trainer Panel`    | Tools for group creation and task assignment                 |
| `Trainee Panel`    | View assigned tasks and perform them with webcam detection   |
| `Change Password`  | Update account password securely                             |

---

## 🧪 Real-Time Exercise Tracking

- Activates webcam when a task is started
- Uses **MediaPipe** to detect human body pose
- Records and verifies the correctness of posture
- Stores completion status in MongoDB

---

## 📷 Screenshots

### 🌐 Website Screenshots 

