import streamlit as st
from pymongo import MongoClient
import datetime
import time
import secrets
import smtplib
from email.mime.text import MIMEText
from pymongo.errors import DuplicateKeyError
import random
import string
import os
import re
import json
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from bson.binary import Binary
import base64
from login import main as login_main
import pandas as pd
import functools
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import calendar
import matplotlib.pyplot as plt
from datetime import  timedelta

# Streamlit configuration
st.set_page_config(
    page_title="EXERCISE MONITORING SYSTEM",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="auto",
)


# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["EMS"]
collection = db["user_info"]
trainers_col = db["trainer"]
trainees_col = db["trainee"]
groups_col = db["groups"]


# Custom CSS
custom_css = """
<style>
    .center-heading {
            background-color: royalblue;
            color: black;
            padding: 10px;
            border: 2px solid white;
            text-align: center;
            color: black;
            margin-top: 2px;
            border-radius: 10px;
           
        }
        .center-main-heading{
            background-color: gray;
            color: black;
            padding: 10px;
            border: 2px solid white;
            text-align: center;
            color: black;
            margin-top: 2px;
            border-radius: 10px;
        }
       
        .user-id{
            background-color: teal;
            color: black;
            padding: 10px;
            border: 2px solid white;
            text-align: center;
            color: black;
            margin-top: 2px;
            border-radius: 10px;
        }
        .mail-id{
            color: firebrick;
            text-align: center;
            border-radius: 10px;
           
        }
       
</style>
"""

profile_img = Image.open("profile.png")
def home():
    with st.container(border = True):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)
           
        # Exercise data
        exercises = {
            "Curls": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/03/hammercurl-1456956209.gif",
                "instructions": [
                    "Stand with your feet hip-width apart, holding a dumbbell in each hand.",
                    "Keep your back straight and shoulders relaxed.",
                    "With palms facing forward, bend your elbows and curl the weights toward your shoulders.",
                    "Squeeze your biceps at the top of the movement.",
                    "Slowly lower the weights back to the starting position.",
                ],
                "tips": [
                    "Maintain proper form throughout the exercise.",
                    "Avoid swinging your body to lift the weights.",
                    "Control the movement and focus on the bicep contraction.",
                    "Choose a weight that challenges you without sacrificing form.",
                ],
            },
            "V-Ups": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/08/vupfull-1472154765.gif",
                "instructions": [
                    "Lie on your back with your arms extended overhead and legs straight.",
                    "Engage your core muscles and lift your legs and upper body off the ground simultaneously.",
                    "Reach toward your toes with your hands, forming a V-shape with your body.",
                    "Pause at the top of the movement, feeling the contraction in your abs.",
                    "Slowly lower your legs and upper body back to the starting position.",
                ],
                "tips": [
                    "Focus on using your abdominal muscles to lift, not just momentum.",
                    "Keep your legs straight and avoid bending at the knees.",
                    "Exhale as you lift your legs and upper body for better muscle engagement.",
                    "Modify the movement if needed, bending your knees slightly to reduce intensity.",
                ],
                },
            "Push-ups": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/pushup-1462808858.gif",
                "instructions": [
                    "Start in a plank position with your hands placed slightly wider than shoulder-width apart.",
                    "Keep your body in a straight line from head to heels.",
                    "Lower your chest towards the ground by bending your elbows.",
                    "Ensure your elbows are at a 90-degree angle or smaller.",
                    "Push through your palms to return to the starting position.",
                ],
                "tips": [
                    "Maintain a tight core to prevent your lower back from sagging.",
                    "Engage your chest, triceps, and shoulders during the movement.",
                    "Keep your neck in a neutral position to avoid straining it.",
                    "Modify the exercise by performing it on your knees if needed.",
                ],
                },
            "High Knees": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/03/highkneerun-1457044203.gif",
                "instructions": [
                    "Start with left knee first, to start the count."
                    "Stand with your feet hip-width apart.",
                    "Lift your right knee as high as possible, and quickly switch to lift your left knee.",
                    "Continue alternating knees in a running motion.",
                    "Pump your arms to mimic a running motion.",
                    "Keep your core engaged throughout the exercise.",
                ],
                "tips": [
                    "Maintain a brisk and controlled pace.",
                    "Land softly on the balls of your feet.",
                    "Focus on lifting your knees toward your chest.",
                    "Keep your upper body upright and avoid leaning forward.",
                ],
                },
            "Squats": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/03/bodyweightsquat-1457041691.gif",
                "instructions": [
                    "Stand with your feet shoulder-width apart.",
                    "Keep your back straight and shoulders relaxed.",
                    "Bend your knees and lower your body, keeping your chest up.",
                    "Lower yourself until your thighs are parallel to the ground.",
                    "Push through your heels to return to the starting position.",
                ],
                "tips": [
                    "Ensure proper alignment of your knees with your toes.",
                    "Engage your core muscles throughout the movement.",
                    "Maintain a controlled and steady pace.",
                ],
            },
            "Forward Extension": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/03/frontraise-1456955633.gif?resize=640:*",
                "instructions": [
                    "Stand with your feet shoulder-width apart, holding a dumbbell in each hand.",
                    "Maintain a slight bend in your elbows and keep your back straight.",
                    "Raise the weights directly in front of you, keeping your arms parallel to the ground.",
                    "Lift until your arms are at shoulder height, or slightly below.",
                    "Slowly lower the weights back to the starting position.",
                ],
                "tips": [
                    "Engage your core muscles to stabilize your body.",
                    "Keep a controlled pace throughout the exercise.",
                    "Avoid using momentum to lift the weights.",
                    "Choose a weight that challenges you without compromising your form.",
                ],
            },
            "Jumping Jacks": {
                "gif": "https://hips.hearstapps.com/hmg-prod/images/workouts/2016/03/jumpingjack-1457045563.gif",
                "instructions": [
                    "Start with your feet together and arms at your sides.",
                    "Jump and spread your legs while raising your arms overhead.",
                    "Land with your feet shoulder-width apart and arms back at your sides.",
                    "Repeat the motion continuously with a brisk pace.",
                ],
                "tips": [
                    "Maintain a steady and controlled rhythm throughout.",
                    "Engage your core muscles to stabilize your body.",
                    "Land softly to reduce impact on your joints.",
                    "Keep your knees slightly bent during the exercise.",
                ],
            },
            "Forward Bend": {
                "gif": "https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2021/08/400x400_How_to_Fix_a_Forward_Head_Posture_Standing_Forward_Bend.gif",
                "instructions": [
                    "Stand with your feet hip-width apart and your knees slightly bent.",
                    "Engage your core muscles to support your lower back.",
                    "Slowly hinge at your hips and bend forward at the waist.",
                    "Let your arms hang down toward the floor or reach for your toes.",
                    "Feel a stretch in your hamstrings and lower back.",
                ],
                "tips": [
                    "Keep a slight bend in your knees to avoid hyperextension.",
                    "Maintain a straight spine throughout the movement.",
                    "Breathe deeply and relax into the stretch.",
                    "Don't force yourself into the stretch; go only as far as comfortable.",
                ],
            }
            # Repeat the structure for other exercises
        }

        # Title and Introduction
        st.title("Exercise Demonstration")
        

        # Collapsible sections for each exercise
        for exercise_name, exercise_info in exercises.items():
            with st.expander(exercise_name):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Textual Instructions
                    st.markdown("### Instructions:")
                    for instruction in exercise_info["instructions"]:
                        st.write(instruction)

                    # Additional Tips
                    st.markdown("### Tips:")
                    for tip in exercise_info["tips"]:
                        st.write(tip)

                with col2:
                    # Display the GIF on the right
                    st.image(exercise_info["gif"], caption=f"{exercise_name} Demonstration", use_column_width=True, width=300)
                    
                    
# function to calculate angle
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
   
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)
   
    if angle>180.0:
        angle = 360-angle
       
    return angle


def start_exercise(exercise_name, count, given):
    mp_drawing = mp.solutions.drawing_utils   #visualizing poses
    mp_pose = mp.solutions.pose     #importing pose estimation model
    if exercise_name == "Curl":
        cap = cv2.VideoCapture(0)

        # Curl counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                   
                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                    # Get coordinates for right arm
                    shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                   
                    # Calculate angle
                    angle1 = calculate_angle(shoulder1, elbow1, wrist1)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle1),
                                tuple(np.multiply(elbow1, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                    # Curl counter logic
                    if angle > 160 and angle1 > 160:
                        stage = "down"
                    if angle < 50 and angle1 < 50 and stage =='down':
                        stage="up"
                        counter +=1
                        # print(counter)
                           
                except:
                    pass
               
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (300,100), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (120,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (120,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "Forward-extensions":
        cap = cv2.VideoCapture(0)

        # V-UP counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates
                    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    i=0
                    # Calculate angle for left leg
                    angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                    # Visualize angle
                    elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)
                   
                    # Get coordinates for right leg
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    elbow_angle_r = calculate_angle(wrist_r, elbow_r, shoulder_r)

                    # Calculate angle
                    angle_r = calculate_angle(elbow_r, shoulder_r, hip_r)
                    # Visualize angle
                   
                    cv2.putText(image, str(elbow_angle_r),
                                tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    # V-UP counter logic
                    if (angle_l > 80 and angle_r > 80) and (elbow_angle_l > 120 and elbow_angle_r > 120):
                        stage = "up"
                    if (angle_l < 30 and angle_r < 30) and stage =='up':
                        stage = "down"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render High knee counter
                # Setup status box
                cv2.rectangle(image, (0,0), (350,73), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "V-up" or exercise_name == "Forward-bend":
        cap = cv2.VideoCapture(0)

        # V-UP counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates
                    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                   
                    i=0
                    # Calculate angle for left leg
                    high_angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                    low_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
                   
                    # Get coordinates for right leg
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    # Calculate angle
                    high_angle_r = calculate_angle(elbow_r, shoulder_r, hip_r)
                    low_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
                    # Visualize angle
                    cv2.putText(image, str(low_angle_r),
                                tuple(np.multiply(hip_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                    # V-UP counter logic
                    if (high_angle_l > 140 or high_angle_r > 140) and (low_angle_l > 160 or low_angle_r > 160):
                        stage = "up"
                    if  stage =='up' and (low_angle_l < 100 or low_angle_r < 100):
                        stage = "down"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render High knee counter
                # Setup status box
                cv2.rectangle(image, (0,0), (350,73), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "High-knees":
        cap = cv2.VideoCapture(0)

        # High Knee counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                   
                    i=0
                    # Calculate angle for left leg
                    angle_l = calculate_angle(hip_l, knee_l, ankle_l)
                    #print("left ",i, " ",  angle_l)
                   
                    high_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_l),
                                tuple(np.multiply(knee_l, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                    # Get coordinates for right leg
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    # Calculate angle
                    angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                    #print("right ",i, " ",  angle_r)
                    high_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
                    # Visualize angle
                    cv2.putText(image, str(angle_r),
                                tuple(np.multiply(knee_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                    # High Knee counter logic
                    if (angle_l < 100 and angle_r > 170) and (high_angle_l < 100 and high_angle_r > 170):
                        stage = "left up"
                    if angle_l > 170 and angle_r < 100 and stage =='left up' and (high_angle_r < 100 and high_angle_l > 170):
                        stage = "right up"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render High knee counter
                # Setup status box
                cv2.rectangle(image, (0,0), (350,73), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "Squat":
        cap = cv2.VideoCapture(0)
        counter = count
        stage = None
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    i=0
                    # Calculate angle for left leg
                    angle_l = calculate_angle(hip_l, knee_l, ankle_l)
                    #print("left ",i, " ",  angle_l)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_l),
                                tuple(np.multiply(knee_l, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                    # Get coordinates for right leg
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                   
                    # Calculate angle
                    angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                    #print("right ",i, " ",  angle_r)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_r),
                                tuple(np.multiply(knee_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                    # Curl counter logic
                    if angle_l < 80 and angle_r < 80:
                        stage = "down"
                    if angle_l >160 and angle_r > 160 and stage =='down':
                        stage = "up"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "Jumping-jack":
        cap = cv2.VideoCapture(0)

        # Curl counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # angles at shoulders
                   
                    # left shoulder
                    # Get coordinates
                    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                   
                    # Calculate angle at left shoulder
                    angle_shoulder_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_shoulder_l),
                                tuple(np.multiply(shoulder_l, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
               
                    # right shoulder
                    # Get coordinates
                    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                   
                    # Calculate angle
                    angle_shoulder_r = calculate_angle(elbow_r, shoulder_r, hip_r)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_shoulder_r),
                                tuple(np.multiply(shoulder_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                   
                   
                    # angles at hips
                   
                    # left hip
                    # Get coordinates
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]            
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        #             ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                   
                    # Calculate angle at left hip
                    angle_hip_l = calculate_angle(hip_r, hip_l, knee_l)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_hip_l),
                                tuple(np.multiply(hip_l, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                    # right hip
                    # Get coordinates
                    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]            
                    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        #             ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                   
                    # Calculate angle
                    angle_hip_r = calculate_angle(hip_l, hip_r, knee_r)
                    #print("right ",i, " ",  angle_r)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_hip_r),
                                tuple(np.multiply(hip_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                   
                    # Curl counter logic
                    if angle_shoulder_r > 90 and angle_shoulder_l > 90 and angle_hip_r > 110 and angle_hip_l > 110 :
                        stage = "up"
                    if angle_shoulder_r < 50 and angle_shoulder_l < 50 and angle_hip_r < 90 and angle_hip_l < 90 and stage == 'up':
                        stage = "down"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (275,80), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)
                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
    if exercise_name == "Push-up":
        cap = cv2.VideoCapture(0)

        # Curl counter variables
        counter = count
        stage = None

        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
               
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
           
                # Make detection
                results = pose.process(image)
           
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
               
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                   
                    # Get coordinates for left arm
                    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                   
                    # Calculate angle
                    angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_elbow_l),
                                tuple(np.multiply(elbow_l, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   
                    # Get coordinates for right arm
                    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                   
                    # Calculate angle
                    angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                   
                    # Visualize angle
                    cv2.putText(image, str(angle_elbow_r),
                                tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                   

                   
                    # Curl counter logic
                    if angle_elbow_l > 150 and angle_elbow_r > 150 :
                        stage = "up"
                    if angle_elbow_l < 90 and angle_elbow_r < 90  and stage == 'up':
                        stage="down"
                        counter +=1
                        print(counter)
                           
                except:
                    pass
               
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
               
                # Rep data
                cv2.putText(image, 'REPS', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
                # Stage data
                cv2.putText(image, 'STAGE', (80,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (80,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
               
               
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )              
               
                cv2.imshow('Mediapipe Feed', image)
                if (cv2.waitKey(10) & 0xFF == ord('q')) or counter == given:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(counter)
            return counter
           
def tasks():
    with st.container(border = True, height = 660):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)    
                   
        tab1, tab2, tab3 = st.tabs(["Assigned","Pending", "Finished"])
        with tab1:
            st.success(":point_right: :point_right::point_right: Move to **Pending tab** to finish the tasks")
            email = st.session_state.user_email
            data = trainees_col.find_one({"_id": email})
            group_ids = [group["group_id"] for group in data["groups"]]
            current_date = datetime.datetime.now().strftime("%d-%b-%Y")
            found = False
            for group_id in group_ids:
                group_details = groups_col.find_one({"_id": group_id})
               
                # Check if the group has tasks assigned for the current date
                if "assigned" in group_details:
                    for task in group_details["assigned"]:
                        if task["date"] == current_date:
                            tasks = task.get("exercises_info")
                            if tasks:
                                found = True
                                with st.container(border= True):
                                    c1, c2, c3 = st.columns([1,3,1])
                                    c1.markdown(f'<h3 style="color: red;">Group Name:</h3>', unsafe_allow_html=True)
                                    c2.markdown(f'<h3 style="color: blue;"> {group_details["name"]}</h3>', unsafe_allow_html=True)
                                    c3.markdown(f'<h3 > Today Tasks</h3>', unsafe_allow_html=True)
                                    for i in tasks:
                                        with st.container(border = True):
                                            c1, c2 , c3, c4, c5= st.columns([2,2,2, 2,1])
                                            c1.markdown(f'<h4 style="color: green;">Exercise Name: </h3>', unsafe_allow_html = True)
                                            c2.markdown(f'<h4>{i["name"]}</h3>', unsafe_allow_html = True)
                                            c4.markdown(f'<h4 style="color: green;">Assigned Count : </h3>', unsafe_allow_html = True)
                                            c5.markdown(f'<h4>{i["count"]}</h3>', unsafe_allow_html = True)
                                           
            if found == False:
                st.error("No tasks assigned till now")
       
       
        with tab2:
            with st.container(border = True):
                st.markdown('<h4 style="color: black; font-style : sans-serif;">--> Position the camera directly in front of you to ensure your entire body is visible</h4>', unsafe_allow_html = True)
                st.markdown('<h4 style="color: blue; font-style: italic;"> ***Click <span style="color: red;">\'q\'</span> to finish the task</h4>', unsafe_allow_html=True)

                email = st.session_state.user_email
                data = trainees_col.find_one({"_id": email})
                group_ids = [group["group_id"] for group in data["groups"]]
                current_date = datetime.datetime.now().strftime("%d-%b-%Y")
                found = False
                for day in data.get("daily_info", []):
                    if day.get("date") == current_date:
                        for info in day.get("todays_info", []):
                            for group_id in group_ids:
                                if info.get("group_id") == group_id:
                                    
                                    exer_info = info.get("details", {}).get("exercises_info", [])
                                    for e in exer_info:
                                        if e.get("counts", ["", ""])[1] > e.get("counts", ["", ""])[0]:
                                            found = True
                                            with st.container(border=True):
                                                c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 2, 1, 1, 1, 1, 1])
                                                c1.markdown('<h4 style="color: green;">Exercise Name: </h4>', unsafe_allow_html=True)
                                                c2.markdown(f'<h4 style = "background-color: white;">{e.get("name", "")}</h4>', unsafe_allow_html=True)
                                                c3.markdown('<h4 style="color: green;">Assigned : </h4>', unsafe_allow_html=True)
                                                c4.markdown(f'<h4 style = "background-color: white;">{e.get("counts", ["", ""])[1]}</h4>', unsafe_allow_html=True)
                                                c5.markdown('<h4 style="color: green;">Finished : </h4>', unsafe_allow_html=True)
                                                c6.markdown(f'<h4 style = "background-color: white;">{e.get("counts", ["", ""])[0]}</h4>', unsafe_allow_html=True)
                                                if c7.button("Start", key = email+group_id+e.get("name", ""), type = "primary", use_container_width = True):
                                                    count = start_exercise(e.get("name", ""), e.get("counts", ["", ""])[0], e.get("counts", ["", ""])[1])
                                                    exercise_name = e.get("name", "")
                                                    query = {"_id": email,"daily_info": {"$elemMatch": {"date": current_date,"todays_info": {"$elemMatch": {"group_id": group_id,"details.exercises_info": {"$elemMatch": {"name": exercise_name}}}}}}}

                                                    update_query = {"$set": {"daily_info.$[outer].todays_info.$[inner].details.exercises_info.$[exercise].counts.0": count}}
                                                    array_filters = [{"outer.date": current_date},{"inner.group_id": group_id},{"exercise.name": exercise_name}]
                                                    result = trainees_col.update_one(query, update_query, array_filters=array_filters)

                                                    if result.modified_count > 0:
                                                        st.success("Count updated successfully")
                                                    else:
                                                        st.error("An error occurred. Try again")
                                                    if result.modified_count > 0 and count == e.get("counts", ["", ""])[1]:
                                                            ex_name = e.get("name","")
                                                            msg = f"Hurray !! You finished {ex_name} task. Great Job"
                                                            st.toast(msg, icon = '🎊')

                if found == False:
                    st.error("No tasks pending till now")
       
        
        with tab3:
            found = False
            with st.container(border= True):
               
                email = st.session_state.user_email
                data = trainees_col.find_one({"_id": email})
                group_ids= []
                for group in data["groups"]:
                    group_ids.append(group["group_id"])
                for group_id in group_ids:
                    for day in data.get("daily_info", []):
                        if day.get("date") == current_date:
                            for info in day.get("todays_info", []):
                                if info.get("group_id") == group_id:
                                    c1, c2= st.columns([1,5])
                                    group_details = groups_col.find_one({"_id": group_id})
                                    group_name = group_details["name"]
                                    with st.container(border = True):
                                        c1.markdown(f'<h3 style="color: red;">Group Name:</h3>', unsafe_allow_html=True)
                                        c2.markdown(f'<h3 style="color: blue;"> {group_name} </h3>', unsafe_allow_html=True)
                                   
                                        exer_info = info.get("details", {}).get("exercises_info", [])
                                        for e in exer_info:
                                            if e.get("counts", ["", ""])[1] == e.get("counts", ["", ""])[0]:
                                                found = True
                                                with st.container(border=True):
                                                    c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1, 1, 1, 1])
                                                    c1.markdown('<h4 style="color: green;">Exercise Name: </h4>', unsafe_allow_html=True)
                                                    c2.markdown(f'<h4 style = "background-color: white;">{e.get("name", "")}</h4>', unsafe_allow_html=True)
                                                    c3.markdown('<h4 style="color: green;">Assigned : </h4>', unsafe_allow_html=True)
                                                    c4.markdown(f'<h4 style = "background-color: white;">{e.get("counts", ["", ""])[1]}</h4>', unsafe_allow_html=True)
                                                    c5.markdown('<h4 style="color: green;">Finished : </h4>', unsafe_allow_html=True)
                                                    c6.markdown(f'<h4 style = "background-color: white;">{e.get("counts", ["", ""])[0]}</h4>', unsafe_allow_html=True)
                if found == False:
                    st.error("No tasks finished till now")
           
   
                                               

def generate_random_id():
    # Generate 3 random digits
    digits = ''.join(random.choices(string.digits, k=4))
   
    # Generate 3 random alphabets
    alphabets = ''.join(random.choices(string.ascii_letters, k=4))
   
    # Generate 2 random characters (either digit or alphabet)
    others = ''.join(random.choices(string.ascii_letters + string.digits, k=2))
   
    # Shuffle all characters
    id_characters = list(digits + alphabets + others)
    random.shuffle(id_characters)
   
    # Convert the list of characters to a string
    return ''.join(id_characters)



def create_group(group_name):
    # Check if the group name already exists
    email = st.session_state.user_email
    details = trainers_col.find_one({"_id": email})
    groups = details["groups"]
    existing_groups = [value for value in groups.values()]
    if group_name in existing_groups:
        return 0
    group_id = generate_random_id()
   
    # Insert the group into the database
    status = groups_col.find_one({"_id": group_id})
    if status:
        group_id = generate_random_id()
    group_data = {"_id": group_id, "name" : group_name, "creator_id": st.session_state.user_email, "members": [],
                  "created_on": datetime.datetime.now().strftime("%d-%b-%Y"), "assigned":[], "history":[]}
   
    try:
        groups_col.insert_one(group_data)
    except Exception as e:
        return -1

    # Update the trainers collection to include the group
    try:
        update_result = trainers_col.update_one({"_id": st.session_state.user_email}, {"$set": {"groups."+group_id: group_name}})
        if update_result.modified_count > 0:
            return 1
        else:
            return -1
    except Exception as e:
        # If updating the trainer's groups fails, delete the group created
        groups_col.delete_one({"_id": group_id})
        return -1


def update_group_assignments(selected_exercises, group_ids):
    current_date = datetime.datetime.now().strftime("%d-%b-%Y")
    s = 0
    assigned = 0
    for group_id in group_ids:
        data = groups_col.find_one({"_id": group_id})
        today_tasks = data["assigned"]
        present = len(today_tasks)
        for i in today_tasks:
            if i["date"] == current_date:
                break
            else:
                assigned+=1
                
        if assigned == present:        
            # Transform selected exercises and push into 'assigned' field
            transformed_exercises = [{'name': ex['Exercise'], 'count': ex['Count']} for ex in selected_exercises]
            result = groups_col.update_one(
                {"_id": group_id},
                {"$push": {"assigned": {"date": current_date, "exercises_info": transformed_exercises}}}
            )
            details = groups_col.find_one({"_id": group_id})
            result1 = groups_col.update_one({"_id": group_id}, {"$push":{"history":{"date": current_date, "members": details["members"]}}})

            exercises_info = []
            for t in transformed_exercises:
                d = {"name": t["name"], "counts": [0, t["count"]]}
                exercises_info.append(d)

            members = details["members"]
            for email in members:
                details1 = trainees_col.find_one({"_id": email})
                assigned1 = len(details1["daily_info"])
                dates_length = 0
                for date in details1["daily_info"]:
                    if date["date"] == current_date:
                        assigned2 = len(date["todays_info"])
                        groups_length = 0
                        for groups in date["todays_info"]:
                            if groups["group_id"] == group_id:
                                break
                            else:
                                groups_length+=1
                        if groups_length ==assigned2:
                            trainees_col.update_one({"_id": email, "daily_info.date": current_date},
                                                    {"$push": {"daily_info.$.todays_info": {"group_id": group_id,"details": {"status": 0,"exercises_info": exercises_info}}}})
                    
                    else:
                        dates_length+=1
                                
                if dates_length == assigned1:
                    trainees_col.update_one({"_id": email}, {"$push": {"daily_info": {"date": current_date , "todays_info" : [{"group_id" : group_id, "details" : { "status":0, "exercises_info": exercises_info }}]}}})
                    
    st.success("Tasks assigned successfully")                    
                                   
                    
                    
                
def assign_task(groups):
    if "user_email" in st.session_state:
        user_email = st.session_state.user_email
    user_details = trainers_col.find_one({"_id": user_email})
    exercises_list = user_details["exercises"]
    selected_exercises = {}
    exercise_counts = {}
    with st.form("exercise_form", clear_on_submit=True):
        mid = len(exercises_list) // 2
        exercises_list_1 = exercises_list[:mid]
        exercises_list_2 = exercises_list[mid:]
        col1, col2 = st.columns(2)
        for exercise_1, exercise_2 in zip(exercises_list_1, exercises_list_2):
       
            exercise_count_1 = col1.number_input(exercise_1, min_value=0, format="%d")
            exercise_counts[exercise_1] = exercise_count_1

            exercise_count_2 = col2.number_input(exercise_2, min_value=0, format="%d")
            exercise_counts[exercise_2] = exercise_count_2
        submit_button = st.form_submit_button("**Submit**", type = "primary", use_container_width=True)
   
    if submit_button:
    # Filter exercises with counts > 0
        selected_exercises = {k: v for k, v in exercise_counts.items() if v > 0}
       
        if not selected_exercises:
            st.error("Please select at least one exercise to proceed.")
        else:
            st.markdown('<h3>Task Details</h3>', unsafe_allow_html=True)
            data = [{"Exercise": exercise, "Count": count} for exercise, count in selected_exercises.items()]
            st.table(data)
            update_group_assignments(data, groups)
           
   
def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def groups1():
    with st.container():
       
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)        
        st.markdown('<br>', unsafe_allow_html=True)      
        tab1, tab2, tab3, tab4 = st.tabs([ "Tasks Created","Assign Tasks", "Create Group", "Group Details"])
       
        with tab1:
            with st.container(border = True):    
                st.markdown('<h2 class = "center-main-heading" >Tasks Assigned Today</h2><br>', unsafe_allow_html=True)
                email = st.session_state.user_email
                details = trainers_col.find_one({"_id": email})
                group_ids = list(details["groups"].keys())
                group_names = list(details["groups"].values())
                current_date = datetime.datetime.now().strftime("%d-%b-%Y")
                found = False
                for group_id in group_ids:
                    group_details = groups_col.find_one({"_id": group_id})
                   
                    # Check if the group has tasks assigned for the current date
                    if "assigned" in group_details:
                        for task in group_details["assigned"]:
                            if task["date"] == current_date:
                                tasks = task.get("exercises_info")
                                if tasks:
                                    found = True
                                    with st.container(border= True):
                                        c1, c2, c3 = st.columns([1,3,1])
                                        c1.markdown(f'<h3 style="color: red;">Group Name:</h3>', unsafe_allow_html=True)
                                        c2.markdown(f'<h3 style="color: blue;"> {group_details["name"]}</h3>', unsafe_allow_html=True)
                                        c3.markdown(f'<h3 > Today Tasks</h3>', unsafe_allow_html=True)
                                        # st.markdown(f'<h3 style="color: blue;">Group Name: {group_details["name"]}</h3>', unsafe_allow_html=True)
                                        for i in tasks:
                                            with st.container(border = True):
                                                c1, c2 , c3, c4, c5= st.columns([2,2,2, 2,1])
                                                c1.markdown(f'<h4 style="color: green;">Exercise Name: </h3>', unsafe_allow_html = True)
                                                c2.markdown(f'<h4>{i["name"]}</h3>', unsafe_allow_html = True)
                                                c4.markdown(f'<h4 style="color: green;">Assigned Count : </h3>', unsafe_allow_html = True)
                                                c5.markdown(f'<h4>{i["count"]}</h3>', unsafe_allow_html = True)
                                               
                if found == False:
                    st.error("No tasks assigned till now")
           
               
        with tab2:  
            col11, col12, col13 = st.columns([1,3,1])  
            with col12:
                with st.container(border = True):    
                    st.markdown('<h2 class = "center-main-heading" >Select Group(s)</h2><br>', unsafe_allow_html=True)
                    # Input for group name
                    if "group" not in st.session_state:
                        st.session_state.group = None        
                    if "user_email" in st.session_state:
                        user_email = st.session_state.user_email

                    if "group" not in st.session_state:
                        st.session_state.group = None  
                    if "option" not in st.session_state:
                        st.session_state.option = None
                    if "options" not in st.session_state:
                        st.session_state.options = []
                       
                    st.session_state.group = st.radio("**Choose One**", ["Single", "Multiple"], horizontal = True, index = None)
                    if st.session_state.group:
                        groups_info = trainers_col.find_one({"_id": user_email})
                        groups_ids = groups_info["groups"].keys()
                        groups_ids = list(groups_ids)
                        groups_names = groups_info["groups"].values()
                        groups_names = list(groups_names)
                       
                        groups_list = []
                        for i in range(len(groups_ids)):
                            groups_list.append(groups_ids[i]+"-"+groups_names[i])
                           
                        if st.session_state.group == "Single":
                            st.session_state.option = st.selectbox("**Select a group**", groups_list, index=None, placeholder="Choose a group from below...")
                        elif st.session_state.group == "Multiple":
                            st.session_state.options = st.multiselect("**Select multiple groups**", groups_list,  placeholder="Choose groups from below...")

                       
                       
                        groups_selected = []
                        groups_selected_ids = []
                       
                        # Handling the display of selected groups based on the choice of single or multiple
                        if st.session_state.group == "Single" and st.session_state.option:
                            with st.container():
                                st.markdown('<h4>Group selected is : </h4>', unsafe_allow_html=True)
                                with st.container(border= True):    
                                    groups_selected.append(st.session_state.option[11:])
                                    groups_selected_ids.append(st.session_state.option[:10])
                                    st.write(groups_selected[0])
                                st.markdown('<h4>Exercises : </h4>', unsafe_allow_html=True)
                                assign_task(list(groups_selected_ids))
                        elif st.session_state.group == "Multiple" and st.session_state.options:
                            with st.container():
                                st.markdown('<h4>Groups selected are : </h4>', unsafe_allow_html=True)
                                with st.container(border= True):    
                                    for i in st.session_state.options:
                                        groups_selected.append(i[11:])
                                        groups_selected_ids.append(i[:10])
                                    # grps = ""
                                    grps = ", ".join(groups_selected)
                                    # for j in groups_selected:    
                                    #     grps = grps+j+", "
                                    st.write(grps)
                                    # st.write(", ".join(st.session_state.options))
                                st.markdown('<h4>Exercises : </h4>', unsafe_allow_html=True)
                                # assign_task(list(st.session_state.options[:10]))
                                assign_task(list(groups_selected_ids))      
                               
                           
                           
                   
        with tab3:
            col11, col12, col13 = st.columns([1,3,1])
            with col12:
                with st.container(border = True):    
                    st.markdown('<h2 class = "center-main-heading" >Create Group</h2><br>', unsafe_allow_html=True)
                    # Input for group name
                    group_name = st.text_input("**Group Name**", placeholder="Enter Group Name here...")


                    # Button to create the group
                    if st.button("Submit", key = "create_group", type="primary",  use_container_width=True):
                        if group_name:
                            res = create_group(group_name)
                            if res<0:
                                st.error("Group creation failed. Retry again.")
                                time.sleep(2)
                                st.rerun()
                            elif res==0:
                                st.error(f"Group '{group_name}' already exists. Please try with another name.")
                               
                            else:
                                st.success("Group created successfully")
                                time.sleep(2)
                                st.rerun()
                        else:
                            st.warning("Please enter a group name.")
                       
        with tab4:
            email = st.session_state.user_email
            trainer_data = trainers_col.find_one({"_id" : email})
            ids = list(trainer_data["groups"].keys())
            names = list(trainer_data["groups"].values())
           
            for i in range(len(ids)):
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        c1, c2 = st.columns([1, 2])
                        c1.markdown('<h3 style="color: red;">Group Name:</h3>', unsafe_allow_html=True)
                        c2.markdown(f'<h3>{names[i]}</h3>', unsafe_allow_html=True)
                        c1.markdown('<h3 style="color: red;">Group ID:</h3>', unsafe_allow_html=True)
                        c2.markdown(f'<h3>{ids[i]}</h3>', unsafe_allow_html=True)
                    with col2:    
                    # if c5.button("Members", key=ids[i], type="primary", use_container_width=True):
                        data = groups_col.find_one({"_id": ids[i]})
                        members = data["members"]
                       
                        # List to store members' details
                        member_details = []
                        for mail in members:
                            details = collection.find_one({"_id": mail})
                            f_name = details["details"]["first_name"]
                            l_name = details["details"]["last_name"]
                            name = f_name + " " + l_name
                            member_details.append({"Mail": mail, "Name": name})
                        if len(member_details) == 0:
                            col2.error("**Group is empty**", icon = "⚠️")
                        else:
                            coll1, coll2, coll3 = st.columns([3,1,1])
                            df = pd.DataFrame(member_details)
                            close = False
                            coll1.markdown('<h3 style = "color: red;">Group Details:</h3>', unsafe_allow_html=True)
                            random_string = generate_random_string(20)
                            if coll2.button("View", key= str(email)+ str(ids[i]), type="primary"):
                                with st.container(border = True):
                                    col2.dataframe(df, use_container_width=True)
                                if coll3.button("Close", key = str(email)+str(names[i]), type = "primary"):
                                    st.rerun()
                                    
                                    
    
    
def dashboard1():
    with st.container(border = True):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1><br>', unsafe_allow_html=True)   
        tab1, tab2 = st.tabs(["Yesteday result", "Overall"])
        with tab1:
            email = st.session_state.user_email
            details = trainers_col.find_one({"_id": email})      
            group_ids = list(details["groups"].keys())
            group_names = list(details["groups"].values())
            group_ids_names = []
            for i in range(len(group_names)):
                group_ids_names.append(str(group_ids[i])+"-"+str(group_names[i]))
            
            if group_ids_names == None:
                st.error("You didn't create any groups yet!!!")
            else:
                selected_group = st.selectbox("Choose a group", group_ids_names, index = None, key = "trainers_groups_yesterdays_info")
                if selected_group:
                    selected_group_id = selected_group[:10]
                    group_details = groups_col.find_one({"_id": selected_group_id})
                    group_members = group_details["members"]
                    if group_members == None:
                        c1.error("Group is empty!")
                    else:
                        current_datetime = datetime.datetime.now()
                        yesterday_datetime = current_datetime - timedelta(days=1)
                        yesterday_date = yesterday_datetime.strftime('%d-%b-%Y')
                        present = group_details["assigned"]
                        found = False
                        for date in present:
                            if date["date"] == yesterday_date:
                                found = True
                                history = group_details["history"]
                                for date in history: 
                                    if date["date"] == yesterday_date:
                                        members = date["members"]
                                        member_ids = []
                                        names = []
                                        results = []
                                        
                                        for member in members:
                                            member_details = trainees_col.find_one({"_id": member})
                                            personal_details = collection.find_one({"_id": member})
                                            f_name = personal_details["details"]["first_name"]
                                            l_name = personal_details["details"]["last_name"]
                                            name = l_name+" "+ f_name
                                            daily_info = member_details["daily_info"]
                                            if daily_info != None:
                                                found = False
                                                for date in daily_info:
                                                    if date["date"] == yesterday_date:
                                                        groups = date["todays_info"]
                                                        for detail in groups:
                                                            if detail["group_id"] == selected_group_id:
                                                                found = True
                                                                exercises_info = detail["details"]["exercises_info"]
                                                                assigned_total = 0
                                                                finished_total = 0
                                                                for exercise in exercises_info:
                                                                    assigned_total += exercise["counts"][1]
                                                                    finished_total += exercise["counts"][0]
                                                                member_ids.append(str(member))
                                                                names.append(str(name))
                                                                results.append(int((finished_total/assigned_total)*100))
                                                                break
                                                            
                                            break                    
                                break   
                            
                        if found:
                            # Create a DataFrame
                            df = pd.DataFrame({'MailID': member_ids, 'Name': names, 'Result': results})

                            # Define a function to apply styles based on the 'Result' column
                            def highlight_row(row):
                                if row['Result'] >= 70:
                                    return ['background-color: lightgreen'] * len(row)
                                elif 40 <= row['Result'] < 70:
                                    return ['background-color: lightblue'] * len(row)
                                else:
                                    return ['background-color: red'] * len(row)

                            # Apply styles to the DataFrame
                            styled_df = df.style.apply(highlight_row, axis=1)

                            st.dataframe(styled_df, use_container_width=True)     
                        else:
                            st.error("No data found")              
                                                
                                                            
                                                            
        
        with tab2:
            email = st.session_state.user_email
            details = trainers_col.find_one({"_id": email})      
            group_ids = list(details["groups"].keys())
            group_names = list(details["groups"].values())
            group_ids_names = []
            for i in range(len(group_names)):
                group_ids_names.append(str(group_ids[i])+"-"+str(group_names[i]))
            
            if group_ids_names == None:
                st.error("You didn't create any groups yet!!!")
            else:
                c1, c2 = st.columns(2)
                selected_group = c1.selectbox("Choose a group", group_ids_names, index = None, key = "trainers_groups")
                if selected_group:
                    selected_group_id = selected_group[:10]
                    group_details = groups_col.find_one({"_id": selected_group_id})
                    group_members = group_details["members"]
                    if group_members == None:
                        c1.error("Group is empty!")
                    else:
                        selected_member_id = c2.selectbox("Choose a trainee", group_members, index = None, key = "trainers_groups_members")
                        if selected_member_id: 
                            member_details = trainees_col.find_one({"_id": selected_member_id})
                            daily_info = member_details["daily_info"]
                            if daily_info != None:
                                dates = []
                                percentages = []
                                found = False
                                for date in daily_info:
                                    groups = date["todays_info"]
                                    for detail in groups:
                                        if detail["group_id"] == selected_group_id:
                                            found = True
                                            exercises_info = detail["details"]["exercises_info"]
                                            assigned_total = 0
                                            finished_total = 0
                                            for exercise in exercises_info:
                                                assigned_total += exercise["counts"][1]
                                                finished_total += exercise["counts"][0]
                                            dates.append(date['date'])
                                            percentages.append(int((finished_total/assigned_total)*100))
                                        
                                if found:                                                 
                                    # Create line graph using Matplotlib
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(dates, percentages, marker='o', color='b', linestyle='-')
                                    ax.set_title('Percentage Progress Over Time')
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Percentage')
                                    ax.set_ylim(0, 100)  # Set maximum y-value as 100
                                    ax.tick_params(axis='x', rotation=45)
                                    ax.grid(True)
                                    plt.tight_layout()

                                    # Display the line chart using Streamlit
                                    st.pyplot(fig)
                            else:
                                c2.error("No data found")
                
            


def dashboard2():
    with st.container(border=True):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1><br>', unsafe_allow_html=True)   
        
        email = st.session_state.user_email 
        selected = st.radio("**Select One**",["Day-Wise", "Overall"] , index=None, horizontal=True)
        if selected == "Day-Wise":
            first_document = trainees_col.find_one({"_id": email})
            if len(first_document['daily_info']) == 0:
                st.error("No data found ")
               
            else:
                start_date = first_document['daily_info'][0]['date']
                parsed_date = datetime.datetime.strptime(start_date, "%d-%b-%Y").date()

                # Extract year, month, and day from the parsed date
                year = parsed_date.year
                month = parsed_date.month
                day = parsed_date.day
                c1, c2 = st.columns([1, 1])
                date_selected = c1.date_input(":darkblack[**Choose date to see results**]", format="DD/MM/YYYY", value=None, min_value=datetime.date(year, month, day), max_value=datetime.date.today())
                if date_selected:
                    formatted_date = date_selected.strftime("%d-%b-%Y")
                    details = trainees_col.find_one({"_id": email})
                    daily_info = details["daily_info"]
                    found = False
                    for date in daily_info:
                        if str(formatted_date) == date["date"]:
                            found = True
                            todays_info = date["todays_info"]
                            group_ids_names = []
                            for groups in todays_info:
                                group_id = groups["group_id"]
                                group_info = groups_col.find_one({"_id": group_id}) 
                                group_ids_names.append(str(group_id) + "-" + str(group_info["name"]))
                            group_selected = c2.selectbox("**Select a group**", group_ids_names ,index=None, key = "single_day"+str(formatted_date))
                            if group_selected:
                                group_id = group_selected[:10]
                                for groups in todays_info:
                                    if groups["group_id"] == group_id:
                                        exercises_info = groups["details"]["exercises_info"]
                                        names = []
                                        assigned = []
                                        finished = []
                                        assigned_total = 0
                                        finished_total = 0
                                        for exercise in exercises_info:
                                            names.append(exercise["name"])
                                            assigned.append(exercise["counts"][1])
                                            finished.append(exercise["counts"][0])         
                                        
                                        # Set the width of the bars
                                        bar_width = 0.30

                                        # Set the positions of the bars on the x-axis
                                        r1 = np.arange(len(names))
                                        r2 = [x + bar_width for x in r1]

                                        # Create bar chart
                                        fig, ax = plt.subplots()
                                        ax.bar(r1, assigned, color='blue', width=bar_width, label='Assigned')
                                        ax.bar(r2, finished, color='orange', width=bar_width, label='Finished')
                                        ax.set_xlabel('Exercise')
                                        ax.set_ylabel('Counts')
                                        ax.legend()
                                        plt.xticks(r1 + bar_width / 2, names, rotation=45)
                                        st.pyplot(fig)
                                                
                    if not found:
                        c1.error("Details not found")
                    
        if selected == "Overall":    
            details = trainees_col.find_one({"_id": email})
            groups = details["groups"]
            group_ids_names = []
            for group in groups:
                group_ids_names.append(str(group["group_id"])+"-"+str(group["name"]))  
            if len(group_ids_names) == 0:
                st.error("You did not join any group till now.")
            else:     
                group_id_name = st.selectbox("**Select a group**", group_ids_names, index = None, key = "overall")
                if st.button("Show Daily Progress", type = "primary", use_container_width = True):
                    if group_id_name:
                        group_id = group_id_name[:10]
                        daily_info = details["daily_info"]
                        dates = []
                        percentages = []
                        found = False
                        for date in daily_info:
                            groups = date["todays_info"]
                            for detail in groups:
                                if detail["group_id"] == group_id:
                                    found = True
                                    exercises_info = detail["details"]["exercises_info"]
                                    assigned_total = 0
                                    finished_total = 0
                                    for exercise in exercises_info:
                                        assigned_total += exercise["counts"][1]
                                        finished_total += exercise["counts"][0]
                                    dates.append(date['date'])
                                    percentages.append(int((finished_total/assigned_total)*100))
                                
                        if found:                                                 
                            # Create line graph using Matplotlib
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(dates, percentages, marker='o', color='b', linestyle='-')
                            ax.set_title('Percentage Progress Over Time')
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Percentage')
                            ax.set_ylim(0, 100)  # Set maximum y-value as 100
                            ax.tick_params(axis='x', rotation=45)
                            ax.grid(True)
                            plt.tight_layout()

                            # Display the line chart using Streamlit
                            st.pyplot(fig)
                        else:
                            st.error("No data found")
                    else:
                        st.error("Please select a group") 

 
  
def groups2():
    with st.container(border = True, height = 660):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1><br>', unsafe_allow_html=True)      
       
        with st.container(border= True):
            id_list = []
            # Iterate over the documents in the collection
            for doc in groups_col.find({}, {"_id": 1, "name": 1}):
                # Concatenate _id and name and append to the list
                id_list.append(f"{doc['_id']}-{doc['name']}")
               
            group_id_name = st.selectbox("**Search here**", id_list, index=None,  placeholder="Enter Group ID/Group Name")
           
            c11, c22 = st.columns(2)
           
            if c11.button("Join",key = "request to join",  use_container_width = True, type = "primary"):
                if group_id_name != None:
                    group_id = group_id_name[:10]
                    details = groups_col.find_one({"_id": group_id})
                    creator_id = details["creator_id"]
                    group_name = details["name"]
                    details1 = trainers_col.find_one({"_id": creator_id})
                    email = st.session_state.user_email
                   
                    data = trainees_col.find_one({"_id": email})

                    if "groups" in data and any(group.get("group_id") == group_id for group in data["groups"]):
                        st.error(f"You are already a member of {group_name}")
                    else:
                        result1 = trainees_col.update_one({"_id": email},{"$push": {"groups": {"group_id": group_id, "name": group_name}}})
                        result2 = groups_col.update_one({"_id": group_id},{"$push": {"members": email}})
                               
                        if result1.modified_count>0 and result2.modified_count>0:
                            st.success(f"You have joined {group_name} successfully")    
                else:
                    st.error("Please select a group")
           
            if c22.button("Leave",key = "request to leave",  use_container_width = True, type = "primary"):
                if group_id_name != None:
                    group_id = group_id_name[:10]
                    details = groups_col.find_one({"_id": group_id})
                    creator_id = details["creator_id"]
                    group_name = details["name"]
                    details1 = trainers_col.find_one({"_id": creator_id})
                    email = st.session_state.user_email
                   
                    data = trainees_col.find_one({"_id": email})
                    if not any(group.get("group_id") == group_id for group in data.get("groups", [])):
                        st.error(f"You are not a member of {group_name}")
                    else:
                        result1 = trainees_col.update_one({"_id": email},{"$pull": {"groups": {"group_id": group_id, "name": group_name}}})
                        result2 = groups_col.update_one({"_id": group_id},{"$pull": {"members": email}})
                       
                        if result1.modified_count > 0 and result2.modified_count > 0:
                            st.success(f"You left {group_name} successfully")
               
                else:
                    st.error("Please select a group")
           
        with st.container(border= True):
            st.markdown('<h3 class="center-main-heading">JOINED GROUPS</h3><br>', unsafe_allow_html=True)
            data = trainees_col.find_one({"_id" : st.session_state.user_email})
            for group in data.get("groups", []):
                with st.container(border= True):
                    c1, c2, c3, c4 = st.columns([1,3,1,1])
                    c1.markdown('<h3 style = "color: red;">Group Name:</h3>', unsafe_allow_html=True)
                    c2.markdown(f'<h3>{group["name"]}</h3>', unsafe_allow_html=True)
                    c3.markdown('<h3 style = "color: red;">Group ID:</h3>', unsafe_allow_html=True)
                    c4.markdown(f'<h3>{group["group_id"]}</h3>', unsafe_allow_html=True)
                   
                           
# check password strength
def check_password_strength(password):
    # Length of password
    if len(password)<8:
        return False
       
    # Check for at least one uppercase letter
    uppercase_regex = re.compile(r'[A-Z]')
    if not uppercase_regex.search(password):
        return False

    # Check for at least one lowercase letter
    lowercase_regex = re.compile(r'[a-z]')
    if not lowercase_regex.search(password):
        return False

    # Check for at least one digit
    digit_regex = re.compile(r'\d')
    if not digit_regex.search(password):
        return False

    # Check for at least one special character
    special_char_regex = re.compile(r'[!@#$%^&*(),.?":{}|<>]')
    if not special_char_regex.search(password):
        return False

    return True


def profile():
    with st.container(border = True, height = 660):
        # Apply the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)  
        st.markdown('<br>', unsafe_allow_html=True)
   
        if "user_email" in st.session_state:
            user_email = st.session_state.user_email
            user_details = collection.find_one({"_id" : user_email})
            first_name = user_details["details"]["first_name"]
            last_name = user_details["details"]["last_name"]
            password = user_details["details"]["password"]
           
            if "first_name" not in st.session_state:
                st.session_state["first_name"] = first_name
           
            if "last_name" not in st.session_state:
                st.session_state["last_name"] = last_name
               
           
            if "password" not in st.session_state:
                st.session_state["password"] = password
                   
        with st.container(border=True):
           
            col1, col2, col3 = st.columns(3)
            col1.markdown('<h3 class = "user-id" >Name</h3>', unsafe_allow_html=True)
            col1.markdown(f'<h5 class = "mail-id" >{first_name} {last_name}</h5>', unsafe_allow_html=True)
           
            col2.markdown('<h3 class = "user-id" >User ID</h3>', unsafe_allow_html=True)
            col2.markdown(f'<h5 class = "mail-id" >{user_email}</h5>', unsafe_allow_html=True)
           
            col3.markdown('<h3 class = "user-id" >Rating</h3>', unsafe_allow_html=True)
            col3.markdown(f'<h5 class = "mail-id" >8.5/10</h5>', unsafe_allow_html=True)
           
        with st.container():
            col1, col2 = st.columns([2,1])
            with col1.form(key="personal_details"):
                st.markdown('<h2 class = "center-main-heading" >Update Personal Details</h2>', unsafe_allow_html=True)
                st.markdown('<br>', unsafe_allow_html=True)
                col3, col4 = st.columns([2, 1])
                st.session_state.first_name = col3.text_input("**First Name**", value=st.session_state.first_name)
                st.session_state.last_name = col4.text_input("**Last Name**", value=st.session_state.last_name)
                st.session_state.password = st.text_input("**Password**", type="password", value=st.session_state.password)
                # st.markdown('<br>', unsafe_allow_html=True)
                if st.form_submit_button("***Submit***", type="primary", use_container_width=True):
                   
                    if st.session_state.first_name == '' or st.session_state.last_name == '' or st.session_state.password == '':
                        st.warning("Enter all details to continue")
                        st.stop()
                    else:
                        if check_password_strength(password):
                            # Update the document in MongoDB
                            query = {"_id": user_email}  
                            update_query = {
                                "$set": {
                                    "details.first_name": st.session_state.first_name,
                                    "details.last_name": st.session_state.last_name,
                                    "details.password": st.session_state.password
                                }
                            }
                            result = collection.update_one(query, update_query)

                            if result.modified_count > 0:
                                st.success("Details updated successfully.")
                               
                                st.rerun()
                            else:
                                st.error("Details not updated")
                        else:
                            st.warning("Password should contain 1 uppercase, 1 lowercase, 1 digit, 1 special character and minimum 8 characters")      
                           
                           
            with col2.container(border=True, height = 350):
                st.write("Profile Picture")
                st.image(profile_img)
       
                           
def logout():
    with st.container(border = True):
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)  
        st.markdown('<br>', unsafe_allow_html=True)
   
        c1, c2, c3 = st.columns(3)
        with c2:
            st.markdown('<h3> Are you sure to logout?</h3>', unsafe_allow_html = True)
            c11, c12 , c13, c14= st.columns(4)
            if c12.button("**Yes**", type = "primary", use_container_width= True):
                st.switch_page("login.py")
         
                           



def main_1():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",  # required
            options=["Home", "Groups", "Dashboard", "Profile", "Logout"],  # required
            icons=["house", "people-fill", "display-fill", "person-circle", "box-arrow-right"],  # optional
            menu_icon="menu-button-wide-fill",  # optional
            default_index=0,  # optional
            styles={
                "container": {"padding": "0!important", "background-color": "#C3E3EB",
                              "display": "flex", "justify-content": "space-around", },
                "icon": {"color": "Indianred", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "9px",
                    "--hover-color": "#90EE90",
                    "font-weight": "bold",    
                },
                "nav-link-selected": {"background-color": "#061E45"},

            },
        )

    if selected == "Home":
        home()
    elif selected == "Groups":
        groups1()
    elif selected == "Dashboard":
        dashboard1()
    elif selected == "Profile":
        profile()
    elif selected == "Logout":
        logout()


def main_2():
       
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",  # required
            options=["Home", "Tasks", "Dashboard", "Groups", "Profile", "Logout"],  # required
            icons=["house", "list-task", "display-fill", "people-fill", "person-circle", "box-arrow-right"],  # optional
            menu_icon="menu-button-wide-fill",  # optional
            default_index=0,  # optional
            styles={
                "container": {"padding": "0!important", "background-color": "#C3E3EB",
                              "display": "flex", "justify-content": "space-around", },
                "icon": {"color": "Indianred", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "9px",
                    "--hover-color": "#90EE90",
                    "font-weight": "bold",    
                },
                "nav-link-selected": {"background-color": "#061E45"},

            },
        )

    if selected == "Home":
        home()
    elif selected == "Tasks":
        tasks()
    elif selected == "Dashboard":
        dashboard2()
    elif selected == "Groups":
        groups2()
    elif selected == "Profile":
        profile()
    elif selected == "Logout":
        logout()
       
       
def main():
    if "user_email" in st.session_state:
        user_email = st.session_state.user_email
        user_details = collection.find_one({"_id" : user_email})
        role = user_details["role"]
        if role == "Trainer":
            main_1()
        else:
            main_2()
   
   
if __name__ == "__main__":
    st.session_state.local_status = False
    st.session_state.status = False
    if "login_status" in st.session_state:
        st.session_state.status = st.session_state["login_status"]
        if st.session_state.status:
            st.session_state.local_status = True
    if st.session_state.status or st.session_state.local_status:
       
        main()
       
    else:
        with st.container(border = True):
            # Apply the custom CSS
            st.markdown(custom_css, unsafe_allow_html=True)
            st.markdown('<h1 class="center-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)    
            st.markdown('<br><br>', unsafe_allow_html=True)          
            col1, col2, col3 = st.columns(3)
               
            col2.error("You are not logged in. Please click below to login")
            if col2.button("**Login**", type = "primary"):
                st.switch_page("login.py")
