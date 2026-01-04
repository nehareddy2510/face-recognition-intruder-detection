# Face Recognition & Intruder Detection System

## Overview
This project implements a real-time face recognition system that identifies authorized individuals and classifies unknown faces as intruders using webcam input. The system uses deep learning–based facial embeddings for accurate face matching and real-time surveillance.

## Features
- Real-time face detection using dlib
- Face recognition using 128-D facial embeddings
- Live webcam-based surveillance
- Bounding box and name display for detected faces

## Tech Stack
- Python
- OpenCV
- dlib
- NumPy

## How It Works
1. Detect faces in real-time using dlib’s frontal face detector  
2. Extract facial landmarks and generate 128-D embeddings  
3. Compare live embeddings with stored encodings using Euclidean distance  
4. Label unknown faces as **Intruder** when distance exceeds threshold  


