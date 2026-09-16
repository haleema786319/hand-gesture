# 🖐️ Hand Gesture Recognition System

A real-time **Hand Gesture Recognition System** that uses a webcam to detect hand gestures and control computer actions. The project uses **OpenCV** for video processing, **MediaPipe** for hand landmark detection, and **PyAutoGUI** to perform keyboard actions based on detected gestures.

---

## 📌 Project Overview

This project captures live video from the webcam and detects hand gestures using MediaPipe's hand-tracking technology.

The system counts the number of visible fingers and detects **thumb-up** and **thumb-down** gestures. Based on the detected gesture, it automatically performs keyboard actions such as play/pause, navigation, scrolling, and switching videos.

This provides a **touch-free and interactive way to control media or browser applications** using hand gestures.

 

## ✨ Features

- 🎥 Real-time webcam-based hand detection
- 🖐️ Detects up to one hand at a time
- 🔢 Counts the number of raised fingers
- 👍 Detects **Thumb Up** gesture
- 👎 Detects **Thumb Down** gesture
- ⌨️ Performs keyboard actions automatically
- 🎬 Can be used for media/video navigation
- 🔄 Supports multiple camera indices
- ⚡ Real-time gesture processing
- 🖥️ Displays detected hand landmarks on the webcam feed


**Working** 

- one finger : play/pause
- two fingers : volume increase
- three fingers: volume decrease
- four fingers : next video
- five fingures : before video
 

## 🛠️ Technologies Used

 **Technology & Purpose**

 
 - **Python**    : Main programming language 
 - **OpenCV**    : Webcam access and image processing 
 - **MediaPipe** : Hand landmark detection 
 - **PyAutoGUI** : Simulating keyboard actions 
 - **Time**      : Gesture timing and delay control 

 
