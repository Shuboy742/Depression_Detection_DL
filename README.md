#Emotion Recognition and Depression Detection System using Deep Learning
-This project implements a web application for detecting signs of depression in user-uploaded videos.

#Tech Stack
<br>1)Backend: Python (Flask)</br>
<br>2)Database: SQLite</br>
<br>3)Machine Learning: Keras (TensorFlow backend)</br>
<br>4)Computer Vision: OpenCV</br>
<br>5)Natural Language Processing: TextBlob</br>
<br>6)Speech Recognition: SpeechRecognition</br>
<br>7)Web Development: Flask-Migrate, Flask-SQLAlchemy</br>

#Algorithms
<br>1.Facial Emotion Recognition: Convolutional Neural Network (CNN) trained on facial expressions</br>
<br>2.Sentiment Analysis: TextBlob library for analyzing audio transcripts</br>

#Functionalities
<br1.Users can upload videos.</br>
<br>2.The system detects emotions in the video frames using facial recognition.</br>
<br>3.Audio from the video is extracted and analyzed for sentiment using speech recognition and text analysis.</br>
<br>4.The system combines the results from facial recognition and sentiment analysis to provide an overall indication of potential depression.</br>
<br>5.The results are displayed on the user interface.

#Installation and Usage (Modify as needed)
1. Dependencies:

#Make sure you have the following libraries installed in your Python environment:

numpy
requests
Flask
Flask-SQLAlchemy
Flask-Migrate
keras
OpenCV
tensorflow
TextBlob
SpeechRecognition
moviepy

2. Database Setup:

The system uses an SQLite database for storing user information. You can create the database tables by running the application with the debug=True flag:
<br>python app.py<br/>

3. Running the Application:
Start the application:
<br>python app.py<br/>

Here's a View of the Actual Website

1)The homepage with a video upload button-

![1](https://github.com/user-attachments/assets/ab208e3a-0336-4c47-8a05-d0aa0e16a4b4)

![2](https://github.com/user-attachments/assets/d139e6a3-78e0-46d2-933e-94ff5695c1be)

![4](https://github.com/user-attachments/assets/d71d889f-24db-4882-bdc5-8a40b9519beb)


2)The results page displaying detected emotions and sentiment analysis output-

![7](https://github.com/user-attachments/assets/2e7cc899-f0f0-4f40-bd01-c1695820b927)

3) Service provided like Doctor Consultation/Recommendation-
   
![8](https://github.com/user-attachments/assets/8d8293ce-9bc0-4703-b5b7-991176ea8a6c)

![9](https://github.com/user-attachments/assets/b9be3cb6-93e5-4a85-813c-2a614040a273)

![10](https://github.com/user-attachments/assets/dcb245cc-5b3b-451d-956b-8d1cbbba53b0)


