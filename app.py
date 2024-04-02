import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
print(cv2.__version__)


st.header("Emotion Detection")




model = load_model('model\CNN_Model.h5')




label_mapping={5: 'sad',
 4: 'neutral',
 3: 'happy',
 0: 'angry',
 6: 'surprise',
 2: 'fear',
 1: 'disgust'}


# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = image[y:y+h, x:x+w]
        
        face = cv2.resize(face, (48, 48))  # Resize to match model input shape
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = face.astype(np.float32) / 255.0  # Normalize

        predicted=np.argmax(model.predict(face))
        
        ans=label_mapping[predicted]
        ans=ans.capitalize()
        
        text_size = cv2.getTextSize(ans, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(image, ans, (x + text_size[0] + 25, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    
    return image, len(faces),ans





uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.divider()
if uploaded_image is not None:
    classifier = cv2.CascadeClassifier('./harrcasscade/haarcascade_frontalface_default.xml')
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    detected_image, num_faces,ans = detect_faces(image)

    if num_faces is not None:
        st.image(detected_image, caption=f"Detected {num_faces} faces", use_column_width=True)

        if ans=="Happy":
            st.subheader(f"The Predicted Model is {ans} 😃")
        elif ans=="Sad":
            st.subheader(f"The Predicted Model is {ans} 😔")
        elif ans=="Angry":
            st.subheader(f"The Predicted Model is {ans} 😡")
        elif ans=="Surprise":
            st.subheader(f"The Predicted Model is {ans} 😮")
        elif ans=="Fear":
            st.subheader(f"The Predicted Model is {ans} 😨")
        elif ans=="Disgust":
            st.subheader(f"The Predicted Model is {ans} 🤢")
        elif ans=="Neutral":
            st.subheader(f"The Predicted Model is {ans} 😑")
    else:
        st.subheader("Image Has No Human Face")


