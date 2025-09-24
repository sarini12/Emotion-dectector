
 
import cv2
from deepface import DeepFace
from datetime import datetime
from collections import Counter
import numpy as np
                
cap = cv2.VideoCapture(0)

print("Starting real-time emotion detection...")
print("Detected emotions: happy, sad, angry, fear, surprise, disgust, neutral")
print("Press 'q' to quit")


emotion_history = []

frame_count = 0

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    
    frame = cv2.resize(frame, (640, 480))
    
    try:
        
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
       
        if isinstance(result, dict):  
            result = [result] 
        
        
        frame_count += 1
        
        
        detection_status = "NO FACES DETECTED"
        color = (0, 255, 0)                  
        top_emotions = []                 
       
        if len(result) > 0:
            detection_status = "EMOTIONS DETECTED!"
            color = (0, 255, 255)        
            
            
            for face_result in result:
                
                if 'region' in face_result:
                    x, y, w, h = face_result['region']['x'], face_result['region']['y'], \
                                 face_result['region']['w'], face_result['region']['h']
                    
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                   
                    x, y, w, h = 10, 100, 200, 50
                
                
                emotions = face_result['emotion']
                
                
                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion]
                
               
                top_emotions.append(top_emotion)
                
               
                label = f'{top_emotion}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
  
        if top_emotions:
            emotion_history.append(top_emotions[0])
        else:
            emotion_history.append('none')
        
       
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Emotions - {top_emotions}")
        
        
        cv2.putText(frame, detection_status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    
        cv2.putText(frame, f'Frame: {frame_count}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        if top_emotions:
            most_common = Counter(top_emotions).most_common(1)[0][0]
            cv2.putText(frame, f'Top Emotion: {most_common}', (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
      



        cv2.imshow('Real-time Emotion Detection (DeepFace)', frame)
        
    except Exception as e:
       
        frame_count += 1
        cv2.putText(frame, "Detection Error - Retrying...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Real-time Emotion Detection (DeepFace)', frame)
        print(f"Frame {frame_count}: Error - {str(e)}")
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()


valid_emotions = [e for e in emotion_history if e != 'none']


most_common_emotion = Counter(valid_emotions).most_common(1)[0][0] if valid_emotions else 'none'


detection_rate = len(valid_emotions) / len(emotion_history) * 100 if emotion_history else 0


print("\n=== REAL-TIME EMOTION DETECTION SUMMARY (DeepFace) ===")
print(f"Total frames processed: {frame_count}")
print(f"Frames with detected faces: {len(valid_emotions)} ({detection_rate:.1f}%)")
print(f"Most common emotion: {most_common_emotion}")
print(f"Final status: {'SUCCESSFUL' if valid_emotions else 'NO DETECTIONS'}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

