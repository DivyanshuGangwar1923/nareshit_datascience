from ultralytics import YOLO

#Load a pre-trained YOLOv8n model
model=YOLO("yolov8n.pt",'v8')


#Predict on a image
detection_output=model.predict(source=r"C:\Users\divya\Downloads\WhatsApp Image 2025-08-12 at 8.37.04 AM20250818161424357.jpeg",conf=0.25,save=True)

#Display tensor array

print(detection_output)

# Display numpy array

print(detection_output[0].numpy())