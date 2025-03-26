import cv2                         
import paho.mqtt.client as mqtt   
import numpy as np                
import base64                    

# MQTT broker address 
MQTT_BROKER = "localhost"

# MQTT topic names
CAPTURE_TOPIC = "capture/image"   
IMAGE_TOPIC = "image/stream"      

# Callback function that triggers when a message is received on a subscribed topic
def on_capture_command(client, message):
    print(f"Received capture command on topic '{message.topic}': {message.payload.decode()}")

    # Capture image from the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    # Read a single frame from the webcam
    ret, frame_data = cap.read()      
    # Release the webcam resource
    cap.release()                

    if ret:
        # Encode the captured frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame_data)
        
        # Convert the image bytes to a base64-encoded string
        image_base64 = base64.b64encode(buffer).decode()

        # Publish the base64-encoded image to the IMAGE_TOPIC
        client.publish(IMAGE_TOPIC, image_base64)
        print("Image captured and published.")
    else:
        print("Failed to capture image.")

# Create MQTT client with a name "SubscriberCamera"
client = mqtt.Client("SubscriberCamera")

# Set the callback function for incoming messages
client.on_message = on_capture_command

# Connect the MQTT client to the broker on default port 1883
client.connect(MQTT_BROKER, 1883)

# Subscribe to the topic to listen for image capture commands
client.subscribe(CAPTURE_TOPIC)

# Start an infinite loop to keep the client running and listening for messages
client.loop_forever()
