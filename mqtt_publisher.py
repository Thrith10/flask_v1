
import base64       
import cv2          
import numpy as np  
import paho.mqtt.client as mqtt  
import time        

# Define MQTT broker address (localhost = same machine)
MQTT_BROKER = "localhost"

# Define MQTT topics
CAPTURE_TOPIC = "capture/image"     # Topic to publish a command to capture an image
IMAGE_TOPIC = "image/stream"        # Topic to receive the image data

# Callback function that runs when an image message is received
def on_image_received(client, userdata, message):
    # Print which topic the image was received from
    print(f"Received image on topic '{message.topic}'")
    
    # Decode the base64-encoded image data from the MQTT message payload
    image_data64 = base64.b64decode(message.payload)
    
    # Convert the binary data into a NumPy array of type uint8
    np_arr = np.frombuffer(image_data64, np.uint8)
    
    # Decode the NumPy array into an OpenCV image (BGR format)
    frame_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # If image decoding was successful
    if frame_data is not None:
        # Display the received image in a window
        cv2.imshow("Received Image", frame_data)
        # Keep the window open for 1 second (1000 milliseconds)
        cv2.waitKey(1000)
        # Close the image window
        cv2.destroyAllWindows()
    else:
        # Show error if decoding failed
        print("Failed to decode received image.")

# Create an MQTT client instance named "PublisherViewer"
client = mqtt.Client("PublisherViewer")

# Attach the image received callback function to the client
client.on_message = on_image_received

# Connect the MQTT client to the broker at the specified address and port
client.connect(MQTT_BROKER, 1883)

# Subscribe to the topic that will receive image data
client.subscribe(IMAGE_TOPIC)

# Start the MQTT client loop in a background thread (for asynchronous message handling)
client.loop_start()

# Send a capture command to the publisher (camera or image source)
print("Sending capture command...")
client.publish(CAPTURE_TOPIC, "capture")

# Wait for 10 seconds to allow time for image to be received and processed
time.sleep(10)

# Stop the MQTT client loop
client.loop_stop()

# Disconnect the MQTT client from the broker
client.disconnect()