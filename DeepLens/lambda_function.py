#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
import os
import json
import time
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk
import boto3
from utils import LocalDisplay

import sys
import onnxruntime as rt

states = [5, 4, 3, 2, 1, 0]
current_state_num = 0
cell_number = '+1XXXXXXXXXX' # TODO: Change to your personal cellphone number

local_display = None

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)



def send_blastoff_text():

    global cell_number

    
    sns = boto3.client('sns'
        aws_access_key_id='_________', # TODO: Add AWS Access Key
        aws_secret_access_key='__________', # TODO: Add AWS Access Key
        region_name = 'us-east-2'
        )

    smsattrs = {
        'AWS.SNS.SMS.SenderID': { 'DataType': 'String', 'StringValue': 'TestSender' },
        'AWS.SNS.SMS.SMSType': { 'DataType': 'String', 'StringValue': 'Transactional'}
    }

    number = cell_number

    sns.publish(PhoneNumber = number,
                Message='BLASTOFF!!!',
                MessageAttributes = smsattrs
    )

    print(f'BLASTOFF!!! (Text Sent to: {cell_number})')

    return

def download_blastoff_image():
    client = boto3.client(
        's3',
        aws_access_key_id='_________', # TODO: Add AWS Access Key
        aws_secret_access_key='__________', # TODO: Add AWS Access Key
        region_name = 'us-east-2'
    )
    client.download_file('jazimmer-eecs-605-module-08-fingers-dl', 'blastoff.jpg', '/tmp/blastoff.jpg')

    return

def display_blastoff_image():
    global local_display

    blastoff = cv2.imread('/tmp/blastoff.jpg')
    local_display.set_frame_data(blastoff)
    time.sleep(5)

    return

def run_blastoff():
    send_blastoff_text()
    display_blastoff_image()
    return

def advance_prediction(observation):
    global current_state_num

    if observation == states[current_state_num]:
        print('State Advances...')

        current_state_num += 1
        if current_state_num >= len(states) - 1:
            run_blastoff()

        current_state_num = current_state_num % (len(states) - 1)
    else:
        pass

    return 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(image):
    # Convert image to gray-scale and invert the pixels.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    
    # TODO: Crop the center square of the given image/frame.
    # Image/frame dimension is around 16:9 ratio, similar to a movie screen.
    # Crop the image/frame to get the center square of it.

    init_height = gray.shape[-2]
    init_width = gray.shape[-1]

    final_width_start = (init_width - init_height)//2
    final_width_end = init_width - (init_width - init_height)//2

    crop_img = gray[...,final_width_start:final_width_end]
    
    # Resizing to make it compatible with the model input size.

    resized_img = cv2.resize(crop_img, (128,128)).astype(np.float32)#/255
    img = np.reshape(resized_img, (1,128,128,1))
    # resized_img = cv2.resize(crop_img, (28,28)).astype(np.float32)/255
    # img = np.reshape(resized_img, (1,1,28,28))
    return img


def makeInferences(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    scores = pred_onx
    # scores = softmax(pred_onx)
    return scores


def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return

# Create an IoT client for sending to messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ["AWS_IOT_THING_NAME"])

def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""    
    try:
        download_blastoff_image()

        model_directory = "/opt/awscam/artifacts/"
        # model_name = "mnist-8" # onnx-model
        model_name = "fingerModel.onnx" # onnx-model

        global local_display

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # When the ONNX model is imported via DeepLens console, the model is copied
        # to the AWS DeepLens device, which is located in the "/opt/awscam/artifacts/".
        model_file_path = os.path.join(model_directory, model_name)
        sess = rt.InferenceSession(model_file_path)
        
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
                
            # Preprocess the frame to crop it into a square and
            # resize it to make it the same size as the model's input size.
            input_img = preprocess(frame)

            # Inference.
            inferences = makeInferences(sess, input_img)
            inference = np.argmax(inferences)

            advance_prediction(inference)

            # TODO: Add the label of predicted digit to the frame used by local display.
            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
            # for more information about the cv2.putText method.
            # Method signature: image, text, origin, font face, font scale, color, and thickness 
            # cv2.putText()
            cv2.putText(frame, str(inference), (20,120), cv2.FONT_HERSHEY_COMPLEX, 5, (243, 252, 61), 4)

            cv2.putText(frame, 'Awaiting: '+str(states[current_state_num]), (1700,120), cv2.FONT_HERSHEY_COMPLEX, 5, (243, 252, 61), 4)
            
            # 255, 0, 0
            # 61, 252, 243

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
 
            # Outputting the result logs as "MQTT messages" to AWS IoT.
            cloud_output = {}
            cloud_output["scores"] = inferences.tolist()
            print(inference, cloud_output)
            print(input_img.shape, inferences.shape)

    except Exception as ex:
        # Outputting error logs as "MQTT messages" to AWS IoT.
        print('Error in lambda {}'.format(ex))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("error details:" + str(exc_type) + str(fname) + str(exc_tb.tb_lineno))

infinite_infer_run()
