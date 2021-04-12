# Environment Variables.
import os
import datetime as dt

# GUI Packages.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
import time

# AWS Packages.
import boto3

# AWS Variables.
accessKeyID = os.environ["AWS_ACCESS_KEY_ID"]
secretAccessKey = os.environ["AWS_SECRET_ACCESS_KEY"]
s3BucketName = "jazimmer-eecs-605-module-08-fingers"
inputImageFileName = "hand.jpeg"
resultsDataFileName = "results.txt"

states = [5, 4, 3, 2, 1, 0]
current_state_num = 0

cell_number = None

def send_blastoff_text():

    global cell_number
    
    sns = boto3.client('sns',
        aws_access_key_id=accessKeyID,
        aws_secret_access_key=secretAccessKey,
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
        aws_access_key_id=accessKeyID,
        aws_secret_access_key=secretAccessKey,
        region_name = 'us-east-2'
    )

    client.download_file(s3BucketName, 'blastoff.jpg', 'blastoff.jpg')
    bl_img = mpimg.imread('blastoff.jpg')
    bl_imgplot = plt.imshow(bl_img)
    plt.show()
    return

def run_blastoff():
    send_blastoff_text()
    download_blastoff_image()
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
        print('State Retained...')

    return 

def parseAndShowResults(resultsDataFileName):

    with open(resultsDataFileName, "r") as results:
        # Extract prediction results.

        prediction = ''
        probability = ''
        for il, line in enumerate(results.readlines()):
            line = line.strip()
            if il == 0:
                prediction = line
                pred_num = int(line.split(':')[-1].strip())
            elif int(line.split(';')[0].split('=')[-1].strip()) == pred_num:
                probability = line.split(';')[-1].split('=')[-1].strip()
                break

        # Find the prediction value with the highest prediction value.

        # Display predicted value, prediction probability, and image of the hand-writtent digit that was classified.
        img = mpimg.imread(inputImageFileName)
        imgplot = plt.imshow(img, cmap='gray')
        plt.title(prediction+' — Probability: '+probability)
        plt.plot()
        plt.show()

    advance_prediction(pred_num)
    createDashBoard(cell_number)

## AWS Image Upload callback function and button ##

# Upload digit.png to S3 to produce the results.txt using lambda.
def awsImageUpload(buttonObject):
    
    client = boto3.client(
        's3',
        aws_access_key_id=accessKeyID,
        aws_secret_access_key=secretAccessKey,
        region_name = 'us-east-2'
    )
    
    upload_time = dt.datetime.now(dt.timezone(dt.timedelta()))
    response = client.upload_file(f'./{inputImageFileName}', s3BucketName, inputImageFileName)
    
    # Waiting and checking to see if the results.txt has been produced and placed in S3 from Lambda.
    while True:
        print('Waiting for result')
        time.sleep(awsProgressRefreshRateSlider.value)
        
        obj = client.list_objects(Bucket=s3BucketName)
        file_times = {_['Key']: _['LastModified'] for _ in obj['Contents'] if _['Key'] in [inputImageFileName,
                                                                                           resultsDataFileName
                                                                                          ]}
        
#         print(file_times)
        if inputImageFileName not in file_times or resultsDataFileName not in file_times:
            continue
        elif upload_time > file_times[inputImageFileName] or upload_time > file_times[resultsDataFileName]:
            continue
        
        break
    
    print('Prediction Complete! Downloading...')
    client.download_file(s3BucketName, resultsDataFileName, resultsDataFileName)
    print('Done!')
    
    # Removing input digit.png and output results.txtx from S3.
    print('Deleting files on server...')
    client.delete_objects(Delete={'Objects': [{'Key': _} for _ in file_times.keys()]}, Bucket=s3BucketName)
    print('Done!')
    
    # Display Results
    parseAndShowResults(resultsDataFileName)

    return client

## Image upload callback function and button ##

def selectimage2upload(imageData):
    
    # Due to the file structure, image file name needs to be
    # extracted to access the bytes data of the image.
#     imageFileName = list(imageData["new"].keys())[0]
#     imageFileName = imageData["new"][0]
    
    # Image bytes data.
#     imageBytesData = imageData["new"][imageFileName]["content"]
    imageBytesData = imageData["new"][0]
    
    # Writing image file to current directory with "inputImageFileName".
    with open(inputImageFileName, "wb") as imageFile:
        imageFile.write(imageBytesData)
    
    # Displaying uploaded image in GUI.
    display(widgets.Image(value=imageBytesData))
    
    # Showing AWS GUI Components after image is uploaded.
    display(awsProgressRefreshRateSlider)
    display(awsUploadButton)

def createDashBoard(phone_number = '+19548296800'):

    if len(phone_number) != 12:
        raise ValueError('Phone Number should follow the format "+1XXXXXXXXXX" where each X is a number 0-9')
    elif phone_number[0:2] != '+1':
        raise ValueError('+1 Country Code Required — US only supported')
    elif not str.isnumeric(phone_number[2:]):
        raise ValueError('Phone Number should be an integer after country code')

    # Allows the buttons to be accessed globally: Necessary
    # since some callback functions are dependent on these
    # widgets.
    global awsUploadButton
    global awsProgressRefreshRateSlider
    global cell_number

    if phone_number != cell_number:
        print('Texting Results to:', phone_number)

    cell_number = phone_number
    
    print('> Current State (i.e. Next Number to Show):', states[current_state_num])
    # AWS Image Upload Button.
#     button = widgets.Button(description='Upload')
    button = widgets.FileUpload(accept='.jpeg', multiple=False)
    button.observe(selectimage2upload, names='data')
    display(button)
    
    # AWS Progress Refresh Rate Selector.
    awsProgressRefreshRateSlider = widgets.FloatSlider(description='Refresh Rate', min=0.50, max=2.00, step=0.01, value=1.00)
    awsUploadButton = widgets.Button(description='Upload to AWS')
    awsUploadButton.on_click(awsImageUpload)
    
    # Display GUI.
    
    return button


# def dashBoardLoop():
#     # for i in range(5):
#     #     cdb = createDashBoard()

#     while True:
#         if not os.path.exists('./results.txt'):
#             cdb = createDashBoard()
#         else:
#             try:
#                 os.wait()
#             except:
#                 pass
#             os.remove('./results.txt')
#         time.sleep(10)


#     return cdb
