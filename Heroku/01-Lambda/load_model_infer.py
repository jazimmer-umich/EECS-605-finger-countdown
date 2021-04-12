import sys
sys.path.append("./packages")
import os
import numpy as np
from datetime import datetime
import boto3
import onnxruntime as rt
from PIL import Image, ImageOps
from scipy.special import softmax

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess(image):
    height, width = image.size
    presquared = (height == width)

    min_dim = min(width, height)

    crp_height = (height - min_dim)//2
    crp_width = (width - min_dim)//2

    top = crp_height
    bot = height - crp_height
    left = crp_width
    right = width - crp_width


    # image = Image.crop((image.size[0]-3224, 0, image.size[0]-200, image.size[1]))
    image = image.crop((top, left, bot, right))
    
    smaller_image = image.resize((int(128), int(128)), Image.ANTIALIAS)
    numpy_smaller_image = np.asarray(smaller_image)
    numpy_smaller_image = numpy_smaller_image.astype("float32")# / 255

    # if not presquared:
    #     numpy_smaller_image = np.flip(np.transpose(numpy_smaller_image, [1, 0]),-1)

    numpy_smaller_image = np.expand_dims(np.expand_dims(numpy_smaller_image, axis=0),axis=-1)

    # processed_image = np.reshape(numpy_smaller_image,(1,3,128,128))
    # processed_image = np.reshape(numpy_smaller_image,(1,3,128,128))

    processed_image = numpy_smaller_image
    return processed_image

def makeInference(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    # scores = softmax(pred_onx)
    scores = pred_onx
    return scores


def lambda_handler(event, context):

    s3_bucket_name = "jazimmer-eecs-605-module-08-fingers"
    lambda_tmp_directory = "/tmp"
    model_file_name = "finger_model.onnx"
    # input_file_name = "hand.png"
    input_file_name = "hand.jpeg"
    output_file_name = "results.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    try:
        # Download test image and model from S3.
        client = boto3.client('s3')
        client.download_file(s3_bucket_name, input_file_name, os.path.join(lambda_tmp_directory, input_file_name))
        client.download_file(s3_bucket_name, model_file_name, os.path.join(lambda_tmp_directory, model_file_name))
    except:
        pass

    # Import input image in grayscale and preprocess it.
    image = Image.open(os.path.join(lambda_tmp_directory, input_file_name)).convert("L")
    processed_image = preprocess(image)

    # Make inference using the ONNX model.
    sess = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_file_name))
    inferences = makeInference(sess, processed_image)

    # Output probabilities in an output file.
    f = open(os.path.join(lambda_tmp_directory, output_file_name), "w+")
    f.write("Predicted: %d \n" % np.argmax(inferences))
    for i in range(6):
        f.write("class=%s ; probability=%f \n" % (i, inferences[0][i]))
    f.close()


    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    try:
        # Upload the output file to the S3 bucket.
        client.upload_file(os.path.join(lambda_tmp_directory, output_file_name), s3_bucket_name, output_file_name)
    except e:
        print('Error:', e)
        pass

# #Uncomment to run locally.
# lambda_handler(None, None)