
def detect_labels(path):
    """Detects labels in the file."""

def detect_faces(path):
    """Detects faces in an image."""

def detect_landmarks(path):
    """Detects landmarks in the file."""

def detect_logos(path):
    """Detects logos in the file."""

def detect_web(path):
    """Detects web annotations given an image."""

import os
import io


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='/Users/ashvin/workspace/API-integrations/credentials.json'

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('taj.jpg')


# Names of likelihood from google.cloud.vision.enums
likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

response_faces = client.face_detection(image=image)
faces = response_faces.face_annotations

response_landmark = client.face_detection(image=image)
landmarks = response_landmark.landmark_annotations

response_logo = client.face_detection(image=image)
logos = response_logo.logo_annotations


annotations = response.web_detection

data = {}

#====label===========
label_list = []
for label in labels:
    label_list.append(label.description)
if label_list:
    data['Labels'] = label_list


#======Faces=========
face_list = []
for face in faces:
    face_list.append('anger: {}'.format(likelihood_name[face.anger_likelihood]))
    face_list.append('joy: {}'.format(likelihood_name[face.joy_likelihood]))
    face_list.append('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
        for vertex in face.bounding_poly.vertices])
    face_list.append('face bounds: {}'.format(','.join(vertices)))
if face_list:
    data['Faces'] = face_list


#======Landmarks=========
landmark_list = []
for landmark in landmarks:
    print(landmark.description)
    for location in landmark.locations:
        lat_lng = location.lat_lng
        landmark_list.apeend('Latitude {}'.format(lat_lng.latitude))
        landmark_list.apeend('Longitude {}'.format(lat_lng.longitude))
if landmark_list:
    data['Landmarks'] = landmark_list



#=====logo=====
logo_list = []
for logo in logos:
    logo_list.append(logo.description)
if logo_list:
    data['Logos'] = logo_list

#=======Annotations=====
annotation_list = []
if annotations.best_guess_labels:
    for label in annotations.best_guess_labels:
        annotation_list.apeend('\nBest guess label: {}'.format(label.label))
if annotation_list:
    data['Annotations'] = annotation_list
print ">>>>>>>", data

