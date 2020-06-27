
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='C:\\Temp\\DAG-cda728af956d.json'

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('image.jpg')


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

print('*** Labels: ***')
for label in labels:
    print(label.description)

print('*** Faces: ***')
for face in faces:
    print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
    print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
    print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
        for vertex in face.bounding_poly.vertices])
    print('face bounds: {}'.format(','.join(vertices)))

print('*** Landmarks: ***')
for landmark in landmarks:
    print(landmark.description)
    for location in landmark.locations:
        lat_lng = location.lat_lng
        print('Latitude {}'.format(lat_lng.latitude))
        print('Longitude {}'.format(lat_lng.longitude))

print('*** Logos: ***')
for logo in logos:
    print(logo.description)

print('*** Annotations: ***')
if annotations.best_guess_labels:
    for label in annotations.best_guess_labels:
        print('\nBest guess label: {}'.format(label.label))

if annotations.pages_with_matching_images:
    print('\n{} Pages with matching images found:'.format(
        len(annotations.pages_with_matching_images)))

    for page in annotations.pages_with_matching_images:
        print('\n\tPage url   : {}'.format(page.url))

        if page.full_matching_images:
            print('\t{} Full Matches found: '.format(
                len(page.full_matching_images)))

            for image in page.full_matching_images:
                print('\t\tImage url  : {}'.format(image.url))

        if page.partial_matching_images:
            print('\t{} Partial Matches found: '.format(
                len(page.partial_matching_images)))

            for image in page.partial_matching_images:
                print('\t\tImage url  : {}'.format(image.url))

if annotations.web_entities:
    print('\n{} Web entities found: '.format(
        len(annotations.web_entities)))

    for entity in annotations.web_entities:
        print('\n\tScore      : {}'.format(entity.score))
        print(u'\tDescription: {}'.format(entity.description))

if annotations.visually_similar_images:
    print('\n{} visually similar images found:\n'.format(
        len(annotations.visually_similar_images)))

    for image in annotations.visually_similar_images:
        print('\tImage url    : {}'.format(image.url))

