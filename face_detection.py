import face_recognition
import cv2

# Read the input image
img = cv2.imread('amman_large.jpeg')

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read the haarcascade to detect the faces in an image
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
print('Number of detected faces:', len(faces))

# loop over all detected faces
if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces):
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = img[y:y + h, x:x + w]
        cv2.imshow("Cropped Face", face)
        cv2.imwrite(f'face{i}.jpg', face)
        print(f"face{i}.jpg is saved")

# display the image with detected faces
cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print('p1 finished')
#
known_image = face_recognition.load_image_file("amman_input.jpeg")
present = False
for i in range(len(faces)):
    unknown_image = face_recognition.load_image_file(f"face{i}.jpg")
    biden_encoding = face_recognition.face_encodings(known_image, model="large")[0]
    if len(unknown_encoding) > 0:
        unknown_encoding = face_recognition.face_encodings(unknown_image, model="large")[0]
    face_locations = face_recognition.face_locations(unknown_image, model="cnn")
    print(face_locations)
    # print(unknown_encoding)
    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    present = present or results[0]
print(present)
print('done')
from deepface import DeepFace
# import face_recognition
#
# known_image = face_recognition.load_image_file("amman_input.jpeg")
# unknown_image = face_recognition.load_image_file("amman_group.jpeg")
#
# biden_encoding = face_recognition.face_encodings(known_image, model="large")[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image, model="large")[0]
#
# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# print(results)
# face_locations = face_recognition.face_locations(unknown_image)
# print(face_locations)

# def compare(img1, img2):
#     resp = DeepFace.verify(img1, img2)
#     print(resp["verified"])
#
#
# for face in face_locations:
#     top, right, bottom, left = face
#     face_img = unknown_image[top:bottom, left:right]
#
#     face_recognition.compare_faces(known_image, face_img)


# import cv2
#
# # Read the input image
# img = cv2.imread('amman_group.jpeg')
#
# # convert to grayscale of each frames
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # read the haarcascade to detect the faces in an image
# face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
#
# # detects faces in the input image
# faces = face_cascade.detectMultiScale(gray, 1.3, 4)
# print('Number of detected faces:', len(faces))
#
# # loop over all detected faces
# if len(faces) > 0:
#     for i, (x, y, w, h) in enumerate(faces):
#         # To draw a rectangle in a face
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
#         face = img[y:y + h, x:x + w]
#         cv2.imshow("Cropped Face", face)
#         cv2.imwrite(f'face{i}.jpg', face)
#         print(f"face{i}.jpg is saved")
#
# # display the image with detected faces
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
