"""
Game displaying a boo that responds to whether you're looking at the webcam.
Press q to quit.
"""
import cv2
import matplotlib.pyplot as plot


# IMAGE_PATH = "image.jpg"
IMAGE_PATH = "image-hiding.jpg" # This one detects a face on my knuckle
# IMAGE_PATH = "image-hiding-2.jpg"
# IMAGE_PATH = "image-hiding-3.jpg"
# IMAGE_PATH = "image-noone.jpg"
# IMAGE_PATH = "image-looking-away.jpg"


HAARCASCADE_PATH = "haarcascade_frontalface_alt2.xml"

LBF_MODEL_PATH = "lbfmodel.yaml"

detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

def image_has_face(image):
    faces = detector.detectMultiScale(image)
    return len(faces) > 0



capture = cv2.VideoCapture(0)
print("Running, press q to quit...")
while True:
 frame_captured, frame = capture.read()
 if not frame_captured:
    print("Can't receive frame (stream end?). Exiting ...")
    break
 cv2.imshow('frame', frame)
 print("Has face", image_has_face(frame))
 if cv2.waitKey(1) == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()

# image = cv2.imread(IMAGE_PATH)

# faces = detector.detectMultiScale(image)
# print("Faces:\n", faces)

# if len(faces) and faces.any():
#     # We've found the faces, now find facial geometry to see if there's eyes.
#     # The face detector will detect e.g. backs of heads, so we do need to
#     # check for features.
#     landmark_detector  = cv2.face.createFacemarkLBF()
#     landmark_detector.loadModel(LBF_MODEL_PATH)

#     xxx, landmarks = landmark_detector.fit(image, faces)
#     # TODO: What is this matching? It's seem to be basically anything.
#     # print(xxx, landmarks)

#     # Testing: display faces
#     # for face in faces:
#     #     # save the coordinates in x, y, w, d variables
#     #     (x,y,w,d) = face
#     #     # Draw a white coloured rectangle around each face using the face's coordinates
#     #     # on the "image" with the thickness of 2 
#     #     cv2.rectangle(image,(x,y),(x+w, y+d),(255, 255, 255), 2)

#     # Testing: display features
#     for landmark in landmarks:
#         for x,y in landmark[0]:
#             # display landmarks on "image"
#             # with white colour in BGR and thickness 1
#             cv2.circle(image, (int(x),int(y)), 1, (255, 255, 255), 1)

#     print("DISPLAYING")
#     plot.axis("off")
#     plot.imshow(image)
#     plot.title('Face Detection')
#     plot.show()
# else:
#     print("No faces")