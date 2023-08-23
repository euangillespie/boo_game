"""
Game displaying a boo that responds to whether you're looking at the webcam.
Press q to quit.
"""
import cv2
import matplotlib.pyplot as plot
import math
import numpy


# IMAGE_PATH = "image.jpg"
# IMAGE_PATH = "image-hiding.jpg" # This one detects a face on my knuckle
# IMAGE_PATH = "image-hiding-2.jpg"
# IMAGE_PATH = "image-hiding-3.jpg"
# IMAGE_PATH = "image-noone.jpg"
# IMAGE_PATH = "image-looking-away.jpg"


HAARCASCADE_PATH = "haarcascade_frontalface_alt2.xml"

LBF_MODEL_PATH = "lbfmodel.yaml"

CHARACTER_IMAGE_PATH_MOVING = "images/boo.jpg"
CHARACTER_IMAGE_PATH_HIDING = "images/boo-hiding.jpg"
# CHARACTER_IMAGE_PATH_MOVING = "images/king-boo.jpg"
# CHARACTER_IMAGE_PATH_HIDING = "images/king-boo-hiding-2.jpg"

CHARACTER_STARTING_WIDTH = 100
CHARACTER_STARTING_HEIGHT = 100
CHARACTER_STARTING_POSITION_X = 300
CHARACTER_STARTING_POSITION_Y = 250

NUMBER_OF_FRAMES_BEFORE_CHANGE = 5

detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

def image_has_face(image):
    faces = detector.detectMultiScale(image)
    return len(faces) > 0


def resize_image(image, width, height):
    return cv2.resize(image, (width, height)) 


def overlay_image(background, foreground, overlay_position_x, overlay_position_y):
    # TODO: transparency? https://stackoverflow.com/a/41335241
    foreground_width = foreground.shape[1]
    foreground_height = foreground.shape[0]
    overlay_end_x = overlay_position_x + foreground_width
    overlay_end_y = overlay_position_y + foreground_height
    background[overlay_position_y:overlay_end_y, overlay_position_x:overlay_end_x,:] = foreground
    return background


character_image_moving = cv2.imread(CHARACTER_IMAGE_PATH_MOVING)
character_image_hiding = cv2.imread(CHARACTER_IMAGE_PATH_HIDING)


capture = cv2.VideoCapture(0)
print("Running, press q to quit...")
is_hiding = False
previous_n_frame_states = []
frames_advanced = 0

while True:
    frame_captured, frame = capture.read()
    if not frame_captured:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Boo hides if it's being looked at, i.e. if someone is looking at the webcam
    character_should_hide = image_has_face(frame)
    previous_n_frame_states.append(character_should_hide)
    if not is_hiding:
        frames_advanced += 1
    if len(previous_n_frame_states) >= NUMBER_OF_FRAMES_BEFORE_CHANGE:
        if len(previous_n_frame_states) > NUMBER_OF_FRAMES_BEFORE_CHANGE:
            previous_n_frame_states = previous_n_frame_states[1:]
        # If we've had N or more frames showing the opposite of our current hiding state,
        # switch state.
        if is_hiding:
            if all([not state for state in previous_n_frame_states]):
                is_hiding = False
        else:
            if all(previous_n_frame_states):
                is_hiding = True
                frames_advanced = max(0, frames_advanced - 30)

    if is_hiding:
        character_image = character_image_hiding
    else:
        character_image = character_image_moving

    if frames_advanced >= 175:
        # Reset back to the start after it reaches maximum size
        frames_advanced = 0

    character_size_multiplier = 1 + min(4.5, frames_advanced * 0.02)
    print('aaa', frames_advanced, character_size_multiplier)
    character_width = int(CHARACTER_STARTING_WIDTH * character_size_multiplier)
    character_height = int(CHARACTER_STARTING_HEIGHT * character_size_multiplier)

    resized_character_image = resize_image(
        character_image,
        character_width,
        character_height,
    )

    frame_with_overlay = overlay_image(
        frame,
        resized_character_image,
        int(CHARACTER_STARTING_POSITION_X - (character_width / 2)),
        int(CHARACTER_STARTING_POSITION_Y - (character_height / 2)),
    )
    cv2.imshow('frame', frame_with_overlay)

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