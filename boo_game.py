"""
Game displaying a boo that responds to whether you're looking at the webcam.
Press q to quit.
"""
import cv2
import matplotlib.pyplot as plot
import math
import numpy
from playsound import playsound


HAARCASCADE_PATH = "haarcascade_frontalface_alt2.xml"

CHARACTER_IMAGE_PATH_MOVING = "images/boo-transparent.png"
CHARACTER_IMAGE_PATH_HIDING = "images/boo-hiding-transparent.png"

LAUGH_SOUND = "sounds/boo-laugh.mp3"

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


def overlay_image(background, foreground, overlay_position_x, overlay_position_y, alpha=1):
    # Based on https://www.geeksforgeeks.org/transparent-overlays-with-python-opencv/
    # This is pretty artifact-y with the images I've got. Something based on
    # https://github.com/opencv/opencv/issues/20780 would probably do better.
    overlay_image = foreground
    h, w = overlay_image.shape[:2]
    
    # Create a new numpy array
    shapes = numpy.zeros_like(background, numpy.uint8)
    
    # Put the overlay in the desired position
    shapes[overlay_position_y:overlay_position_y+h, overlay_position_x:overlay_position_x+w] = overlay_image

    # Change this into bool to use it as mask
    mask = shapes.astype(bool)

    # Create the overlay
    background[mask] = cv2.addWeighted(background, 1 - alpha, shapes, alpha, 0)[mask]
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
                # Send the boo back slightly each time it hides.
                frames_advanced = max(0, frames_advanced - 30)

    if is_hiding:
        character_image = character_image_hiding
    else:
        character_image = character_image_moving

    alpha = 1
    if frames_advanced > 150:
        # Start fading when the boo gets close.
        alpha = max(0, 1 - ((frames_advanced - 150) / 25))
    if frames_advanced == 160:
        # Mid-way through fading, play the sound.
        playsound(LAUGH_SOUND, False)
    if frames_advanced >= 190:
        # Reset back to the start after it reaches maximum size and finishes fading.
        frames_advanced = 0

    character_size_multiplier = 1 + min(3.5, frames_advanced * 0.02)
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
        alpha=alpha,
    )
    cv2.imshow('frame', frame_with_overlay)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
