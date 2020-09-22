#-----------------------------------------------------------------------------#
import sys

import numpy as np
import cv2

#-----------------------------------------------------------------------------#
def gray_video(name):
    cap = cv2.VideoCapture(name)
    ret = True
    while (ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('Gray Frame',gray)
        if cv2.waitKey(33) == ord('q'):
            ret = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return 1

#-----------------------------------------------------------------------------#
def estimates_background(name):
    # Display median frame
    cv2.imshow('Estimates Background Frame', get_median(name))
    cv2.waitKey(0)
    return 1

#-----------------------------------------------------------------------------#
def frame_differencing(name):

    cap = cv2.VideoCapture(name)
    # Reset frame number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Convert background to grayscale
    grayMedianFrame = cv2.cvtColor(get_median(name), cv2.COLOR_BGR2GRAY)

    # Loop over all frames
    ret = True
    while(ret):

        # Read frame
        ret, frame = cap.read()
        # Convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate absolute difference of current frame and
        # the median frame
        dframe = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
        # Display image
        cv2.imshow('frame', dframe)
        if cv2.waitKey(33) == ord('q'):
            ret = False
    # Release video object
    cap.release()
    return 1

#-----------------------------------------------------------------------------#
def get_median(name):
    cap = cv2.VideoCapture(name)

    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return medianFrame

#-----------------------------------------------------------------------------#
def menu(name):
    print("************MAIN MENU**************")
    #time.sleep(1)
    print()

    choice = input("""
                      A: Gray Video
                      B: Estimates of the Background
                      C: Frame Differencing
                      D: Exit

                      Please enter your choice: """)

    if choice == "A" or choice =="a":
        if gray_video(name) == 1:
            menu(name)
    elif choice == "B" or choice =="b":
        if estimates_background(name):
            menu(name)
    elif choice == "C" or choice =="c":
        if frame_differencing(name):
            menu(name)
    elif choice=="D" or choice=="d":
        return
    else:
        print("You must only select either A,B,C or D.")
        print("Please try again")
        menu(name)

#-----------------------------------------------------------------------------#


menu('driving.mp4')
