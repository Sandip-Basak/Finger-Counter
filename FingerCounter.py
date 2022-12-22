import cv2
import time
import os
import HandsTrackingModule as hand


def main():
    cap = cv2.VideoCapture(0)

    # Getting the images of hands
    images = os.listdir("Fingers")
    overlayImages = []
    for i in images:
        resizeImg = cv2.resize(cv2.imread(f"Fingers/{i}"), (200, 250))
        overlayImages.append(resizeImg)

    pTime = 0

    # List of the fingertips Landmarks
    tipList = [4, 8, 12, 16, 20]

    detector = hand.HandDetector(min_detection_confidence=0.75, max_hands=1)

    while True:
        success, img = cap.read()

        # Drawing landmarks and connections
        detector.DrawHands(img, draw=False)

        # Getting all the position of the landmarks
        lmList = detector.givePosition(img=img, draw=False)

        # check that the returned list is empty or not
        if len(lmList) != 0:

            countList = []

            # Checking the thumb is open or closed using x axis
            if lmList[tipList[0]][1] > lmList[tipList[0]-1][1]:
                countList.append(1)
            else:
                countList.append(0)

            # Checking the fingers are open or closed using y axis
            for i in range(1, 5):
                if lmList[tipList[i]][2] < lmList[tipList[i]-2][2]:
                    countList.append(1)
                else:
                    countList.append(0)

            # Counting the total open fingers
            totalFingers = countList.count(1)

            # Setting the particular hand image on the frame
            h, w, c = overlayImages[totalFingers-1].shape
            img[0:h, 0:w] = overlayImages[totalFingers-1]

            # Drawing the rectangle and the count of fingers
            cv2.rectangle(img, (0, 250), (199, 500), (0, 225, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (50, 425), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 5)

        # Calculating FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS-{int(fps)}", (540, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)

        # Showing the Image
        cv2.imshow("Finger Counter", img)

        # On 'q' press the loop breaks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
