from flask import Flask,render_template,request,redirect,url_for
import cv2
import numpy as np
import cv2
import keras
from statistics import mode
from gtts import gTTS
from playsound import playsound

app = Flask(__name__)

letter = "a"
pred_word = ""
background = None

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    _, contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)




@app.route("/", methods=["GET","POST"])
def home():
    global letter
    if request.method == "POST":
        letter = request.form["letter"]
        letter = int(letter)
        model = keras.models.load_model(r"D:\Python 3\code\best_model_dataflair_alpha.h5")
        
        accumulated_weight = 0.5
        ROI_top = 100
        ROI_bottom = 300
        ROI_right = 150
        ROI_left = 350
        word_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}
        frames = 0
        i = 0
        mylist = []
        preds = []
        j = 0
        num_frames = 0
        cam = cv2.VideoCapture(0)
        while i < letter:
            ret, frame = cam.read()
            # filpping the frame to prevent inverted image of captured frame...
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            # ROI from the frame
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if num_frames < 70:
                cal_accum_avg(gray_frame, accumulated_weight)
                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            else: 
                # segmenting the hand region
                hand = segment_hand(gray_frame)
                cv2.putText(frame_copy,"Show letter " + str(i+1), (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)

                # Checking if we are able to detect the hand...
                if hand is not None:
                    thresholded, hand_segment = hand

                    # Drawing contours around hand segment
                    cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                    
                    cv2.imshow("Thresholded Hand Image", thresholded)
                    
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))

                    pred = model.predict(thresholded)
                    cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #print('a')
                    #print(mylist)
                    #cv2.putText(frame_copy, mylist[0], (172,47),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    if frames < 400:
                        mylist.append(word_dict[np.argmax(pred)])
                        frames = frames + 1
                    else:
                        preds.append(mylist)
                        mylist = []
                        frames = 0
                        i = i + 1

            # Draw ROI on frame_copy
            cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

            # incrementing the number of frames for tracking
            num_frames += 1

            # Display the frame with segmented hand
            cv2.putText(frame_copy, "Hand Gesture Recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
            cv2.imshow("Sign Detection", frame_copy)


            # Close windows with Esc
            k = cv2.waitKey(1) & 0xFF


            if k == 27:
                break

        # Release the camera and destroy all the windows
        cam.release()
        cv2.destroyAllWindows()
        j = 0
        global pred_word
        while j < letter:
            pred_word = pred_word + mode(preds[j])
            j = j + 1

        print(pred_word)
        output = gTTS(text=pred_word, lang='en', slow=False)
        output.save("output.mp3")

        playsound("output.mp3")

        return redirect(url_for("output"))
    else:
        return render_template("input.html")

@app.route("/out")
def output():
    print(letter)
    return render_template("output.html", letter = pred_word)

if __name__ == '__main__':
    app.run(debug=True)
