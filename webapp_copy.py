from flask import Flask,render_template,request,redirect,url_for
from functions import hand_gesture_recognizer,digit_recognizer

app = Flask(__name__)

letter = "a"
pred_word = ""
background = None

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        if request.form["submit"] == "Word Recognizer":
            return redirect(url_for("word"))
        elif request.form["submit"] == "Digit Recognizer":
            digit_recognizer()
            return render_template("base.html")    
    else:
        return render_template("base.html")

@app.route("/word",  methods=["GET","POST"])
def word():
    global letter
    global pred_word
    if request.method == "POST":
        letter = int(request.form["letter"])
        pred_word = hand_gesture_recognizer(letter)
        return redirect(url_for("output"))
    else:
        return render_template("input.html")


@app.route("/out")
def output():
    print(letter)
    return render_template("output.html", letter = pred_word)



if __name__ == '__main__':
    app.run(debug=True)
