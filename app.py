from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)  # 0 for default camera (you can specify other camera indexes or video files)

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dance')
def dance_page():
    return render_template('dance.html')

@app.route('/webcam_stream')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
