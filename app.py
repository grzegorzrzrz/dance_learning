from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import cv2
import random, time
from src.dance import DanceManager
app = Flask(__name__)

pattern_dance_path = "static/pattern.csv" #@TODO Remove it

video_capture = cv2.VideoCapture(0)  # 0 for default camera (you can specify other camera indexes or video files)

dance_manager = DanceManager(pattern_dance_path, video_capture)

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

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/dance')
def dance_page():
    return render_template('dance.html')

@app.route('/calibrate')
def calibrate_page():
    return render_template('calibrate.html')

@app.route('/video_message', methods=['POST'])
def video_started():
    data = request.get_json()
    message = data.get('message', 'No message received')
    if message == "!VIDEO_START":
        dance_manager.compare_dances()
    if message == "!VIDEO_END":
        dance_manager.set_flag_is_video_being_played(False)
        dance_manager.save_actual_dance("DEBUG.csv")
    print(f"Received message from the client: {message}")
    # Perform any additional actions you need here
    return jsonify(success=True)

def generate_messages():
    for _ in range (7):
        # Send the current message to the client
        yield f"data: {random.choice(['good', 'bad', 'excellent'])}\n\n"
        time.sleep(5)
        # Sleep for 5 seconds

@app.route('/point_stream')
def stream():
    # Use the SSE MIME type
    return Response(generate_messages(), content_type='text/event-stream')

@app.route('/webcam_stream')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibration_message')
def calibration():
    # if calibration_check(): or something like that
    return Response(f"data: !CALIBRATION_OK\n\n", content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
