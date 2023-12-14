from flask import Flask, Response, render_template, request, jsonify, send_from_directory
import cv2
import random, time
import sys
sys.path.append("src")
from src.dance import DanceManager, sse_messages
app = Flask(__name__)

pattern_dance_path = None
video_capture = cv2.VideoCapture(0)  # 0 for default camera (you can specify other camera indexes or video files)

dance_manager = DanceManager(video_capture)

message_time = 3

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
        dance_manager.compare_dances(pattern_dance_path)
    if message == "!VIDEO_END":
        dance_manager.set_flag_is_video_being_played(False)
    print(f"Received message from the client: {message}")
    # Perform any additional actions you need here
    return jsonify(success=True)

@app.route('/get_dance_name', methods=['POST'])
def get_dance_name():
    data = request.get_json()
    clicked_item = data.get('clickedItem')
    global pattern_dance_path
    pattern_dance_path = f"static/data/pattern_dance_data/{clicked_item}.csv"
    return jsonify(success=True)


@app.route('/point_stream')
def sse_stream():
    def generate():
        while True:
            if sse_messages:
                yield f"data: {sse_messages.pop(0)}\n\n"
            time.sleep(3)

    return Response(generate(), content_type='text/event-stream')

@app.route('/webcam_stream')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibration_ok_message')
def calibration_ok_msg():
    dance_manager.set_flag_is_camera_checked(False)
    dance_manager.check_camera(3)
    return Response(f"data: !CALIBRATION_OK\n\n", content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)