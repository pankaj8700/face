from flask import Flask, request, Response, render_template, send_file
from PIL import Image, ImageDraw
from mtcnn.mtcnn import MTCNN
import numpy as np
import io
import cv2
import os

app = Flask(__name__)
detector = MTCNN()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

def generate_frames():
    global out
    video = cv2.VideoCapture('temp.mp4')
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 0, 0), 2)
            # write the frame
            out.write(frame)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # Release everything if job is finished
    video.release()
    out.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.mp4'):
            file.save('temp.mp4')
            return render_template('index.htm')

        else:
            # existing image processing code here
            image = Image.open(file.stream)  # open image
            image = image.convert('RGB')  # convert image to RGB
            pixels = np.asarray(image)  # convert image to numpy array

            faces = detector.detect_faces(pixels)  # detect faces in the image

            # draw a rectangle around each face
            for face in faces:
                x, y, width, height = face['box']
                draw = ImageDraw.Draw(image)
                draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

            # convert PIL Image back to bytes
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='JPEG')
            byte_arr = byte_arr.getvalue()

            return send_file(io.BytesIO(byte_arr), mimetype='image/jpeg')

    return render_template('index.htm')

if __name__ == '__main__':
    app.run(debug=True)