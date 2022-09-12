
'''
הפונקציה הראשית:
מקבלת מריאקט תמונה של תווים
שולחת לפוקציה של חלוקה לשורות

split_to_lines(img)

שולחת לפונקציה של זיהוי קווי חמשה וזיהוי אובייקטים
lines_detection(line_img)
object_detection(line_img)

מודל

make_music(right_hand_notes,right_hand_duration,left_hand_notes,left_hand_duration,.....)

return song (url) to the react
'''

from flask import Flask, request, jsonify, json,session, send_file
from flask_cors import CORS
import Object_recognition
import asyncio

loop = asyncio.get_event_loop()

app = Flask(__name__)
CORS(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.route('/upload', methods=['POST'])
def UploadImage():
    image = request.files['image']
    print(image.filename)
    # image.save(f'Piano_sheets/{image.filename}')
    imgUrl=f'Piano_sheets/{image.filename}'
    res=Object_recognition.run(imgUrl)
    return res


@app.route("/play")
async def play_wav_file():
    try:
        return send_file("Output/audio1.wav", attachment_filename='audio1.wav')
    except Exception as e:
        return str(e)

@app.route("/download")
# async
def get_wav_file():
    try:
        return send_file("Output/audio.wav", attachment_filename='audio.wav')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    asyncio.run(root())
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
