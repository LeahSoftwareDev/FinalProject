# Play Music - Final Project
This project is about music. I noticed that it is much easier to play when you first hear the piece you are going to play, especially for beginners who cannot read the page and understand the melody from it, as well as for people who have no musical sense.

The application will scan the sheet of music you want to play, and play the piece.

For example:
lines and objects detection using openCV:

![detect objects](https://github.com/LeahSoftwareDev/FinalProject/blob/main/ServerSide/pict/all%20objects%20detection.jpg)
![detect lines](https://github.com/LeahSoftwareDev/FinalProject/blob/main/ServerSide/pict/result%20lines.jpg)

I faced several challenges when writing the project, such as identifying the objects in the image, as well as correctly and accurately identifying the relevant characters from the multitude of markings on the page. In addition, after identifying the notes I had to play each note correctly, according to the appropriate rhythm.

The system provides the user with a convenient interface for uploading the sheet music image, when running the application scans the page, processes the data, creates an audio segment from the sheet music, the user can listen to the segment online and/or download it to the hard drive.

## Installations

### Install required libraries

Server side
- Python 3.7 / 3.9
- opencv-python
- numpy
- keras
- Pillow
- playsound
- scipy

Client side
- nodeJS
- react
- react-dom
- axios
- bootstrap
- react-audio-player
- react-fontawesome
- react-images-upload
- react-redux
- react-scripts
- redux
install node modules run `npm i`


## Usage
To run the user interface run `npm start` at the cmd in the project directory.

On the opening screen, upload a notes sheet of image type only, and confirm sending to the server.

You can then download the audio file to your personal computer, or listen to it online.


