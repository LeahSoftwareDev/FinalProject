import numpy as np
from scipy.io import wavfile
import utils
from playsound import playsound


def play_music(right_hand_notes, right_hand_duration, left_hand_notes, left_hand_duration, timesig):
    for i in range(len(right_hand_duration)):
        right_hand_duration[i] *= 2
    for i in range(len(left_hand_duration)):
        left_hand_duration[i] *= 2

    factor = [0.68, 0.26, 0.03, 0.0, 0.03]
    length = [0.01, 0.6, 0.29, 0.1]
    decay = [0.05, 0.02, 0.005, 0.1]
    sustain_level = 0.01

    right_hand = utils.get_song_data(right_hand_notes, right_hand_duration, timesig * 2,
                                     factor, length, decay, sustain_level)
    print('right hand succeed')
    factor = [0.73, 0.16, 0.06, 0.01, 0.02, 0.01, 0.01]
    length = [0.01, 0.29, 0.6, 0.1]
    decay = [0.05, 0.02, 0.005, 0.1]
    sustain_level = 0.06
    left_hand = utils.get_song_data(left_hand_notes, left_hand_duration, timesig * 2,
                                    factor, length, decay, sustain_level)
    print('left hand succeed')
    data = left_hand + right_hand
    data = data * (4096 / np.max(data))

    wavfile.write("../../final-project/src/music/audio0.wav", 44100, data.astype(np.int16))
    # playsound("Output/audio.wav")
