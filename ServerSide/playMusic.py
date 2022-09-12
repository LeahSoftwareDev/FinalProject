import numpy as np
from scipy.io import wavfile
import utils
from playsound import playsound
# import Object_recognition
#
# "moonlight"
# # right_hand_notes = ['B3','E4','G4','B3','E4','G4','B3','E4','G4','B3','E4','G4',
# #                     'B3','E4','G4','B3','E4','G4','B3','E4','G4','B3','E4','G4',
# #                     'C4','E4','G4','C4','E4','G4','C4','F4','A4','C4','F4','A4',
# #                     'B3','d4','A4','B3','E4','G4','B3','E4','f4','A3','d4','f4',]
# #
# # right_hand_duration = [1/2]*12*4
# # left_hand_notes =['E3','D3','C3','A2','B2','B2']
# #
# # left_hand_duration=[6, 6, 3, 3, 3, 3]
# "twinkle twinkle little star"
# # right_hand_notes = ['C4', 'C4', 'G4', 'G4',
# #                    'A4', 'A4', 'G4',
# #                    'F4', 'F4', 'E4', 'E4',
# #                    'D4', 'D4', 'C4',
# #                    'G4', 'G4', 'F4', 'F4',
# #                    'E4', 'E4', 'D4',
# #                    'G4', 'G4', 'F4', 'F4',
# #                    'E4', 'E4', 'D4',
# #                    'C4', 'C4', 'G4', 'G4',
# #                    'A4', 'A4', 'G4',
# #                    'F4', 'F4', 'E4', 'E4',
# #                    'D4', 'D4', 'C4',]
# # right_hand_duration = [0.5, 0.5, 0.5, 0.5,
# #                        0.5, 0.5, 1]*6
# # left_hand_notes = ['C3',
# #                   'A3',
# #                   'F3',
# #                   'D3', 'C3',
# #                   'G3', 'F3',
# #                   'E3', 'D3',
# #                   'G3', 'F3',
# #                   'E3', 'D3',
# #                   'C3', 'E3', 'G3', 'C4',
# #                   'A3', 'A3', 'G3',
# #                   'F3', 'B2', 'E3', 'C3',
# #                   'D3', 'D3', 'C3']
# # left_hand_duration = [2,
# #                       2,
# #                       2,
# #                       1, 1,
# #                       1, 1,
# #                       1, 1,
# #                       1, 1,
# #                       1, 1,
# #                       0.5, 0.5, 0.5, 0.5,
# #                       0.5, 0.5, 1,
# #                       0.5, 0.5, 0.5, 0.5,
# #                       0.5, 0.5, 1]
#
# "fur elise"
# right_hand_notes = [
#     'E5', 'd5', 'E5', 'B4', 'D5', 'C5',
#     'A4', 'C4', 'E4', 'A4', 'B4', 'E4', 'g4', 'B4',
#     'C5', 'E4', 'E5', 'd5',
#     'E5', 'd5', 'E5', 'B4', 'D5', 'C5',
#     'A4', 'C4', 'E4', 'A4', 'B4', 'E4', 'C5', 'B4',
#     # 'A4', 'B4','C5','D5',
#     # 'E5', 'G4','F5','E5',
#     # 'D5', 'F4','E5','D5',
#     # 'C5', 'E4','D5','C5',
#     # 'B4', 'E4','E4','E5','E4',
#     # 'E5','E5','E6',
#     # 'd5','E5','B4','D5','C5',
#     # 'A4','C4','E4','A4','B4','E4','g4','B4',
#     # 'C5','E4','E5','d5',
#     # 'E5','d5','E5','B4','D5','C5',
#     # 'A4','C4','E4','A4','B4','E4','C5','B4',
#     'A4']
#
# right_hand_duration = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        3 / 8, 1 / 8, 1 / 8, 1 / 8, 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        3 / 8, 1 / 8, 1 / 8, 1 / 8, 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 2/8, 1 / 8, 1 / 8, 1 / 8, 1/8,
#                        # 1/8, 1/8, 1/8,
#                        # 1 / 8, 1 / 8, 1 / 8, 1 / 8,  1/8,
#                        # 3/ 8, 1 / 8, 1 / 8,1/8, 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        # 3 / 8, 1 / 8, 1 / 8, 1 / 8, 3 / 8, 1 / 8, 1 / 8, 1 / 8,
#                        6 / 8]
#
# left_hand_notes = ['A0',
#                    'A2', 'E3', 'A3', 'A0',
#                    'E3', 'g3', 'E4', 'A0',
#                    'A2', 'E3', 'A3', 'A0',
#                    'A0',
#                    'A2', 'E3', 'A3', 'A0',
#                    'E3', 'g3', 'E4', 'A0',
#                    'A2', 'E3', 'A3', 'A0']
# left_hand_duration = [3 / 4,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8,
#                       3 / 4,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8,
#                       1 / 8, 1 / 8, 1 / 8, 3 / 8]
#
# "full scale"
#
# # right_hand_notes = ['C3' ,'D3' ,'E3' ,'F3' ,'G3' , 'A3' ,'B3' ,'C4','D4'
# #                    ,'E4' ,'F4' ,'G4' ,'A4' , 'B4' ,'C5' ,'D5', 'E5',
# #                     'F5' ,'G5', 'A5' ,'B5' ,'C6' ,'D6' , 'E6' ,'F6' ,
# #                     'G6' ,'A6', 'B6' ,'C7' ,'D7' ,'d7' ,'E7' ,'F7',  'G7',
# #                     'A7' , 'B7']
# # right_hand_duration = [0.5]*36

# def play_music(right_hand_notes,right_hand_duration,left_hand_notes,left_hand_duration,bar_value):
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

    # wavfile.write('data/twinkle_star.wav', 44100, data.astype(np.int16))
    # wavfile.write('data/moonlight.wav', 44100, data.astype(np.int16))
    # wavfile.write('data/furelise.wav', 44100, data.astype(np.int16))
    # wavfile.write('data/fullscale.wav', 44100, right_hand.astype(np.int16))
    # wavfile.write('Output/audio.wav', 44100, data.astype(np.int16))
    wavfile.write("../../final-project/src/music/audio0.wav", 44100, data.astype(np.int16))
    # playsound("Output/audio.wav")
