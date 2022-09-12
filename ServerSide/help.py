import os
import shutil
import cv2
from PIL import Image


#global list
octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']

def other():
  pass

def key_flat(notes,i):
  '''
  transpose note to flat
  :param notes:
  :param i index:
  :return: transposed notes
  '''
  index=octave.index(notes[i][0])
  if (index == 0):
    index = 12
  notes[i] = notes[i].replace(notes[i][0], octave[index - 1])
  return notes

def key_natural(notes,i):
  '''
  transpose note to natural
  :param notes:
  :param i index:
  :return: transposed notes
  '''
  index=octave.index(notes[i][0])
  if (index == 0):
    index = 12
  notes[i] = notes[i].replace(notes[i][0], octave[index].upper())
  return notes

def key_sharp(notes,i):
  '''
  transpose note to sharp
  :param notes:
  :param i index:
  :return: transposed notes
  '''
  index = octave.index(notes[i][0])
  if(index==11):
    index=-1
  notes[i]=notes[i].replace(notes[i][0],octave[index + 1])
  return notes

def ornament_mordent(notes, duration,i):
  '''
  add the mordent trill action to the notes
  :param notes:
  :param duration:
  :param i:
  :return: notes and duration
  '''
  index = octave.index(notes[i][0])
  n=[octave[index]+notes[i][1], octave[index+1].upper()+notes[i][1], octave[index]+notes[i][1]]
  d=[duration[i]/3]*3
  print(d)
  print(notes, duration)
  notes = notes[0:i] + n + notes[i+1:]
  duration = duration[0:i]+d+duration[i+1:]
  return notes, duration

def right_hand(notes):
  return notes

def left_hand(notes):
  '''
  transpose notes from G key to F key
  :param notes in G key:
  :return: notes in F key
  '''
  for n in range(len(notes)):
    index = octave.index(notes[n][0])
    if(index==10):
      index=-3
    if (index == 11):
      index = -2
    if (index == 9):
      index = -4
    if (index == 8):
      index = -5
    notes[n] = notes[n].replace(notes[n][0], octave[index + 4].upper())
  return notes

def sixteenth(duration,i):
  duration[i]/=4
  # return duration

def eighth(duration,i):
  duration[i]/=2
  # return duration

def repeat_dot(notes,duration,i,j):
  startpoint = i
  endpoint = j + 1
  n = notes[startpoint:endpoint]
  d=duration[startpoint:endpoint]
  print(notes, duration)
  notes = notes[0:endpoint] + n + notes[endpoint:]
  duration = duration[0:endpoint] + d + duration[endpoint:]
  # print(notes,duration)  # => [1, 2, 3, 5, 4, 7, 8, 9]
  return notes,duration

def timesig3():
  return 3/4

def timesig4():
  return 1

def timesig8():
  return 8/6

def rest_8th(notes,duration,i):
  r='A0'
  d=1/8
  notes = notes[0:i] + r + notes[i + 1:]
  duration = duration[0:i] + d + duration[i + 1:]
  return notes,duration

def rest_half(notes,duration,i):
  r = 'A0'
  d = 1 / 2
  notes = notes[0:i] + r + notes[i + 1:]
  duration = duration[0:i] + d + duration[i + 1:]
  return notes, duration

def rest_quarter(notes,duration,i):
  r = 'A0'
  d = 1 / 4
  notes = notes[0:i] + r + notes[i + 1:]
  duration = duration[0:i] + d + duration[i + 1:]
  return notes, duration

# print(['A4','C4','E4','a4','B4','E4','g4','B4'])
# print(left_hand(['A4','C4','E4','a4','B4','E4','g4','B4']))
# ornament_mordent(['A4','C4','E4','A4','B4','E4','g4','B4'],[3/8, 1/8, 1/8, 1/8, 3/8, 1/8, 1/8, 1/8],2)