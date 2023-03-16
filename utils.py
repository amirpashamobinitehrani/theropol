import torch
import os
from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import dispatcher
from pythonosc import osc_server


def param_count(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)



def find_max_epoch(path):
    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            number = f[:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch


class UDP:

    '''
    Send UDP messages to Max msp
    ''' 
    
    def __init__(self, message, oscSender):
        self.oscSender = oscSender
        self.message = message
    
    
    def send(self):
        msg = osc_message_builder.OscMessageBuilder(address = "/udp")
        msg.add_arg(self.message)
        self.oscSender.send(msg.build())


class Vowel:

    '''
    class that maps predicted tensors to target vowels
    '''
    def __init__(self,
                 predictions):
        
        self.pred = predictions
    
    
    def map(predictions):
        vowel = None

        if abs(predictions[0]) < 0.5:
            vowel = 'c'
        
        elif abs(predictions[1]) < 0.8:
            vowel = 'a'

        elif abs(predictions[2]) < 0.9:
            vowel = 'o'

        elif abs(predictions[3]) < 0.9:
            vowel = 'i'

        elif abs(predictions[4]) < 0.7:
            vowel = 'e'
        
        elif (abs(predictions[5])) < 1.8:
            vowel = 'u'

        return vowel    




class Pitch:
    
    def __init__(self,
                distance,
                note_threshold,
                pos_mean,
                octave = False):
                
            self.distance = distance
            self.note_threshold = note_threshold
            self.pos_mean = pos_mean * 100
            self.octave = octave
            
    
    def pitch(self):
    
        count = 0
        if self.distance < self.note_threshold and self.pos_mean  > 0 and self.pos_mean < 10 and count == 0:
            notes = 62
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 10 and self.pos_mean < 20 and count == 0:
            notes = 60
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 20 and self.pos_mean < 30 and count == 0:
            notes = 59
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 30 and self.pos_mean < 40 and count == 0:
            notes = 57
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 40 and self.pos_mean < 50 and count == 0:
            notes = 55
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 50 and self.pos_mean < 60 and count == 0:
            notes = 53
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 60 and self.pos_mean < 70 and count == 0:
            notes = 52
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 70 and self.pos_mean < 80 and count == 0:
            notes = 50
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 80 and self.pos_mean < 90 and count == 0:
            notes = 48
            count = 1
        elif self.distance < self.note_threshold and self.pos_mean > 90 and self.pos_mean < 100 and count == 0:
            notes = 47
            count = 1
        else:
            count = 0
            notes = None
        
        return notes
