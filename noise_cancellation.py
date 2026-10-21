import numpy as np
import sounddevice as sd

sample_rate = 44100

# Wave 1 parameters
frequency1 = 440
amplitude1 = 0.2
phase1 = 0
position1 = 0.0

# Wave 2 parameters
frequency2 = 600
amplitude2 = 0.2
phase2 = 0
position2 = 0.0

# Callback function to play live updated sound
# Is called repeatedly by the output stream from sounddevice with updated params
def play_sound_callback(outdata, frames, time, status): 

    # Callback parameters:
    # Outdata is a 2D writable audio buffer where we write the samples
    # Frames are the number of samples needed per channel, automatically handled by sounddevice
    # Time and status are optional

    # Refer to global position variables in order to update them
    global position1, position2

    # Create an array of frames and divide by sample rate to get the timestamps when each sample will play
    frame_list = np.arange(frames)
    timestamps = frame_list / sample_rate

    # Create waves by calculating its value at each timestamp
    # Sin wave formula: Amplitude * sin(2pi * frequency * time + phase in radians)
    # We add on position to shift phase forward from the previous callback so it isn't choppy
    wave1 = amplitude1 * np.sin(position1 + 2 * np.pi * frequency1 * timestamps + phase1)
    wave2 = amplitude2 * np.sin(position2 + 2 * np.pi * frequency2 * timestamps + phase2)
    
    combined = wave1 + wave2

    # Write the audio to the left and right channels of outdata
    outdata[:, 0] = combined # Write to all rows in in column 0, the left channel
    outdata[:, 1] = combined # Write to all rows in in column 1, the right channel

    # Calculate the position of where the wave will start in the next callback (in radians) by wrapping it to one cycle from 0 to 2pi
    position1 = (position1 + 2 * np.pi * frequency1 * frames / sample_rate) % (2 * np.pi)
    position2 = (position2 + 2 * np.pi * frequency2 * frames / sample_rate) % (2 * np.pi)

# Sounddevice's OutputStream uses a callback function to continuously play audio with data being updated live
with sd.OutputStream(
    samplerate=sample_rate, 
    channels=2, 
    callback=play_sound_callback,
    latency="low"
):
    input("Press Enter to stop.\n")
