import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sample_rate = 44100

# Wave 1 parameters
frequency1 = 440
amplitude1 = 0.5
phase1 = 0
position1 = 0.0

# Wave 2 parameters
frequency2 = 440
amplitude2 = 0.5
phase2 = np.pi
position2 = 0.0

# Callback function to play live updated sound
# Is called repeatedly by the output stream from sounddevice with updated params
def play_sound_callback(outdata, frames, time, status): 

    # Callback parameters:
    # Outdata is a 2D writable audio buffer where we write the samples
    # Frames are the number of samples needed per channel, automatically handled by sounddevice
    # Time and status are required callback parameters but using them is optional

    # Refer to global position variables in order to update them
    global position1, position2

    # Prevent waves of the same frequency from preserving their built up offset from parameter changes 
    # This effectively syncs the phases of the waves so cancellation can be demonstrated with the sliders
    if frequency1 == frequency2:
        position2 = position1

    # Create an array of frames and divide by sample rate to get the relative timestamps of when each sample will play
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

# Plot waveforms
times = np.linspace(0, 0.01, 1000) # 10ms interval
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7.44)) # Share x axis so graphs align

wave1, = axes[0].plot(times, amplitude1 * np.sin(2 * np.pi * frequency1 * times + phase1)) 
wave2, = axes[1].plot(times, amplitude2 * np.sin(2 * np.pi * frequency2 * times + phase2), color="tab:orange")
# ^^^ The commas are to unpack the single line we draw from the plot array

for ax in axes:
    ax.set_ylim(-1, 1) # Limit y axis to 1, the max amplitude
    ax.grid()
    ax.set_ylabel("Amplitude")
axes[1].set_xlabel("Time (sec)")
axes[0].set_title("Wave 1")
axes[1].set_title("Wave 2")

# Add sliders underneath each plot
axes[0].set_position([0.14, 0.71, 0.8, 0.25])
axes[1].set_position([0.14, 0.27, 0.8, 0.30])

# Wave 1 sliders
slider_amplitude1 = Slider(fig.add_axes([0.25, 0.65, 0.62, 0.025]), "Amplitude", 0, 0.5, valinit=amplitude1, valstep=0.05, color="tab:blue")
slider_frequency1 = Slider(fig.add_axes([0.25, 0.625, 0.62, 0.025]), "Frequency", 20, 1000, valinit=frequency1, valstep=1, color="tab:blue")
slider_phase1 = Slider(fig.add_axes([0.25, 0.60, 0.62, 0.025]), "Phase", 0, np.pi, valinit=phase1, valfmt="%.2f rad", valstep=np.pi / 16, color="tab:blue")

# Wave 2 sliders
slider_amplitude2 = Slider(fig.add_axes([0.25, 0.17, 0.62, 0.025]), "Amplitude", 0, 0.5, valinit=amplitude2, valstep=0.05, color="tab:orange")
slider_frequency2 = Slider(fig.add_axes([0.25, 0.145, 0.62, 0.025]), "Frequency", 20, 1000, valinit=frequency2, valstep=1, color="tab:orange")
slider_phase2 = Slider(fig.add_axes([0.25, 0.12, 0.62, 0.025]), "Phase", 0, np.pi, valinit=phase2, valfmt="%.2f rad", valstep=np.pi / 16, color="tab:orange")

# Trim the bottom margins
for ax in fig.axes:
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, (bottom - 0.07) / 0.93, width, height / 0.93])

# Update plot data to reflect changes made to waveform parameters
def update_plots():
    wave1.set_ydata(amplitude1 * np.sin(2 * np.pi * frequency1 * times + phase1))
    wave2.set_ydata(amplitude2 * np.sin(2 * np.pi * frequency2 * times + phase2))
    fig.canvas.draw_idle()

# Define setters for slider updates
def set_amplitude1(value):
    global amplitude1
    amplitude1 = value
    update_plots()

def set_frequency1(value):
    global frequency1
    frequency1 = value
    update_plots()

def set_phase1(value):
    global phase1
    phase1 = value
    update_plots()

def set_amplitude2(value):
    global amplitude2
    amplitude2 = value
    update_plots()

def set_frequency2(value):
    global frequency2
    frequency2 = value
    update_plots()

def set_phase2(value):
    global phase2
    phase2 = value
    update_plots()

# Call setters upon slider update
slider_amplitude1.on_changed(set_amplitude1)
slider_frequency1.on_changed(set_frequency1)
slider_phase1.on_changed(set_phase1)

slider_amplitude2.on_changed(set_amplitude2)
slider_frequency2.on_changed(set_frequency2)
slider_phase2.on_changed(set_phase2)

# Sounddevice's OutputStream uses a callback function to continuously play audio with data being updated live
with sd.OutputStream(
    samplerate=sample_rate, 
    channels=2, 
    callback=play_sound_callback,
    latency="low"
):
    plt.show()




