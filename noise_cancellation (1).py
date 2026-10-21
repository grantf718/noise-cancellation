import numpy as np

from audio_engine import AudioEngine


def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    sample_rate = 44100

    # Wave 1 parameters
    frequency1 = 440
    amplitude1 = 0.5
    phase1 = 0

    # Wave 2 parameters
    frequency2 = 440
    amplitude2 = 0.5
    phase2 = np.pi

    def parameters():
        return (amplitude1, frequency1, phase1, amplitude2, frequency2, phase2)

    # Plot waveforms
    times = np.linspace(0, 0.01, 1000) # 10ms interval
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9)) # Share x axis so graphs align

    wave1, = axes[0].plot(times, amplitude1 * np.sin(2 * np.pi * frequency1 * times + phase1)) 
    wave2, = axes[1].plot(times, amplitude2 * np.sin(2 * np.pi * frequency2 * times + phase2), color="tab:red")
    wave_combined, = axes[2].plot(times, amplitude1 * np.sin(2 * np.pi * frequency1 * times + phase1) + amplitude2 * np.sin(2 * np.pi * frequency2 * times + phase2), color="tab:purple")
    # ^^^ The commas are to unpack the single line we draw from the plot array

    for ax in axes:
        ax.set_ylim(-1, 1) # Limit y axis to 1, the max amplitude
        ax.grid()
        ax.set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (sec)")
    axes[0].set_title("Wave 1")
    axes[1].set_title("Wave 2")
    axes[2].set_title("Combined waveform")

    # Make room for sliders
    axes[0].set_position([0.14, 0.75, 0.8, 0.21])
    axes[1].set_position([0.14, 0.41, 0.8, 0.21])
    axes[2].set_position([0.14, 0.07, 0.8, 0.21])

    # Wave 1 sliders
    slider_amplitude1 = Slider(fig.add_axes([0.25, 0.70, 0.62, 0.025]), "Amplitude", 0, 0.5, valinit=amplitude1, valstep=0.05, color="tab:blue")
    slider_frequency1 = Slider(fig.add_axes([0.25, 0.675, 0.62, 0.025]), "Frequency", 20, 1000, valinit=frequency1, valstep=1, color="tab:blue")
    slider_phase1 = Slider(fig.add_axes([0.25, 0.65, 0.62, 0.025]), "Phase", 0, np.pi, valinit=phase1, valfmt="%.2f rad", valstep=np.pi / 16, color="tab:blue")

    # Wave 2 sliders
    slider_amplitude2 = Slider(fig.add_axes([0.25, 0.36, 0.62, 0.025]), "Amplitude", 0, 0.5, valinit=amplitude2, valstep=0.05, color="tab:red")
    slider_frequency2 = Slider(fig.add_axes([0.25, 0.335, 0.62, 0.025]), "Frequency", 20, 1000, valinit=frequency2, valstep=1, color="tab:red")
    slider_phase2 = Slider(fig.add_axes([0.25, 0.31, 0.62, 0.025]), "Phase", 0, np.pi, valinit=phase2, valfmt="%.2f rad", valstep=np.pi / 16, color="tab:red")

    # Update plot data to reflect changes made to waveform parameters
    def update_plots():
        audio.update(parameters())
        wave1.set_ydata(amplitude1 * np.sin(2 * np.pi * frequency1 * times + phase1))
        wave2.set_ydata(amplitude2 * np.sin(2 * np.pi * frequency2 * times + phase2))
        wave_combined.set_ydata(wave1.get_ydata() + wave2.get_ydata())
        fig.canvas.draw_idle()

    # Define setters for slider updates
    def set_amplitude1(value):
        nonlocal amplitude1
        amplitude1 = value
        update_plots()

    def set_frequency1(value):
        nonlocal frequency1
        frequency1 = value
        update_plots()

    def set_phase1(value):
        nonlocal phase1
        phase1 = value
        update_plots()

    def set_amplitude2(value):
        nonlocal amplitude2
        amplitude2 = value
        update_plots()

    def set_frequency2(value):
        nonlocal frequency2
        frequency2 = value
        update_plots()

    def set_phase2(value):
        nonlocal phase2
        phase2 = value
        update_plots()

    # Call setters upon slider update
    slider_amplitude1.on_changed(set_amplitude1)
    slider_frequency1.on_changed(set_frequency1)
    slider_phase1.on_changed(set_phase1)

    slider_amplitude2.on_changed(set_amplitude2)
    slider_frequency2.on_changed(set_frequency2)
    slider_phase2.on_changed(set_phase2)

    # Audio runs in its own process so mouse events cannot hold up synthesis.
    with AudioEngine(parameters(), sample_rate) as audio:
        def check_audio():
            try:
                underruns = audio.check()
            except RuntimeError as exc:
                print(f"Audio error: {exc}", flush=True)
                plt.close(fig)
                return False
            if underruns != check_audio.previous:
                print(f"Audio output underruns: {underruns}", flush=True)
                check_audio.previous = underruns
            return True

        check_audio.previous = 0
        timer = fig.canvas.new_timer(interval=500)
        timer.add_callback(check_audio)
        timer.start()
        try:
            plt.show()
        finally:
            timer.stop()


if __name__ == "__main__":
    main()
