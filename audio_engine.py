"""Keep audio generation independent of Matplotlib's GUI and Python GIL."""

import multiprocessing as mp

import numpy as np


class WaveSynth:
    """Two phase-continuous oscillators with short, click-reducing transitions."""

    def __init__(self, parameters, sample_rate=44100):
        # Each row contains amplitude, frequency (Hz), and phase (radians).
        self.parameters = np.array(parameters, dtype=float).reshape(2, 3)
        self.positions = np.zeros(2)
        self.sample_rate = sample_rate

    def render(self, parameters, frames):
        target = np.array(parameters, dtype=float).reshape(2, 3)
        # Reach each new setting over 10 ms rather than jumping at a block edge.
        transition = min(frames, round(0.01 * self.sample_rate))
        ramp = np.minimum(np.arange(frames) / transition, 1.0)
        values = self.parameters[:, :, None] + (
            target - self.parameters
        )[:, :, None] * ramp
        amplitudes, frequencies, phases = values[:, 0], values[:, 1], values[:, 2]
        increments = 2 * np.pi * frequencies / self.sample_rate
        angles = self.positions[:, None] + np.cumsum(increments, axis=1) - increments
        next_positions = self.positions + increments.sum(axis=1)

        if target[0, 1] == target[1, 1]:
            # Restore the demo's phase alignment when frequencies match, smoothly.
            correction = (next_positions[0] - next_positions[1] + np.pi) % (2 * np.pi) - np.pi
            angles[1] += correction * ramp
            next_positions[1] += correction

        mono = (amplitudes * np.sin(angles + phases)).sum(axis=0)
        self.positions = next_positions % (2 * np.pi)
        self.parameters = target
        return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)


def _audio_worker(parameters, stop, underruns, connection, sample_rate):
    # Import here so the audio process doesn't need a GUI backend.
    import sounddevice as sd

    try:
        with parameters.get_lock():
            synth = WaveSynth(parameters[:], sample_rate)
        # Blocking writes are safe here: this process has no GUI work to handle.
        with sd.OutputStream(samplerate=sample_rate, channels=2, dtype="float32",
                             blocksize=1024, latency="high") as stream:
            connection.send(("ready", stream.latency))
            while not stop.is_set():
                with parameters.get_lock():
                    snapshot = parameters[:]
                if stream.write(synth.render(snapshot, 1024)):
                    with underruns.get_lock():
                        underruns.value += 1
    except Exception as exc:
        connection.send(("error", str(exc)))
    finally:
        connection.close()


class AudioEngine:
    """A spawned audio process, with only six control values shared with the GUI."""

    def __init__(self, parameters, sample_rate=44100):
        context = mp.get_context("spawn")
        self.parameters = context.Array("d", parameters)
        self.stop = context.Event()
        self.underruns = context.Value("i", 0)
        self.connection, child_connection = context.Pipe(duplex=False)
        self.child_connection = child_connection
        self.process = context.Process(
            target=_audio_worker,
            args=(self.parameters, self.stop, self.underruns, child_connection, sample_rate),
            daemon=True,
        )

    def update(self, parameters):
        with self.parameters.get_lock():
            self.parameters[:] = parameters

    def check(self):
        if self.connection.poll():
            try:
                kind, message = self.connection.recv()
            except EOFError:
                kind, message = "error", "Audio process disconnected."
            if kind == "error":
                raise RuntimeError(message)
        if not self.process.is_alive():
            raise RuntimeError("Audio process stopped unexpectedly.")
        return self.underruns.value

    def __enter__(self):
        self.process.start()
        self.child_connection.close()
        try:
            if not self.connection.poll(10):
                raise RuntimeError("Timed out opening the audio output device.")
            kind, message = self.connection.recv()
            if kind != "ready":
                raise RuntimeError(f"Could not start audio: {message}")
            print(f"Audio output latency: {message * 1000:.0f} ms", flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.connection.close()
