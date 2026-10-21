"""
Two waves cancelling each other out: fast sliders + live sound.

Wave 1 is the noise. Wave 2 is the anti-noise. The bottom plot is what you
would hear when both play together (the residual).

Setup:   pip install numpy matplotlib sounddevice
Run:     python wave_cancellation_v2.py

Notes:
- Uses tkinter (included with most Python installs) for instant sliders.
- The plots only redraw the moving lines (blitting), so dragging stays smooth.
- Sound is optional. Without sounddevice the plots still work.
- The audio tone is 220 Hz. The plot shows 2 cycles so the shapes are easy to see.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sounddevice as sd
except Exception:  # not installed, or no audio device
    sd = None

# ---------------------------------------------------------------- settings
SAMPLE_RATE = 44100
AUDIO_FREQ = 220          # Hz, whole number so the loop is seamless
MASTER_VOLUME = 0.12      # keeps the loudest case (both at 2.0) comfortable
DISPLAY_CYCLES = 2
N_POINTS = 400            # fewer points = faster redraw

t = np.linspace(0, DISPLAY_CYCLES, N_POINTS)   # time axis in cycles
base = 2 * np.pi * t

# Values shared with the audio thread (plain floats, so reads are safe)
state = {"a1": 1.0, "a2": 1.0, "phase": 180.0, "mode": "sum", "n": 0}


def cancellation_text(a1, a2, phase_deg):
    """Cancellation of the sum versus wave 1 alone, from the exact formula."""
    if a1 < 1e-9:
        return "Wave 1 is silent"
    r = a2 / a1
    phi = np.radians(phase_deg)
    # Wave 2 = a2*sin(x + phase), so 180 degrees is the ideal mirror image.
    # Sum amplitude relative to wave 1:
    residual = np.sqrt(max(1 + 2 * r * np.cos(phi) + r * r, 0.0))
    if residual < 1e-4:
        return "Perfect cancellation: silence"
    db = -20 * np.log10(residual)
    if db >= 0:
        return f"Cancellation: {db:.1f} dB (quieter)"
    return f"Cancellation: {db:.1f} dB (LOUDER than the original)"


# ---------------------------------------------------------------- window
root = tk.Tk()
root.title("Noise cancellation: two waves")

fig = Figure(figsize=(9, 6), dpi=100)
ax_top = fig.add_subplot(211)
ax_bot = fig.add_subplot(212)
fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10, hspace=0.45)

(line1,) = ax_top.plot(t, np.sin(base), color="tab:blue",
                       label="Wave 1 (noise)", animated=True)
(line2,) = ax_top.plot(t, np.sin(base + np.pi), color="tab:orange",
                       linestyle="--", label="Wave 2 (anti-noise)", animated=True)
(line_res,) = ax_bot.plot(t, np.zeros_like(t), color="tab:green", animated=True)
status = ax_bot.text(0.02, 0.84, "", transform=ax_bot.transAxes, fontsize=12,
                     animated=True, bbox=dict(facecolor="white", edgecolor="gray"))

for ax, title in ((ax_top, "The two waves"), (ax_bot, "Residual: wave 1 + wave 2")):
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlim(0, DISPLAY_CYCLES)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
ax_bot.set_xlabel("Time (cycles)")
ax_top.legend(loc="upper right")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

animated = [(ax_top, line1), (ax_top, line2), (ax_bot, line_res), (ax_bot, status)]
bg = {"img": None}


def blit():
    if bg["img"] is None:
        return
    canvas.restore_region(bg["img"])
    for ax, art in animated:
        ax.draw_artist(art)
    canvas.blit(fig.bbox)


def on_draw(_event):
    bg["img"] = canvas.copy_from_bbox(fig.bbox)
    for ax, art in animated:
        ax.draw_artist(art)


canvas.mpl_connect("draw_event", on_draw)


def update_plot():
    a1, a2, ph = state["a1"], state["a2"], np.radians(state["phase"])
    w1 = a1 * np.sin(base)
    w2 = a2 * np.sin(base + ph)
    line1.set_ydata(w1)
    line2.set_ydata(w2)
    line_res.set_ydata(w1 + w2)
    status.set_text(cancellation_text(state["a1"], state["a2"], state["phase"]))
    blit()


# ---------------------------------------------------------------- controls
controls = ttk.Frame(root, padding=8)
controls.pack(side=tk.BOTTOM, fill=tk.X)

v_a1 = tk.DoubleVar(value=state["a1"])
v_a2 = tk.DoubleVar(value=state["a2"])
v_ph = tk.DoubleVar(value=state["phase"])


def on_slider(_=None):
    state["a1"] = v_a1.get()
    state["a2"] = v_a2.get()
    state["phase"] = v_ph.get()
    update_plot()


def add_slider(row, label, var, lo, hi, step):
    ttk.Label(controls, text=label, width=20).grid(row=row, column=0, sticky="w")
    tk.Scale(controls, variable=var, from_=lo, to=hi, resolution=step,
             orient=tk.HORIZONTAL, length=520, command=on_slider
             ).grid(row=row, column=1, sticky="ew", padx=6)


add_slider(0, "Wave 1 volume", v_a1, 0.0, 2.0, 0.01)
add_slider(1, "Wave 2 volume", v_a2, 0.0, 2.0, 0.01)
add_slider(2, "Wave 2 phase (deg)", v_ph, 0.0, 360.0, 1.0)
controls.columnconfigure(1, weight=1)

# ---------------------------------------------------------------- sound
stream = {"obj": None}
v_mode = tk.StringVar(value="sum")


def audio_callback(outdata, frames, _time, _status):
    n = state["n"] + np.arange(frames)
    x = 2 * np.pi * AUDIO_FREQ * n / SAMPLE_RATE
    w1 = state["a1"] * np.sin(x)
    w2 = state["a2"] * np.sin(x + np.radians(state["phase"]))
    mode = state["mode"]
    out = w1 if mode == "w1" else w2 if mode == "w2" else w1 + w2
    outdata[:, 0] = (MASTER_VOLUME * out).astype(np.float32)
    state["n"] = (state["n"] + frames) % SAMPLE_RATE   # whole-cycle wrap


def toggle_sound():
    if stream["obj"] is not None:
        stream["obj"].stop()
        stream["obj"].close()
        stream["obj"] = None
        sound_btn.config(text="Sound: OFF (click to play)")
        return
    try:
        stream["obj"] = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                        dtype="float32", callback=audio_callback)
        stream["obj"].start()
        sound_btn.config(text="Sound: ON (click to stop)")
    except Exception as exc:
        stream["obj"] = None
        sound_btn.config(text="Sound unavailable")
        print("Could not start audio:", exc)


def on_mode():
    state["mode"] = v_mode.get()


audio_row = ttk.Frame(controls)
audio_row.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

sound_btn = ttk.Button(audio_row, text="Sound: OFF (click to play)", command=toggle_sound)
sound_btn.pack(side=tk.LEFT, padx=(0, 16))
if sd is None:
    sound_btn.config(text="Sound unavailable (pip install sounddevice)", state="disabled")

ttk.Label(audio_row, text="Hear:").pack(side=tk.LEFT)
for text, value in (("Both waves (residual)", "sum"),
                    ("Wave 1 only", "w1"), ("Wave 2 only", "w2")):
    ttk.Radiobutton(audio_row, text=text, value=value, variable=v_mode,
                    command=on_mode).pack(side=tk.LEFT, padx=6)


def on_close():
    if stream["obj"] is not None:
        stream["obj"].stop()
        stream["obj"].close()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

canvas.draw()      # first full draw captures the background for blitting
update_plot()
root.mainloop()
