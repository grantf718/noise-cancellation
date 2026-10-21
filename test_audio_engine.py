import unittest

import numpy as np

from audio_engine import WaveSynth


class WaveSynthTests(unittest.TestCase):
    def test_equal_opposite_waves_cancel_across_blocks(self):
        parameters = (0.5, 440, 0, 0.5, 440, np.pi)
        synth = WaveSynth(parameters)
        for _ in range(100):
            np.testing.assert_allclose(synth.render(parameters, 1024), 0, atol=1e-12)

    def test_steady_tone_is_continuous_across_blocks(self):
        parameters = (0.5, 440, 0, 0, 440, 0)
        synth = WaveSynth(parameters)
        output = np.concatenate([synth.render(parameters, 1024) for _ in range(8)])
        expected = 0.5 * np.sin(2 * np.pi * 440 * np.arange(len(output)) / 44100)
        np.testing.assert_allclose(output[:, 0], expected, atol=1e-7)
        np.testing.assert_array_equal(output[:, 0], output[:, 1])
        self.assertEqual(output.dtype, np.float32)

    def test_slider_change_does_not_jump_at_block_boundary(self):
        initial = (0.5, 440, 0, 0, 440, 0)
        synth = WaveSynth(initial)
        synth.render(initial, 1024)
        expected_first = 0.5 * np.sin(synth.positions[0])
        changed = (0.1, 900, np.pi, 0.5, 300, np.pi)
        output = synth.render(changed, 1024)
        self.assertAlmostEqual(float(output[0, 0]), expected_first, places=6)

    def test_returning_to_equal_frequency_restores_cancellation(self):
        initial = (0.5, 440, 0, 0.5, 660, np.pi)
        synth = WaveSynth(initial)
        for _ in range(5):
            synth.render(initial, 1024)
        matched = (0.5, 440, 0, 0.5, 440, np.pi)
        synth.render(matched, 1024)
        np.testing.assert_allclose(synth.render(matched, 1024), 0, atol=1e-12)

    def test_full_slider_range_stays_finite_and_within_audio_limits(self):
        rng = np.random.default_rng(575)
        synth = WaveSynth((0.5, 440, 0, 0.5, 440, np.pi))
        for _ in range(300):
            parameters = np.column_stack((rng.uniform(0, 0.5, 2),
                                          rng.integers(20, 1001, 2),
                                          rng.uniform(0, np.pi, 2))).ravel()
            output = synth.render(parameters, 1024)
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(np.abs(output).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
