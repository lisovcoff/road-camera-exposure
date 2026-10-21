"""Offline v0.2 tests: real input validation and synthetic feedback only."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from simulate_v02_001 import NS, run_synthetic, synthetic_configuration
from road_core_v02 import (ExposureController, FrameMetadata, SpeedSample,
                          ZoneMask, analyze_y_plane)


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.hardware, self.config, self.analysis = synthetic_configuration()
        self.zones = (ZoneMask("road", np.ones((20, 20), dtype=bool), 1),)

    def scene(self, value=100, **kwargs):
        return analyze_y_plane(np.full((20, 20), value, dtype=np.uint8), self.zones,
                               self.analysis, mask_version="test-001", transform="identity", **kwargs)

    def metadata(self, frame=0, now=NS, exposure=4000, gain=8, **kwargs):
        return replace(FrameMetadata(frame, now - 10_000_000, exposure, gain, 1.0, 33333), **kwargs)

    def controller(self, **kwargs):
        return ExposureController(self.hardware, replace(self.config, **kwargs))

    def update(self, controller, *, frame=0, now=NS, value=100, exposure=4000, gain=8,
               speed=8):
        return controller.update(self.scene(value), self.metadata(frame, now, exposure, gain),
                                 SpeedSample(speed, now), now)


class AnalysisTests(CoreTest):
    def test_rejects_rgb_nan_float_empty_and_nonarray(self):
        for plane in (np.zeros((20, 20, 3), dtype=np.uint8), np.full((20, 20), np.nan),
                      np.zeros((20, 20), dtype=float), np.empty((0, 20), dtype=np.uint8), None):
            with self.subTest(shape=getattr(plane, "shape", None)), self.assertRaises(ValueError):
                analyze_y_plane(plane, self.zones, self.analysis,
                                mask_version="test", transform="identity")

    def test_noncontiguous_y_slice_ignores_padding_and_uv(self):
        buffer = np.full((30, 32), 255, dtype=np.uint8)
        buffer[:20, :20] = 82
        scene = analyze_y_plane(buffer[:20, :20], self.zones, self.analysis,
                                mask_version="stride", transform="rotate180")
        self.assertEqual(scene.meter_y, 82)
        self.assertEqual(scene.clipped_pct, 0)
        self.assertEqual(scene.transform, "rotate180")

    def test_zone_weights_and_exclusions_are_explicit(self):
        first = np.zeros((20, 20), dtype=bool)
        first[:10] = True
        zones = (ZoneMask("near", first, 3), ZoneMask("far", ~first, 1))
        plane = np.full((20, 20), 200, dtype=np.uint8)
        plane[:10] = 40
        exclude = np.zeros((20, 20), dtype=bool)
        exclude[0] = True
        plane[0] = 255
        scene = analyze_y_plane(plane, zones, self.analysis, exclude=exclude,
                                mask_version="weighted", transform="identity")
        self.assertEqual(scene.meter_y, 80)
        self.assertEqual([zone.pixels for zone in scene.zones], [180, 200])
        self.assertEqual(scene.clipped_pct, 0)

    def test_glare_is_reported_without_moving_median_meter(self):
        plane = np.full((20, 20), 40, dtype=np.uint8)
        plane[0] = 255
        scene = analyze_y_plane(plane, self.zones, self.analysis,
                                mask_version="glare", transform="identity")
        self.assertEqual(scene.meter_y, 40)
        self.assertEqual(scene.clipped_pct, 5)
        result = self.controller().update(scene, self.metadata(), SpeedSample(8, NS), NS)
        self.assertIn("HIGHLIGHTS", result.statuses)
        self.assertIsNone(result.command)

    def test_rejects_overlap_wrong_shape_duplicate_names_and_empty_zone(self):
        wrong = ZoneMask("wrong", np.ones((10, 10), dtype=bool), 1)
        empty = ZoneMask("empty", np.zeros((20, 20), dtype=bool), 1)
        overlap = ZoneMask("overlap", np.ones((20, 20), dtype=bool), 1)
        for zones in ((), (wrong,), (empty,), self.zones * 2, self.zones + (overlap,)):
            with self.subTest(zones=len(zones)), self.assertRaises(ValueError):
                analyze_y_plane(np.zeros((20, 20), dtype=np.uint8), zones, self.analysis,
                                mask_version="invalid", transform="identity")
        with self.assertRaises(ValueError):
            self.scene(exclude=np.ones((20, 20), dtype=bool))

    def test_masks_are_copied_and_large_weights_do_not_overflow(self):
        original = np.ones((20, 20), dtype=bool)
        zone = ZoneMask("copied", original, 1e308)
        original[:] = False
        self.assertTrue(zone.mask.all())
        scene = analyze_y_plane(np.full((20, 20), 100, dtype=np.uint8), (zone,),
                                self.analysis, mask_version="copy", transform="identity")
        self.assertEqual(scene.meter_y, 100)


class ValidationTests(CoreTest):
    def test_invalid_speed_produces_no_command_and_json_safe_telemetry(self):
        for speed in (SpeedSample(float("nan"), NS), SpeedSample(-1, NS),
                      SpeedSample(8, NS + 1), "8", SpeedSample(True, NS)):
            result = self.controller().update(self.scene(40), self.metadata(), speed, NS)
            self.assertIn("INVALID_INPUT", result.statuses)
            self.assertIsNone(result.command)
            json.dumps(result.to_record(), allow_nan=False)

    def test_invalid_metadata_and_scene_produce_no_command(self):
        for metadata in (None, self.metadata(exposure_us=0), self.metadata(digital_gain=float("nan")),
                         self.metadata(frame_duration_us=100), self.metadata(exposure_start_ns=NS + 1),
                         self.metadata(gain=32)):
            result = self.controller().update(self.scene(), metadata, SpeedSample(8, NS), NS)
            self.assertIn("INVALID_INPUT", result.statuses)
            self.assertIsNone(result.command)
        result = self.controller().update(replace(self.scene(), meter_y=float("nan")),
                                          self.metadata(), SpeedSample(8, NS), NS)
        self.assertIn("INVALID_INPUT", result.statuses)

    def test_bad_clock_does_not_advance_controller(self):
        controller = self.controller()
        self.update(controller)
        for bad in (NS, NS - 1, 1.5, True, -1):
            result = controller.update(self.scene(), self.metadata(), SpeedSample(8, NS), bad)
            self.assertIn("INVALID_INPUT", result.statuses)
            self.assertIsNone(result.command)
        result = self.update(controller, frame=1, now=NS + 100_000_000)
        self.assertIn("NEAR_TARGET", result.statuses)

    def test_duplicate_old_and_out_of_order_frames_hold(self):
        controller = self.controller()
        self.update(controller)
        now = NS + 100_000_000
        for metadata in (self.metadata(0, now), self.metadata(1, now, exposure_start_ns=NS - 20_000_000)):
            result = controller.update(self.scene(40), metadata, SpeedSample(8, now), now)
            self.assertIn("STALE_FRAME", result.statuses)
            self.assertIsNone(result.command)
            now += 100_000_000
        result = self.controller().update(self.scene(40), self.metadata(0, 1), None, 2 * NS)
        # Negative frame time is invalid; a valid but too old frame is STALE_FRAME.
        self.assertIn("INVALID_INPUT", result.statuses)
        result = self.controller().update(self.scene(40), self.metadata(), None, 2 * NS)
        self.assertIn("STALE_FRAME", result.statuses)

    def test_configuration_errors_are_rejected_before_updates(self):
        for changes in ({"enter_view_kmh": 8}, {"max_speed_age_ns": 0},
                        {"mode_dwell_ns": 0.1}, {"min_response": 0},
                        {"highlight_pct": 101}, {"max_response": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        with self.assertRaises(ValueError):
            ExposureController(self.hardware, replace(self.config,
                detect=replace(self.config.detect, min_exposure_us=30_000,
                               max_exposure_us=40_000, preferred_exposure_us=30_000)))


class FeedbackTests(CoreTest):
    def test_dispatch_apply_and_quantization_before_next_command(self):
        controller = self.controller()
        first = self.update(controller, value=50)
        self.assertIsNotNone(first.command)
        waiting = self.update(controller, frame=1, now=NS + 100_000_000, value=50)
        self.assertIn("AWAITING_DISPATCH", waiting.statuses)
        self.assertIsNone(waiting.command)
        controller.mark_sent(first.command.command_id, NS + 100_000_000)
        waiting = self.update(controller, frame=2, now=NS + 200_000_000, value=50)
        self.assertIn("AWAITING_APPLY", waiting.statuses)
        observed = self.update(controller, frame=3, now=NS + 300_000_000, value=100,
                               exposure=3981, gain=first.command.analogue_gain + 0.02)
        self.assertIn("COMMAND_OBSERVED", observed.statuses)
        self.assertIn("NEAR_TARGET", observed.statuses)
        self.assertIsNone(observed.command)
        self.assertEqual(observed.observed_command_id, first.command.command_id)
        self.assertEqual(observed.observed_sent_ns, NS + 100_000_000)

    def test_matching_controls_in_frame_started_before_dispatch_do_not_ack(self):
        controller = self.controller()
        first = self.update(controller, value=50)
        controller.mark_sent(first.command.command_id, NS + 50_000_000)
        now = NS + 100_000_000
        md = self.metadata(1, now, gain=first.command.analogue_gain,
                           exposure_start_ns=NS + 20_000_000)
        result = controller.update(self.scene(100), md, SpeedSample(8, now), now)
        self.assertIn("AWAITING_APPLY", result.statuses)
        self.assertNotIn("COMMAND_OBSERVED", result.statuses)

    def test_wrong_duplicate_and_early_dispatch_are_rejected(self):
        controller = self.controller()
        command = self.update(controller, value=50).command
        for command_id, timestamp in ((command.command_id + 1, NS), (command.command_id, NS - 1)):
            with self.assertRaises(ValueError):
                controller.mark_sent(command_id, timestamp)
        controller.mark_sent(command.command_id, NS)
        with self.assertRaises(ValueError):
            controller.mark_sent(command.command_id, NS)

    def test_timeout_latches_even_with_bad_frames_and_late_matching_feedback(self):
        controller = self.controller()
        command = self.update(controller, value=50).command
        controller.mark_sent(command.command_id, NS)
        now = 2 * NS
        result = controller.update(None, None, None, now)
        self.assertIn("CONTROL_TIMEOUT", result.statuses)
        result = self.update(controller, frame=1, now=now + 100_000_000,
                             gain=command.analogue_gain)
        self.assertIn("CONTROL_TIMEOUT", result.statuses)
        self.assertIsNone(result.command)

    def test_shorter_profile_waits_for_outstanding_command_and_reports_limit(self):
        controller = self.controller(mode_dwell_ns=0)
        first = self.update(controller, value=50, exposure=8000, speed=13)
        self.assertEqual(first.mode, "VIEW")
        controller.mark_sent(first.command.command_id, NS)
        now = NS + 100_000_000
        result = controller.update(self.scene(50), self.metadata(1, now, exposure=8000), None, now)
        self.assertIn("PROFILE_LIMIT_PENDING", result.statuses)
        self.assertIn("STALE_SPEED", result.statuses)
        self.assertIsNone(result.command)
        now += 100_000_000
        result = controller.update(self.scene(50),
            self.metadata(2, now, exposure=first.command.exposure_us, gain=first.command.analogue_gain), None, now)
        self.assertIn("COMMAND_OBSERVED", result.statuses)
        self.assertLessEqual(result.command.exposure_us, 4000)
        self.assertNotIn("LOW_LIGHT", result.statuses)


class PolicyTests(CoreTest):
    def test_low_light_gap_is_fixed_with_quantized_readback_and_dwell(self):
        controller = self.controller()
        result = self.update(controller, value=82, exposure=3981, gain=16)
        self.assertIn("SATURATION_PENDING", result.statuses)
        self.assertIsNone(result.command)
        result = self.update(controller, frame=1, now=NS + self.config.status_dwell_ns,
                             value=82, exposure=3981, gain=16)
        self.assertIn("LOW_LIGHT", result.statuses)
        self.assertIsNone(result.command)

    def test_saturation_dwell_resets_when_brightness_recovers(self):
        controller = self.controller()
        self.update(controller, value=82, exposure=3981, gain=16)
        self.update(controller, frame=1, now=NS + 100_000_000, exposure=3981, gain=16)
        result = self.update(controller, frame=2, now=NS + 200_000_000,
                             value=82, exposure=3981, gain=16)
        self.assertIn("SATURATION_PENDING", result.statuses)
        self.assertNotIn("LOW_LIGHT", result.statuses)

    def test_too_bright_at_minimum_and_no_arbitrary_two_ms_floor(self):
        result = self.update(self.controller(status_dwell_ns=0), value=240, exposure=100, gain=1)
        self.assertIn("TOO_BRIGHT", result.statuses)
        self.assertIsNone(result.command)
        result = self.update(self.controller(), value=240, exposure=1000, gain=1)
        self.assertLess(result.command.exposure_us, 1000)

    def test_integer_nanoseconds_switch_at_exact_two_second_boundary(self):
        controller = self.controller()
        self.assertEqual(self.update(controller, speed=13).mode, "DETECT")
        for frame in range(1, 8):
            self.update(controller, frame=frame, now=NS + frame * (NS // 4), speed=13)
        result = self.update(controller, frame=8, now=NS + 2 * NS - 1, speed=13)
        self.assertEqual(result.mode, "DETECT")
        result = self.update(controller, frame=9, now=NS + 2 * NS, speed=13)
        self.assertEqual(result.mode, "VIEW")

    def test_unsampled_speed_gap_and_out_of_order_sample_do_not_count_as_dwell(self):
        controller = self.controller()
        self.update(controller, speed=13)
        result = self.update(controller, frame=1, now=3 * NS, speed=13)
        self.assertEqual(result.mode, "DETECT")
        now = 3 * NS + 100_000_000
        result = controller.update(self.scene(), self.metadata(2, now),
                                    SpeedSample(13, 3 * NS - 1), now)
        self.assertIn("STALE_SPEED", result.statuses)

    def test_long_frame_gap_resets_saturation_confirmation(self):
        controller = self.controller()
        self.update(controller, value=82, exposure=3981, gain=16)
        result = self.update(controller, frame=1, now=2 * NS, value=82, exposure=3981, gain=16)
        self.assertIn("SATURATION_PENDING", result.statuses)
        self.assertNotIn("LOW_LIGHT", result.statuses)

    def test_invalid_sample_interrupts_mode_dwell(self):
        controller = self.controller()
        self.update(controller, speed=13)
        controller.update(self.scene(), None, SpeedSample(13, 2 * NS), 2 * NS)
        result = self.update(controller, frame=1, now=3 * NS, speed=13)
        self.assertEqual(result.mode, "DETECT")

    def test_stale_speed_selects_detect_without_increasing_exposure_product(self):
        controller = self.controller(mode_dwell_ns=0)
        self.update(controller, value=105, exposure=8000, speed=13)
        now = NS + self.config.max_speed_age_ns + 1
        result = controller.update(self.scene(40), self.metadata(1, now, exposure=8000),
                                    SpeedSample(13, NS), now)
        self.assertEqual(result.mode, "DETECT")
        self.assertIn("STALE_SPEED", result.statuses)
        self.assertLessEqual(result.command.exposure_us, 4000)
        self.assertLessEqual(result.command.exposure_us * result.command.analogue_gain, 8000 * 8)

    def test_frame_duration_caps_commands_and_reported_limits(self):
        now = NS
        result = self.controller().update(self.scene(40),
            self.metadata(exposure=1000, frame_duration_us=2000), SpeedSample(8, now), now)
        self.assertEqual(result.limits.max_exposure_us, 2000)
        self.assertLessEqual(result.command.exposure_us, 2000)

    def test_requests_remain_inside_hardware_profile_and_cadence_bounds(self):
        for y in (0, 22, 82, 100, 240, 255):
            for exposure in (100, 1000, 3981, 8000):
                for gain in (1, 8, 16):
                    for speed in (8, 13):
                        controller = self.controller(mode_dwell_ns=0, status_dwell_ns=0)
                        result = self.update(controller, value=y, exposure=exposure, gain=gain, speed=speed)
                        if result.command:
                            command = result.command
                            self.assertLessEqual(result.limits.min_exposure_us, command.exposure_us)
                            self.assertLessEqual(command.exposure_us, result.limits.max_exposure_us)
                            self.assertLessEqual(result.limits.min_gain, command.analogue_gain)
                            self.assertLessEqual(command.analogue_gain, result.limits.max_gain)
                        json.dumps(result.to_record(), allow_nan=False)

    def test_synthetic_closed_loop_converges_under_light_step_and_delay(self):
        rows = run_synthetic()
        self.assertEqual(len(rows), 60)
        self.assertFalse(any("CONTROL_TIMEOUT" in row["statuses"] for row in rows))
        self.assertTrue(any("LOW_LIGHT" in row["statuses"] for row in rows[:15]))
        self.assertTrue(any("NEAR_TARGET" in row["statuses"] for row in rows[15:40]))
        self.assertTrue(any("LOW_LIGHT" in row["statuses"] for row in rows[40:]))
        self.assertTrue(any("AWAITING_APPLY" in row["statuses"] for row in rows))
        for row in rows:
            json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
