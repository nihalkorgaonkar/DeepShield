"""
Unit & Integration Tests for Supervisor Hardening Module
Executed strictly in isolated temporary directories (tmp/test_supervisor_isolated).
Tests:
1. ControllerLock exclusive acquisition, duplicate rejection, and stale cleanup.
2. SupervisorHeartbeat background state updates.
3. EvidenceVerifier domain-specific structural artifact validation.
4. SupervisorTelemetry safe querying.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
import torch
import pandas as pd

from orchestrator.supervisor import (
    ControllerLock,
    ControllerAlreadyRunningError,
    SupervisorHeartbeat,
    EvidenceVerifier,
    SupervisorTelemetry,
    is_pid_alive,
)


class TestSupervisorIsolated(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath("tmp/test_supervisor_isolated")
        os.makedirs(self.test_dir, exist_ok=True)
        self.lock_file = os.path.join(self.test_dir, ".test_controller.lock")
        self.state_file = os.path.join(self.test_dir, "test_state.json")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_pid_alive(self):
        current_pid = os.getpid()
        self.assertTrue(is_pid_alive(current_pid))
        # An absurdly high PID that definitely doesn't exist
        self.assertFalse(is_pid_alive(9999999))

    def test_controller_lock_acquire_and_release(self):
        lock = ControllerLock(self.lock_file)
        self.assertTrue(lock.acquire())
        self.assertTrue(os.path.exists(self.lock_file))

        # Attempt duplicate acquisition from same lock instance should return True
        self.assertTrue(lock.acquire())

        # Second lock instance simulating duplicate controller process
        duplicate_lock = ControllerLock(self.lock_file)
        # Since it's the same PID in this thread, it acknowledges existing PID
        # Let's test with a mock alive PID by writing to lock file directly
        lock.release()
        self.assertFalse(os.path.exists(self.lock_file))

    def test_controller_lock_duplicate_rejection(self):
        # Write lock file with current alive PID
        current_pid = os.getpid()
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump({"pid": current_pid, "start_time": "2026-10-11T00:00:00Z"}, f)

        # Pretend another process tries to acquire
        class MockDifferentPidLock(ControllerLock):
            pass

        # If a lock exists with a living PID that is not us, it must raise
        with open(self.lock_file, "w", encoding="utf-8") as f:
            # PID 4 is Windows System process, always alive
            json.dump({"pid": 4, "start_time": "2026-10-11T00:00:00Z"}, f)

        lock = ControllerLock(self.lock_file)
        with self.assertRaises(ControllerAlreadyRunningError):
            lock.acquire()

    def test_controller_lock_stale_recovery(self):
        # Write lock file with dead PID
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump({"pid": 9999999, "start_time": "2026-10-11T00:00:00Z"}, f)

        lock = ControllerLock(self.lock_file)
        # Should detect dead PID, clear stale lock, and acquire successfully
        self.assertTrue(lock.acquire())
        self.assertTrue(os.path.exists(self.lock_file))
        lock.release()

    def test_supervisor_heartbeat(self):
        initial_state = {
            "project": "DeepShield",
            "last_updated": "2026-10-11T00:00:00Z",
            "tasks": {
                "TEST_TASK": {
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_seconds": 0.0,
                }
            }
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(initial_state, f)

        heartbeat = SupervisorHeartbeat(self.state_file, "TEST_TASK", interval_seconds=0.2)
        heartbeat.start()
        time.sleep(0.5)
        heartbeat.stop()

        with open(self.state_file, "r", encoding="utf-8") as f:
            updated_state = json.load(f)

        self.assertNotEqual(updated_state["last_updated"], "2026-10-11T00:00:00Z")
        self.assertGreaterEqual(updated_state["tasks"]["TEST_TASK"]["elapsed_seconds"], 0.2)

    def test_evidence_verifier_artifacts(self):
        # 1. Missing file
        ok, msg = EvidenceVerifier.verify_artifact(os.path.join(self.test_dir, "nonexistent.pt"))
        self.assertFalse(ok)

        # 2. Empty file
        empty_path = os.path.join(self.test_dir, "empty.txt")
        with open(empty_path, "w") as f:
            pass
        ok, msg = EvidenceVerifier.verify_artifact(empty_path)
        self.assertFalse(ok)

        # 3. Valid JSON
        json_path = os.path.join(self.test_dir, "valid.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
        ok, msg = EvidenceVerifier.verify_artifact(json_path)
        self.assertTrue(ok)

        # 4. Valid CSV
        csv_path = os.path.join(self.test_dir, "valid.csv")
        df = pd.DataFrame([{"model": "test", "eer": 0.02}])
        df.to_csv(csv_path, index=False)
        ok, msg = EvidenceVerifier.verify_artifact(csv_path)
        self.assertTrue(ok)

        # 5. Valid PNG magic bytes
        png_path = os.path.join(self.test_dir, "valid.png")
        with open(png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        ok, msg = EvidenceVerifier.verify_artifact(png_path)
        self.assertTrue(ok)

        # 6. Valid PyTorch Checkpoint
        ckpt_path = os.path.join(self.test_dir, "valid.pt")
        torch.save({"epoch": 1, "model_state_dict": {"w": torch.tensor([1.0])}}, ckpt_path)
        ok, msg = EvidenceVerifier.verify_artifact(ckpt_path)
        self.assertTrue(ok)

        # Multi-artifact verify_task_outputs
        ok, msg, hashes = EvidenceVerifier.verify_task_outputs([json_path, csv_path, png_path, ckpt_path])
        self.assertTrue(ok)
        self.assertEqual(len(hashes), 4)

    def test_telemetry(self):
        disk_info = SupervisorTelemetry.get_disk_telemetry("C:\\")
        self.assertIn("free_gb", disk_info)
        self.assertGreater(disk_info["free_gb"], 0)

        gpu_info = SupervisorTelemetry.get_gpu_telemetry()
        self.assertIn("available", gpu_info)


if __name__ == "__main__":
    unittest.main()
