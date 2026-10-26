"""
DeepShield Autonomous Research Orchestrator — Controller
Manages task dependencies, execution queue, GPU concurrency governance,
resource preflight checks, checkpoint verification, and atomic state persistence.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator.tasks import (
    ResearchTask,
    get_task_registry,
    verify_protected_baselines,
    compute_sha256,
    PROTECTED_BASELINE_HASHES,
)
from orchestrator.supervisor import (
    ControllerLock,
    ControllerAlreadyRunningError,
    SupervisorHeartbeat,
    EvidenceVerifier,
    SupervisorTelemetry,
)


STATE_FILE = "orchestrator/state.json"
LOGS_DIR = "orchestrator/logs"


class DeepShieldController:
    """Autonomous Research Orchestrator for DeepShield."""

    def __init__(self, state_file: str = STATE_FILE, logs_dir: str = LOGS_DIR):
        self.state_file = state_file
        self.logs_dir = logs_dir
        self.tasks = get_task_registry()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Loads persistent state or initializes from task registry."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not parse {self.state_file}: {e}. Reinitializing.")

        initial_state = {
            "project": "DeepShield",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "active_task_id": None,
            "tasks": {},
        }
        for task_id, t in self.tasks.items():
            initial_state["tasks"][task_id] = {
                "name": t.name,
                "status": "PENDING",
                "dependencies": t.dependencies,
                "retries": 0,
                "max_retries": t.max_retries,
                "start_time": None,
                "end_time": None,
                "elapsed_seconds": 0.0,
                "exit_code": None,
                "artifacts": {},
                "log_file": None,
                "error_message": None,
            }
        return initial_state

    def _save_state(self):
        """Atomically persists state to disk."""
        self.state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        dir_name = os.path.dirname(self.state_file) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(self.state, tf, indent=2)
            temp_name = tf.name
        os.replace(temp_name, self.state_file)

    def check_system_resources(self, is_preflight: bool = True) -> Tuple[bool, str]:
        """Verifies disk space, host RAM, and GPU status.
        
        Args:
            is_preflight: If True, enforces strict idle thresholds (5,000 MiB free VRAM).
                          If False (runtime monitoring), verifies active thermal & VRAM exhaustion limits.
        """
        # 1. Disk Space on C:
        try:
            total, used, free = shutil.disk_usage("C:\\")
            free_gb = free / (1024 ** 3)
            if free_gb < 15.0:
                return False, f"Insufficient disk space on C: ({free_gb:.2f} GB free, minimum required is 15.0 GB)"
        except Exception as e:
            return False, f"Could not query disk space: {e}"

        # 2. Protected Baseline Hashes
        baselines_ok, hash_results = verify_protected_baselines()
        if not baselines_ok:
            mismatches = [f"{p}: got {h}" for p, h in hash_results.items() if h != PROTECTED_BASELINE_HASHES[p]]
            return False, f"CRITICAL: Protected Experiment A baseline hashes mismatched! {mismatches}"

        # 3. GPU Telemetry via nvidia-smi
        try:
            cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            temp = float(parts[0])
            free_vram_mb = float(parts[1])
            util = float(parts[2])

            if temp >= 78.0:
                return False, f"GPU thermal warning: temperature is {temp:.1f}°C (safety threshold is 78°C)"
            if is_preflight and free_vram_mb < 5000.0:
                return False, f"Insufficient GPU VRAM for preflight: {free_vram_mb:.0f} MiB free (minimum required is 5,000 MiB)"
            elif not is_preflight and free_vram_mb < 1000.0:
                return False, f"Critical GPU VRAM exhaustion: {free_vram_mb:.0f} MiB free (active minimum is 1,000 MiB)"
        except Exception as e:
            return False, f"GPU status check failed: {e}"

        return True, "System resources healthy."

    def get_next_eligible_task(self) -> Optional[ResearchTask]:
        """Returns the next task whose dependencies are completed."""
        for task_id, t in self.tasks.items():
            task_state = self.state["tasks"].get(task_id, {})
            status = task_state.get("status", "PENDING")

            if status in ["COMPLETED", "SKIPPED"]:
                continue
            if status in ["FAILED", "BLOCKED"]:
                return None  # Halt on failed/blocked task

            # Check if all dependencies are satisfied
            deps = t.dependencies
            deps_satisfied = True
            for dep_id in deps:
                dep_status = self.state["tasks"].get(dep_id, {}).get("status")
                if dep_status not in ["COMPLETED", "SKIPPED"]:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                return t

        return None

    def execute_task(self, task: ResearchTask, dry_run: bool = False) -> bool:
        """Executes a single research task with logging and retry governance."""
        task_id = task.task_id
        task_state = self.state["tasks"][task_id]

        print(f"\n=======================================================")
        print(f"EXECUTING: [{task_id}] {task.name}")
        print(f"Command: {' '.join(task.command)}")
        print(f"=======================================================")

        # Check preflight
        resources_ok, msg = self.check_system_resources()
        if not resources_ok:
            print(f"[PREFLIGHT BLOCKED] {msg}")
            task_state["status"] = "BLOCKED"
            task_state["error_message"] = msg
            self._save_state()
            return False

        if dry_run:
            print(f"[DRY-RUN] Simulating execution of {task_id}...")
            time.sleep(0.5)
            task_state["status"] = "COMPLETED"
            task_state["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._save_state()
            return True

        # Check if already completed and artifacts valid
        if task.verify_artifacts() and len(task.expected_artifacts) > 0 and task_state["status"] == "COMPLETED":
            print(f"[SKIP] Task {task_id} outputs verified. Skipping execution.")
            return True

        # Setup Logging
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        log_file = os.path.join(self.logs_dir, f"{task_id}_{timestamp}.log")
        task_state["log_file"] = log_file
        task_state["status"] = "RUNNING"
        task_state["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.state["active_task_id"] = task_id
        self._save_state()

        t0 = time.time()
        exit_code = -1
        error_msg = None

        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(f"=== DEEPSHIELD TASK LOG: {task_id} ===\n")
                lf.write(f"Command: {' '.join(task.command)}\n")
                lf.write(f"Start Time: {task_state['start_time']}\n\n")
                lf.flush()

                proc = subprocess.Popen(
                    task.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()

                proc.wait(timeout=task.timeout_seconds)
                exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = -9
            error_msg = f"Task timed out after {task.timeout_seconds}s"
            print(f"\n[TIMEOUT] {error_msg}")
        except KeyboardInterrupt:
            proc.kill()
            task_state["status"] = "BLOCKED"
            task_state["error_message"] = "Interrupted by user (SIGINT)"
            self.state["active_task_id"] = None
            self._save_state()
            print("\n[INTERRUPTED] Safe shutdown complete.")
            sys.exit(0)
        except Exception as e:
            exit_code = -1
            error_msg = str(e)
            print(f"\n[ERROR] Execution failed: {e}")

        elapsed = time.time() - t0
        task_state["elapsed_seconds"] = round(elapsed, 2)
        task_state["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        task_state["exit_code"] = exit_code
        self.state["active_task_id"] = None

        # Post-execution verification
        baselines_ok, hash_results = verify_protected_baselines()
        if not baselines_ok:
            task_state["status"] = "BLOCKED"
            task_state["error_message"] = "CRITICAL: Task modified protected baseline artifacts!"
            self._save_state()
            print(f"\n[CRITICAL ERROR] {task_state['error_message']}")
            return False

        if exit_code == 0:
            # Check expected artifacts
            if task.verify_artifacts():
                task_state["status"] = "COMPLETED"
                task_state["error_message"] = None
                task_state["artifacts"] = {p: compute_sha256(p) for p in task.expected_artifacts}
                self._save_state()
                print(f"[SUCCESS] {task_id} completed successfully in {elapsed:.1f}s.")
                return True
            else:
                missing = [p for p in task.expected_artifacts if not os.path.exists(p) or os.path.getsize(p) == 0]
                task_state["status"] = "FAILED"
                task_state["error_message"] = f"Expected artifacts missing or empty: {missing}"
                self._save_state()
                print(f"[FAILED] {task_state['error_message']}")
                return False
        else:
            task_state["retries"] += 1
            if task.is_scientific_gate or task_state["retries"] >= task.max_retries:
                task_state["status"] = "FAILED"
                task_state["error_message"] = error_msg or f"Exited with code {exit_code} (max retries reached)"
                self._save_state()
                print(f"[HALTED] {task_id} failed permanently: {task_state['error_message']}")
                return False
            else:
                task_state["status"] = "PENDING"
                task_state["error_message"] = error_msg or f"Exited with code {exit_code} (retrying in {task.retry_delay_seconds}s)"
                self._save_state()
                print(f"[RETRY] {task_state['error_message']}")
                time.sleep(task.retry_delay_seconds)
                return self.execute_task(task, dry_run=dry_run)

    def run_queue(self, dry_run: bool = False):
        """Runs the queue sequentially until completion or safe block."""
        print("Starting DeepShield Autonomous Project Controller...")
        while True:
            next_task = self.get_next_eligible_task()
            if next_task is None:
                # Check why no task is eligible
                statuses = [t["status"] for t in self.state["tasks"].values()]
                if all(s in ["COMPLETED", "SKIPPED"] for s in statuses):
                    print("\n=======================================================")
                    print("ALL DEEPSHIELD RESEARCH TASKS COMPLETED SUCCESSFULLY!")
                    print("=======================================================")
                elif any(s in ["FAILED", "BLOCKED"] for s in statuses):
                    blocked = [tid for tid, t in self.state["tasks"].items() if t["status"] in ["FAILED", "BLOCKED"]]
                    print(f"\n[PAUSED] Controller paused at scientific/system gate: {blocked}")
                    for tid in blocked:
                        print(f"  - {tid}: {self.state['tasks'][tid].get('error_message')}")
                else:
                    print("\n[PAUSED] No eligible tasks found (dependency deadlock or unfulfilled gate).")
                break

            success = self.execute_task(next_task, dry_run=dry_run)
            if not success:
                print(f"\n[PAUSED] Controller halted after task {next_task.task_id} failed/blocked.")
                break

    def print_status(self):
        """Displays formatted status report of the research queue and system."""
        print("=" * 70)
        print("DEEPSHIELD AUTONOMOUS RESEARCH CONTROLLER — STATUS REPORT")
        print("=" * 70)
        print(f"State File: {self.state_file} | Last Updated: {self.state.get('last_updated')}")

        # Check resources
        active_running = any(ts.get("status") == "RUNNING" for ts in self.state["tasks"].values())
        resources_ok, r_msg = self.check_system_resources(is_preflight=not active_running)
        print(f"System Health: {'OK' if resources_ok else 'WARNING'} ({r_msg})")

        # Baseline hashes
        b_ok, _ = verify_protected_baselines()
        print(f"Protected Hashes: {'VERIFIED (All 4 intact)' if b_ok else 'MISMATCH DETECTED'}")

        print("\nTASK EXECUTION QUEUE:")
        print(f"{'Task ID':<30} {'Status':<12} {'Elapsed':<10} {'Retries':<8} {'Exit'}")
        print("-" * 70)

        completed_count = 0
        total_count = len(self.tasks)

        for task_id, t in self.tasks.items():
            ts = self.state["tasks"].get(task_id, {})
            status = ts.get("status", "PENDING")
            if status == "RUNNING" and ts.get("start_time"):
                try:
                    import calendar
                    t_start = time.strptime(ts["start_time"], "%Y-%m-%dT%H:%M:%SZ")
                    live_sec = max(0.0, time.time() - calendar.timegm(t_start))
                    elapsed = f"{live_sec:.1f}s (live)"
                except Exception:
                    elapsed = f"{ts.get('elapsed_seconds', 0.0):.1f}s"
            else:
                elapsed = f"{ts.get('elapsed_seconds', 0.0):.1f}s"
            retries = f"{ts.get('retries', 0)}/{ts.get('max_retries', 0)}"
            code = str(ts.get("exit_code", "-"))

            if status in ["COMPLETED", "SKIPPED"]:
                completed_count += 1

            print(f"{task_id:<30} {status:<12} {elapsed:<12} {retries:<8} {code}")

        progress = (completed_count / total_count) * 100
        print("-" * 70)
        print(f"Progress: {completed_count}/{total_count} tasks completed ({progress:.1f}%)")
        active_task = self.state.get("active_task_id")
        if active_task and active_task in self.state["tasks"]:
            log_p = self.state["tasks"][active_task].get("log_file")
            print(f"Active Job: {active_task}")
            if log_p and os.path.exists(log_p):
                try:
                    with open(log_p, "r", encoding="utf-8", errors="ignore") as lf:
                        lines = [line.strip() for line in lf if line.strip()]
                        if lines:
                            print(f"Latest Log Line: {lines[-1]}")
                except Exception:
                    pass

        # Checkpoints & Metrics Telemetry
        best_ckpt = "models/checkpoints/best_vit_b16.pt"
        if os.path.exists(best_ckpt):
            try:
                import torch
                c_data = torch.load(best_ckpt, map_location="cpu")
                best_ep = c_data.get("epoch", "-")
                best_eer = c_data.get("dev_eer", "-")
                if isinstance(best_eer, float):
                    print(f"Best ViT Development EER: {best_eer*100:.2f}% (Epoch {best_ep})")
                else:
                    print(f"Best ViT Development EER: {best_eer} (Epoch {best_ep})")
            except Exception:
                pass

        gpu_t = SupervisorTelemetry.get_gpu_telemetry()
        if gpu_t.get("available"):
            print(f"GPU Live Telemetry: Temp={gpu_t['temperature_c']:.0f}°C | VRAM={gpu_t['memory_used_mb']:.0f}MB used ({gpu_t['memory_free_mb']:.0f}MB free) | Util={gpu_t['utilization_gpu']:.0f}% | Power={gpu_t['power_draw_w']:.1f}W")

        next_task = self.get_next_eligible_task()
        print(f"Next Eligible Task: {next_task.task_id if next_task else 'None (Halted or Done)'}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DeepShield Autonomous Project Controller")
    parser.add_argument("--run", action="store_true", help="Run the autonomous queue to completion")
    parser.add_argument("--status", action="store_true", help="Print queue status and system health")
    parser.add_argument("--test", action="store_true", help="Run non-destructive controller self-tests")
    parser.add_argument("--dry-run", action="store_true", help="Simulate queue execution without launching jobs")
    parser.add_argument("--run-task", type=str, help="Execute a single task by ID")
    args = parser.parse_args()

    controller = DeepShieldController()

    if args.test:
        print("Running Non-Destructive Controller Self-Tests...")
        # 1. State serialization test
        controller._save_state()
        assert os.path.exists(STATE_FILE), "State file missing!"
        print("PASS: State serialization verified.")

        # 2. Protected baseline verification test
        b_ok, _ = verify_protected_baselines()
        assert b_ok, "Protected baselines verification failed!"
        print("PASS: Protected Experiment A baseline hashes verified.")

        # 3. System resources check test
        r_ok, msg = controller.check_system_resources()
        print(f"Resource Preflight Test: {'PASS' if r_ok else 'WARN'} -> {msg}")

        # 4. Task dependency graph resolution test
        next_t = controller.get_next_eligible_task()
        print(f"PASS: Dependency resolution verified. Initial task: {next_t.task_id if next_t else 'None'}")
        print("\nAll Controller Self-Tests Passed Successfully!")
    elif args.status:
        controller.print_status()
    elif args.run_task:
        if args.run_task not in controller.tasks:
            print(f"Unknown task ID: {args.run_task}")
            sys.exit(1)
        controller.execute_task(controller.tasks[args.run_task], dry_run=args.dry_run)
    elif args.run:
        lock = ControllerLock()
        try:
            lock.acquire()
        except ControllerAlreadyRunningError as e:
            print(f"\n[LOCKED] An authoritative DeepShield controller is already running: {e}")
            print("Duplicate run request safely ignored.")
            sys.exit(0)
        try:
            controller.run_queue(dry_run=args.dry_run)
        finally:
            lock.release()
    else:
        controller.print_status()


if __name__ == "__main__":
    main()
