"""Fail-closed OS sandbox with explicit syscall policy and bounded pipe RPC.

Vendored from ``learnagents/runtime/sandbox.py`` (agent_methods_bench@2af2879, sha256 67c5c4b9…). The isolation
boundary is unchanged: prlimit (CPU/AS/FSIZE/NOFILE/CORE) -> bwrap ``--unshare-all --die-with-parent
--new-session --cap-drop ALL`` with read-only ``/usr``, ``/lib``, ``/lib64`` and the interpreter prefix, tmpfs
``/tmp`` and ``/work``, a scrubbed environment, and a seccomp filter denying sockets, process/thread creation,
ptrace, mounts, namespaces, bpf, keyrings and module loading. Changes (see ``_vendor/PROVENANCE.json``):

* ``python_prefix`` defaults to the certified interpreter prefix, never ``sys.prefix`` (uv venv symlinks break
  inside bwrap);
* an explicit ``RLIMIT_STACK`` (the login node inherits ``unlimited``);
* candidate stdout/stderr overflow is truncated (flagged) instead of failing the whole batch;
* every ``SandboxError`` carries a ``kind``, the request ``phase`` ("startup" | "trusted" | "candidate") and the
  process return code, so graders can separate limit hits (real verdicts) from infrastructure faults;
* a host-enforced scratch quota: bwrap 0.4.0 cannot size its tmpfs mounts (default: half of host RAM), so the host
  polls the sandbox's own ``/tmp`` + ``/work`` usage through ``/proc/<pid>/root`` every 20 ms while a request is in
  flight and kills the sandbox above ``limits.scratch_bytes``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import selectors
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CERTIFIED_PYTHON_PREFIX = "/orcd/home/002/mabdel03/conda_envs/asys_env"

# SandboxError.kind values
SPAWN_FAILED = "spawn_failed"          # tools/interpreter missing, seccomp build or exec failure
STARTUP_FAILED = "startup_failed"      # init handshake did not complete (bwrap setup, interpreter startup)
WALL_TIMEOUT = "wall_timeout"          # host wall-clock limit for one request expired
PROCESS_EXITED = "process_exited"      # sandbox process exited while a request was in flight
PROTOCOL_ERROR = "protocol_error"      # malformed / oversized protocol traffic
INPUT_FAILED = "input_failed"          # request could not be written (pipe closed or stalled)
SCRATCH_LIMIT = "scratch_limit"        # /tmp + /work usage exceeded limits.scratch_bytes (host-enforced)
SCRATCH_POLL_SECONDS = 0.02


class SandboxError(RuntimeError):
    def __init__(self, message: str = "", *, kind: str = PROTOCOL_ERROR, phase: str = "startup",
                 returncode: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.phase = phase
        self.returncode = returncode


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 10
    wall_seconds: float = 15
    memory_bytes: int = 2 * 1024 ** 3
    file_bytes: int = 64 * 1024 ** 2
    output_bytes: int = 64 * 1024
    max_protocol_bytes: int = 2 * 1024 ** 2
    max_open_files: int = 64
    max_rpc_calls: int = 128
    stack_bytes: int = 64 * 1024 ** 2
    scratch_bytes: int = 64 * 1024 ** 2    # bwrap 0.4.0 has no tmpfs --size: the host polls and kills on overflow


def seccomp_fd() -> int:
    """Build an allow-by-default compatibility policy with dangerous calls denied.

    Namespaces/mounts provide the filesystem boundary. Socket and process creation
    are denied in the registered single-process Python task profile.
    """
    lib = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)
    if not ctx:
        raise SandboxError("seccomp_init failed", kind=SPAWN_FAILED)
    fd, write_fd = os.pipe()
    try:
        for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                     "clone", "clone3", "fork", "vfork", "ptrace", "process_vm_readv",
                     "process_vm_writev", "mount", "umount2", "pivot_root", "setns", "unshare",
                     "bpf", "perf_event_open", "userfaultfd", "keyctl", "add_key", "request_key",
                     "kexec_load", "init_module", "finit_module", "delete_module", "reboot",
                     "open_by_handle_at", "name_to_handle_at", "io_uring_setup"):
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, nr, 0) != 0:
                raise SandboxError(f"Cannot deny syscall {name}", kind=SPAWN_FAILED)
        if lib.seccomp_export_bpf(ctx, write_fd) != 0:
            raise SandboxError("Cannot export seccomp filter", kind=SPAWN_FAILED)
        return fd
    except BaseException:
        os.close(fd)
        raise
    finally:
        os.close(write_fd)
        lib.seccomp_release(ctx)


class Sandbox:
    def __init__(self, public: dict | None = None, *, allow_children: bool = False,
                 limits: SandboxLimits | None = None, python_prefix: str | None = None,
                 rpc_handler: Callable[[str, dict], object] | None = None):
        self.limits = limits or SandboxLimits()
        self.rpc_handler = rpc_handler
        self.buffer = b""
        self.stderr_buffer = b""
        self.stderr_truncated = False
        self.closed = False
        self.process = None
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise SandboxError("bubblewrap is required; host execution is forbidden", kind=SPAWN_FAILED)
        prlimit = shutil.which("prlimit")
        if not prlimit:
            raise SandboxError("prlimit is required for externally enforced sandbox resource limits",
                               kind=SPAWN_FAILED)
        prefix = Path(python_prefix or CERTIFIED_PYTHON_PREFIX).resolve()
        python = prefix / "bin" / "python"
        if not python.exists():
            raise SandboxError(f"No dedicated interpreter at {python}", kind=SPAWN_FAILED)
        bootstrap = Path(__file__).with_name("sandbox_worker.py").resolve()
        fd = seccomp_fd()
        args = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib64", "/lib64",
                "--ro-bind", "/lib", "/lib", "--ro-bind", str(prefix), "/runtime",
                "--ro-bind", str(bootstrap), "/runner.py", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--tmpfs", "/work", "--chdir", "/work",
                "--seccomp", str(fd), "/runtime/bin/" + python.name, "-I", "-u", "/runner.py"]
        lim = self.limits
        args = [prlimit, f"--cpu={lim.cpu_seconds}:{lim.cpu_seconds+1}",
                f"--as={lim.memory_bytes}:{lim.memory_bytes}",
                f"--fsize={lim.file_bytes}:{lim.file_bytes}",
                f"--nofile={lim.max_open_files}:{lim.max_open_files}",
                f"--stack={lim.stack_bytes}:{lim.stack_bytes}", "--core=0:0", "--", *args]
        try:
            self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, pass_fds=(fd,),
                                            start_new_session=True,
                                            env={"PATH": "/runtime/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                                                 "LD_LIBRARY_PATH": "/runtime/lib", "OPENBLAS_NUM_THREADS": "1",
                                                 "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        except OSError as exc:
            raise SandboxError(f"Cannot start sandbox: {exc}", kind=SPAWN_FAILED) from exc
        finally:
            os.close(fd)
        self.selector = selectors.DefaultSelector()
        os.set_blocking(self.process.stdin.fileno(), False)
        for stream, label in ((self.process.stdout, "stdout"), (self.process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, label)
        self.worker_pid = None
        self._scratch_checked = 0.0
        try:
            self._request({"op": "init", "public": public or {}, "allow_children": allow_children,
                           "output_limit": self.limits.output_bytes}, phase="startup")
            self.worker_pid = self._innermost_pid()
            if self.worker_pid is None or self._sees_host_root(self.worker_pid):
                raise SandboxError("Cannot locate the sandboxed worker for the scratch quota", kind=STARTUP_FAILED)
        except SandboxError as exc:
            if exc.kind != SPAWN_FAILED:
                exc.kind = STARTUP_FAILED
            raise

    def _cpu_seconds(self):
        """Kernel-accounted CPU for this exact process tree, read outside sandbox."""
        pending, seen, ticks = [self.process.pid], set(), 0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                ticks += int(fields[11])+int(fields[12])
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                pending.extend(int(p) for p in children)
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                continue
        return ticks / os.sysconf("SC_CLK_TCK")

    def _innermost_pid(self) -> int | None:
        pid = self.process.pid
        for _ in range(8):
            try:
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except (FileNotFoundError, ProcessLookupError):
                return None
            if not children:
                return pid
            pid = int(children[-1])
        return pid

    @staticmethod
    def _sees_host_root(pid: int) -> bool:
        try:
            inner, host = os.stat(f"/proc/{pid}/root"), os.stat("/")
        except OSError:
            return True
        return (inner.st_dev, inner.st_ino) == (host.st_dev, host.st_ino)

    def scratch_used_bytes(self) -> int | None:
        """Bytes used in the sandbox's own tmpfs mounts, read from the host through /proc/<pid>/root."""
        if self.worker_pid is None:
            return None
        used = 0
        for mount in ("tmp", "work"):
            try:
                stat = os.statvfs(f"/proc/{self.worker_pid}/root/{mount}")
            except OSError:
                return None
            used += (stat.f_blocks - stat.f_bfree) * stat.f_frsize
        return used

    def _check_scratch(self, phase: str) -> None:
        now = time.monotonic()
        if self.worker_pid is None or now - self._scratch_checked < SCRATCH_POLL_SECONDS:
            return
        self._scratch_checked = now
        used = self.scratch_used_bytes()
        if used is not None and used > self.limits.scratch_bytes:
            raise SandboxError(f"Sandbox scratch usage {used} bytes exceeds {self.limits.scratch_bytes}",
                               kind=SCRATCH_LIMIT, phase=phase)

    def _send(self, obj: dict, phase: str) -> None:
        payload = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8", "surrogatepass")
        if len(payload) > self.limits.max_protocol_bytes:
            raise SandboxError("Sandbox input exceeds protocol cap", kind=PROTOCOL_ERROR, phase=phase)
        try:
            deadline = time.monotonic()+self.limits.wall_seconds
            sent = 0
            with selectors.DefaultSelector() as writable:
                writable.register(self.process.stdin, selectors.EVENT_WRITE)
                while sent < len(payload):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise SandboxError("Sandbox input pipe wall-time limit", kind=INPUT_FAILED, phase=phase)
                    if not writable.select(timeout=min(0.1, remaining)):
                        continue
                    try:
                        sent += os.write(self.process.stdin.fileno(), payload[sent:sent+65536])
                    except BlockingIOError:
                        continue
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise SandboxError("Sandbox process exited: " + self.stderr_buffer.decode(errors="replace"),
                               kind=PROCESS_EXITED, phase=phase, returncode=self.process.returncode) from exc
        except SandboxError:
            self.close()
            raise

    def _request(self, obj: dict, *, phase: str = "candidate") -> dict:
        cpu_started = self._cpu_seconds()
        self._send(obj, phase)
        deadline = time.monotonic() + self.limits.wall_seconds
        rpc_count = 0
        try:
            while time.monotonic() < deadline:
                if b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    try:
                        message = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        raise SandboxError("Malformed sandbox protocol", kind=PROTOCOL_ERROR, phase=phase) from exc
                    if not isinstance(message, dict):
                        raise SandboxError("Sandbox protocol response must be an object", kind=PROTOCOL_ERROR,
                                           phase=phase)
                    if message.get("type") == "rpc":
                        rpc_count += 1
                        if rpc_count > self.limits.max_rpc_calls:
                            raise SandboxError("Sandbox RPC count limit", kind=PROTOCOL_ERROR, phase=phase)
                        try:
                            if self.rpc_handler is None:
                                raise SandboxError("No tool capability in this sandbox")
                            result = self.rpc_handler(message["kind"], message["payload"])
                            self._send({"ok": True, "result": result}, phase)
                        except Exception as exc:
                            self._send({"ok": False, "error": str(exc)}, phase)
                        # LM time is charged by the broker, not sandbox code wall time.
                        deadline = time.monotonic() + self.limits.wall_seconds
                        continue
                    if message.get("type") != "result":
                        raise SandboxError("Unknown sandbox response", kind=PROTOCOL_ERROR, phase=phase)
                    stdout = message.get("stdout", "")
                    if not isinstance(stdout, str):
                        raise SandboxError("Sandbox stdout must be text", kind=PROTOCOL_ERROR, phase=phase)
                    encoded = stdout.encode("utf-8", "surrogatepass")
                    if len(encoded) > self.limits.output_bytes:
                        # Truncate instead of failing: output volume is not a verdict.
                        message["stdout"] = encoded[:self.limits.output_bytes].decode("utf-8", "ignore")
                        message["stdout_truncated"] = True
                    # Ignore any telemetry emitted by candidate code; measure host
                    # kernel process counters instead of process-local Python clocks.
                    message["_resource_usage"] = {
                        "sandbox_cpu_seconds": max(0.0, self._cpu_seconds()-cpu_started),
                        "source": "host_proc_process_tree", "clock_resolution_seconds": 1/os.sysconf("SC_CLK_TCK")}
                    return message
                self._check_scratch(phase)
                events = self.selector.select(timeout=min(SCRATCH_POLL_SECONDS, max(0, deadline - time.monotonic())))
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        self.selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        self.buffer += chunk
                        if len(self.buffer) > self.limits.max_protocol_bytes:
                            raise SandboxError("Sandbox protocol output limit", kind=PROTOCOL_ERROR, phase=phase)
                    else:
                        room = self.limits.output_bytes - len(self.stderr_buffer)
                        if room > 0:
                            self.stderr_buffer += chunk[:room]
                        if len(chunk) > max(room, 0):
                            self.stderr_truncated = True   # drained and discarded, never a failure
                if self.process.poll() is not None and b"\n" not in self.buffer:
                    # Drain whatever the exited process left in its pipes before declaring the exit.
                    for key in list(self.selector.get_map().values()):
                        while key.data == "stdout" and len(self.buffer) <= self.limits.max_protocol_bytes:
                            try:
                                chunk = os.read(key.fileobj.fileno(), 65536)
                            except (BlockingIOError, OSError):
                                break
                            if not chunk:
                                break
                            self.buffer += chunk
                    if b"\n" in self.buffer:
                        continue
                    raise SandboxError("Sandbox exited: " + self.stderr_buffer.decode(errors="replace")[-2000:],
                                       kind=PROCESS_EXITED, phase=phase, returncode=self.process.returncode)
            raise SandboxError("Sandbox wall-time limit", kind=WALL_TIMEOUT, phase=phase)
        except BaseException:
            self.close()
            raise

    def execute(self, code: str, *, phase: str = "candidate") -> dict:
        try:
            return self._request({"op": "exec", "code": code}, phase=phase)
        except SandboxError as exc:
            exc.execution_failure = True
            raise

    def invoke(self, value: object, entry_point: str = "solve", *, phase: str = "candidate") -> dict:
        try:
            return self._request({"op": "invoke", "input": value, "entry_point": entry_point}, phase=phase)
        except SandboxError as exc:
            exc.execution_failure = True
            raise

    def close(self):
        if self.closed or self.process is None:
            return
        self.closed = True
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
