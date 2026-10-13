"""Bounded subprocess I/O with cancellation and process-group cleanup."""

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path

from ..errors import GatewayError


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


async def run_process(
    args: list[str], *, cwd: Path, env: dict[str, str], limit: int, stdin: str = ""
) -> ProcessResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except (OSError, ValueError) as exc:
        raise GatewayError(
            "provider_unavailable",
            "Cannot launch provider executable. Check installation and executable path.",
        ) from exc
    total = 0

    async def read(stream):
        nonlocal total
        chunks = []
        while data := await stream.read(65536):
            total += len(data)
            if total > limit:
                raise GatewayError("output_too_large", "Provider output exceeded max_output_bytes.")
            chunks.append(data)
        return b"".join(chunks).decode("utf-8", errors="replace")

    async def feed():
        try:
            process.stdin.write(stdin.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    async def terminate():
        def send(sig):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                elif process.returncode is None:
                    process.terminate() if sig == signal.SIGTERM else process.kill()
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=0.5)
        except TimeoutError:
            pass
        # A child can outlive its parent or keep stdout open.
        send(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
        await process.wait()

    tasks = [
        asyncio.create_task(read(process.stdout)),
        asyncio.create_task(read(process.stderr)),
        asyncio.create_task(feed()),
        asyncio.create_task(process.wait()),
    ]
    try:
        stdout, stderr, _, code = await asyncio.gather(*tasks)
        return ProcessResult(code, stdout, stderr)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await terminate()
        await asyncio.gather(*tasks, return_exceptions=True)
