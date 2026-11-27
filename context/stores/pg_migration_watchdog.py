"""Supervise one offline migration and stop it when disk headroom is exhausted.

This is a periodic alarm, not a filesystem quota: intervening writes, blocked
kernel calls, PostgreSQL WAL and rollback can consume additional space. The
caller must identify every snapshot, PostgreSQL and WAL filesystem correctly,
fence all writers, and inspect a possibly partial destination on any failure.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import shutil
import signal
import sys
from pathlib import Path


class DiskReserveError(RuntimeError):
    """The migration did not complete because disk reserve cannot be assured."""


class MigrationWorkerError(RuntimeError):
    """The supervised migration failed or its outcome is unknown."""


class _MigrationCancelled(BaseException):
    pass


def _worker(send, target, payload, parent_pid):
    def cancelled(_signal, _frame):
        raise _MigrationCancelled()

    try:
        signal.signal(signal.SIGTERM, cancelled)
        # Linux: the worker must not continue unattended after supervisor death.
        if sys.platform.startswith('linux'):
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), 'cannot bind worker to supervisor')
            if os.getppid() != parent_pid:
                raise _MigrationCancelled()
        send.send(('ready', None))
        result = target(payload)
        send.send(('success', result))
    except BaseException as exc:
        # Error details may contain a DSN; send only a category, never secrets.
        try:
            send.send(('failed', type(exc).__name__))
        except (BrokenPipeError, OSError):
            pass
    finally:
        send.close()


def _run_guarded(target, payload, *, paths, min_free_bytes, poll_seconds=1.0,
                 stop_seconds=10.0, disk_free=None):
    """Run exactly one spawned child, with bounded polling and termination."""
    if type(min_free_bytes) is not int or not 1 <= min_free_bytes <= 2**50:
        raise ValueError('positive bounded disk reserve required')
    if not 0.01 <= poll_seconds <= 60 or not 0.1 <= stop_seconds <= 60:
        raise ValueError('invalid bounded watchdog timing')
    if not paths:
        raise ValueError('at least one monitored filesystem is required')
    watched = [Path(path).resolve(strict=True) for path in paths]
    if any(not path.is_dir() for path in watched):
        raise ValueError('monitored paths must be existing directories')
    if disk_free is None:
        disk_free = lambda path: shutil.disk_usage(path).free  # noqa: E731

    def check():
        try:
            insufficient = any(disk_free(path) <= min_free_bytes for path in watched)
        except OSError:
            raise DiskReserveError('cannot check disk reserve; migration outcome may be partial') from None
        if insufficient:
            raise DiskReserveError('disk reserve reached; migration outcome may be partial')

    def stop():
        if worker.is_alive():
            worker.terminate()  # signals only the exact managed child, never a process group
            worker.join(stop_seconds)
        if worker.is_alive():
            worker.kill()       # closes the child's sockets even if rollback is blocked
            worker.join(stop_seconds)
        if worker.is_alive():
            raise MigrationWorkerError('migration child could not be stopped')

    check()
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_worker, args=(send, target, payload, os.getpid()))
    started = False
    try:
        worker.start()
        started = True
        send.close()
        ready = False
        while True:
            check()
            if receive.poll(poll_seconds):
                try:
                    kind, result = receive.recv()
                except EOFError:
                    raise MigrationWorkerError('migration child exited without result') from None
                if kind == 'ready' and not ready:
                    ready = True
                    continue
                if kind == 'success' and ready:
                    worker.join(stop_seconds)
                    if worker.is_alive() or worker.exitcode != 0:
                        raise MigrationWorkerError('migration child did not finish cleanly')
                    check()
                    return result
                if kind == 'failed':
                    raise MigrationWorkerError(f'migration child failed ({result}); outcome may be partial')
                raise MigrationWorkerError('invalid migration child response')
            if not worker.is_alive():
                raise MigrationWorkerError('migration child exited without result; outcome may be partial')
    finally:
        if started:
            stop()
        else:
            send.close()
        receive.close()


def _migrate(payload):
    from .pg_offline_migration import migrate_offline

    return migrate_offline(**payload)


def migrate_offline_guarded(*, source_path, snapshot_path, dsn, database, system_identifier,
                            database_oid, max_snapshot_bytes, postgres_storage_paths,
                            min_free_bytes, snapshot_timeout_seconds=300,
                            poll_seconds=1.0, stop_seconds=10.0, offline=False):
    """Run migrate_offline in a monitored child; never switch production runtime.

    postgres_storage_paths must identify existing directories on *every*
    PostgreSQL data and WAL filesystem; the snapshot filesystem is added here.
    Run from a main-guarded Python program: multiprocessing uses spawn. The DSN
    is transmitted through a local IPC pipe rather than a command-line argument.
    """
    if offline is not True:
        raise RuntimeError('offline migration opt-in is required')
    if not postgres_storage_paths:
        raise ValueError('PostgreSQL data and WAL monitoring paths are required')
    payload = dict(source_path=str(source_path), snapshot_path=str(snapshot_path), dsn=dsn,
                   database=database, system_identifier=system_identifier, database_oid=database_oid,
                   max_snapshot_bytes=max_snapshot_bytes, snapshot_timeout_seconds=snapshot_timeout_seconds,
                   min_free_after_bytes=min_free_bytes, offline=True)
    return _run_guarded(_migrate, payload, paths=[Path(snapshot_path).parent, *postgres_storage_paths],
                        min_free_bytes=min_free_bytes, poll_seconds=poll_seconds, stop_seconds=stop_seconds)
