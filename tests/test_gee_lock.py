from __future__ import annotations

import os

import pytest

from rivwq.gee import lock


def test_second_acquire_is_refused_while_the_owner_is_alive(tmp_path):
    path = tmp_path / "run.lock"
    lock.acquire(path)
    with pytest.raises(RuntimeError):
        lock.acquire(path)
    assert path.read_text(encoding="utf-8") == str(os.getpid())


def test_stale_lock_is_replaced(tmp_path):
    path = tmp_path / "run.lock"
    path.write_text("999999999", encoding="utf-8")
    lock.acquire(path)
    assert path.read_text(encoding="utf-8") == str(os.getpid())


def test_garbage_lock_is_replaced(tmp_path):
    path = tmp_path / "run.lock"
    path.write_text("not a pid", encoding="utf-8")
    lock.acquire(path)
    assert path.exists()


def test_release_allows_a_new_acquire_and_is_safe_twice(tmp_path):
    path = tmp_path / "run.lock"
    lock.acquire(path)
    lock.release(path)
    lock.release(path)
    lock.acquire(path)
    assert path.exists()
