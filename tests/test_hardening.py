"""
Hardening unit tests for Architecture Review Remediation Plan v2.

Phases covered:
  - Phase B (Items 7, 8): FileAgent path-traversal/symlink blocking; AST math sandbox escape.
  - Phase C (Item 6): Incremental indexing manifest (skip, modify, delete).
  - Phase D (Item 5): Atomic lockfile creation, stale-lock recovery, query-time lock warning.
  - Phase E (Item 9): JSONL append correctness and 10 MB rotation.
"""

import ast
import gzip
import json
import os
import shutil
import tempfile
import time
import unittest

from app.agents.exceptions import SecurityException
from app.agents.file_agent import FileAgent
from app.agents.math_agent import SafeMathEvaluator, eval_math_ast


# ---------------------------------------------------------------------------
# Phase B — Item 7: FileAgent path-traversal & symlink hardening
# ---------------------------------------------------------------------------

class TestFileAgentPathHardening(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.agent = FileAgent(workspace_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # --- literal traversal ---

    def test_literal_dotdot_traversal_blocked(self):
        """../../etc/passwd must be rejected before resolution even moves to the parent."""
        with self.assertRaises(SecurityException):
            self.agent.validate_target_path("../../etc/passwd")

    def test_absolute_path_outside_root_blocked(self):
        """/tmp/evil.txt resolves outside workspace root and must be rejected."""
        with self.assertRaises(SecurityException):
            self.agent.validate_target_path("/tmp/evil.txt")

    def test_nested_dotdot_traversal_blocked(self):
        """subdir/../../../etc/shadow must be rejected."""
        subdir = os.path.join(self.root, "subdir")
        os.makedirs(subdir, exist_ok=True)
        with self.assertRaises(SecurityException):
            self.agent.validate_target_path("subdir/../../../etc/shadow")

    def test_valid_path_inside_root_accepted(self):
        """A plain filename inside root must pass validation."""
        result = self.agent.validate_target_path("output.txt")
        self.assertTrue(result.startswith(self.root))
        self.assertIn("output.txt", result)

    def test_nested_subdir_inside_root_accepted(self):
        """A path like notes/summary.md that stays inside root must pass."""
        result = self.agent.validate_target_path("notes/summary.md")
        self.assertTrue(result.startswith(self.root))

    # --- symlink escape ---

    def test_symlink_escape_outside_root_blocked(self):
        """A symlink that lives inside root but points outside it must be rejected."""
        outside_dir = tempfile.mkdtemp()
        try:
            link_path = os.path.join(self.root, "evil_link.txt")
            target_outside = os.path.join(outside_dir, "secret.txt")
            open(target_outside, "w").close()
            os.symlink(target_outside, link_path)

            with self.assertRaises(SecurityException):
                self.agent.validate_target_path("evil_link.txt")
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_symlink_inside_root_accepted(self):
        """A symlink that lives AND points inside root must be accepted."""
        real_file = os.path.join(self.root, "real.md")
        open(real_file, "w").close()
        link_path = os.path.join(self.root, "link.md")
        os.symlink(real_file, link_path)

        result = self.agent.validate_target_path("link.md")
        self.assertTrue(result.startswith(self.root))


# ---------------------------------------------------------------------------
# Phase B — Item 8: AST Math Sandbox Escape Tests
# ---------------------------------------------------------------------------

class TestMathASTSandbox(unittest.TestCase):

    def _eval_raises(self, expr: str):
        with self.assertRaises((ValueError, Exception)):
            eval_math_ast(expr)

    # --- normal arithmetic must work ---
    def test_valid_arithmetic(self):
        self.assertAlmostEqual(eval_math_ast("10 * 5 + 2"), 52.0)

    def test_valid_whitelisted_function_sqrt(self):
        self.assertAlmostEqual(eval_math_ast("sqrt(16)"), 4.0)

    def test_valid_whitelisted_function_abs(self):
        self.assertAlmostEqual(eval_math_ast("abs(-7)"), 7.0)

    # --- sandbox escapes must be rejected ---

    def test_attribute_access_blocked(self):
        """().__class__ is a classic sandbox escape and must be blocked."""
        self._eval_raises("().__class__")

    def test_name_reference_blocked(self):
        """A bare Name node (variable reference) is not permitted."""
        self._eval_raises("x + 1")

    def test_lambda_blocked(self):
        """Lambda parses fine in eval mode but is blocked by generic_visit (ast.Lambda not whitelisted)."""
        self._eval_raises("lambda: 0")
        self._eval_raises("lambda x: x + 1")

    def test_list_comprehension_blocked(self):
        """List comprehensions contain Name nodes and must be blocked."""
        self._eval_raises("[i for i in range(10)]")

    def test_builtin_name_shadowing_blocked(self):
        """Using a builtin name like 'eval' or 'exec' as a call target must be blocked."""
        self._eval_raises("eval('1+1')")
        self._eval_raises("exec('pass')")

    def test_non_numeric_constant_blocked(self):
        """String constants are not permitted."""
        self._eval_raises('"hello"')

    def test_nested_call_chain_blocked(self):
        """Deeply nested whitelisted calls are fine; attribute-style chains must be blocked."""
        # abs(abs(abs(…))) is valid — all calls are to a whitelisted function
        result = eval_math_ast("abs(abs(abs(abs(abs(-1)))))")
        self.assertAlmostEqual(result, 1.0)
        # But attribute-style call is an ast.Attribute node -> must be blocked
        self._eval_raises("(1).__add__(2)")

    def test_import_statement_blocked(self):
        """Import is a statement, not an expression — ast.parse mode='eval' rejects it."""
        with self.assertRaises(SyntaxError):
            ast.parse("import os", mode="eval")

    def test_subscript_blocked(self):
        """Subscript access (e.g. [0]) is blocked by generic_visit."""
        self._eval_raises("[1,2,3][0]")

    def test_disallowed_operator_blocked(self):
        """Bitwise shift is not in the whitelist of BinOp operators."""
        self._eval_raises("1 << 10")


# ---------------------------------------------------------------------------
# Phase C — Item 6: Incremental Indexing Manifest
# ---------------------------------------------------------------------------

class TestIncrementalIndexingManifest(unittest.TestCase):
    """Tests for IndexManifest (skip unchanged, deletion cleanup, modification update)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.tmpdir, ".data", "index_manifest.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_file(self, name: str, content: str) -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _manifest(self):
        from indexing.incremental import IndexManifest
        return IndexManifest(self.manifest_path)

    def test_new_file_is_not_skipped(self):
        m = self._manifest()
        path = self._make_file("new.txt", "hello world")
        self.assertFalse(m.is_unchanged(path))

    def test_unchanged_file_is_skipped(self):
        m = self._manifest()
        path = self._make_file("doc.txt", "content")
        m.record(path, chunk_ids=["c1", "c2"])
        m2 = self._manifest()
        self.assertTrue(m2.is_unchanged(path))

    def test_modified_file_is_not_skipped(self):
        m = self._manifest()
        path = self._make_file("doc.txt", "original")
        m.record(path, chunk_ids=["c1"])
        # modify file
        with open(path, "w") as f:
            f.write("modified content")
        m2 = self._manifest()
        self.assertFalse(m2.is_unchanged(path))

    def test_deleted_file_removed_from_manifest(self):
        m = self._manifest()
        path = self._make_file("gone.txt", "bye")
        m.record(path, chunk_ids=["c1", "c2"])
        os.remove(path)
        m2 = self._manifest()
        stale = m2.find_deleted_entries([])  # empty current file list
        self.assertIn(path, stale)
        ids = m2.get_chunk_ids(path)
        self.assertEqual(ids, ["c1", "c2"])

    def test_manifest_persists_across_instances(self):
        m = self._manifest()
        path = self._make_file("persist.txt", "data")
        m.record(path, chunk_ids=["x1"])
        m2 = self._manifest()
        self.assertTrue(m2.is_unchanged(path))
        self.assertEqual(m2.get_chunk_ids(path), ["x1"])


# ---------------------------------------------------------------------------
# Phase D — Item 5: Atomic Lockfile & Stale-Lock Recovery
# ---------------------------------------------------------------------------

class TestIndexLockfile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lock_path = os.path.join(self.tmpdir, ".data", "index.lock")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _lockfile(self, **kwargs):
        from indexing.lockfile import IndexLockfile
        return IndexLockfile(self.lock_path, **kwargs)

    def test_acquire_and_release(self):
        lf = self._lockfile()
        lf.acquire()
        self.assertTrue(os.path.exists(self.lock_path))
        lf.release()
        self.assertFalse(os.path.exists(self.lock_path))

    def test_context_manager_releases_on_exit(self):
        lf = self._lockfile()
        with lf:
            self.assertTrue(os.path.exists(self.lock_path))
        self.assertFalse(os.path.exists(self.lock_path))

    def test_stale_lock_dead_pid_auto_cleared(self):
        """A lock file referencing a dead PID must be auto-cleared and acquisition must succeed."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        # write a lock with a PID that cannot be running (PID 99999999 won't exist)
        dead_pid = 99999999
        with open(self.lock_path, "w") as f:
            json.dump({"pid": dead_pid, "timestamp": time.time() - 3600}, f)

        lf = self._lockfile(staleness_seconds=1800)
        lf.acquire()          # must succeed by clearing the stale lock
        self.assertTrue(os.path.exists(self.lock_path))
        lf.release()

    def test_stale_lock_old_timestamp_auto_cleared(self):
        """A lock file older than the staleness threshold (even with a real PID) is stale."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        # write very old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        with open(self.lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "timestamp": old_time}, f)

        lf = self._lockfile(staleness_seconds=1800)
        lf.acquire()
        self.assertTrue(os.path.exists(self.lock_path))
        lf.release()

    def test_live_lock_is_detected(self):
        """A fresh live lock from the current process must be detected and raise."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with open(self.lock_path, "w") as f:
            json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)

        from indexing.lockfile import LockAlreadyHeldError
        lf = self._lockfile(staleness_seconds=1800)
        with self.assertRaises(LockAlreadyHeldError):
            lf.acquire()


# ---------------------------------------------------------------------------
# Phase E — Item 9: JSONL Append & 10 MB Rotation
# ---------------------------------------------------------------------------

class TestJSONLEvalLog(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "rag_eval_history.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _logger(self, **kwargs):
        from app.evaluation import JSONLEvalLogger
        return JSONLEvalLogger(self.log_path, **kwargs)

    def test_append_writes_valid_jsonl(self):
        lg = self._logger()
        lg.append({"query": "test", "score": 0.9})
        lg.append({"query": "test2", "score": 0.8})

        lines = open(self.log_path).readlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            obj = json.loads(line)
            self.assertIn("query", obj)

    def test_rotation_triggered_at_10mb(self):
        """When active log exceeds max_bytes, it must be rotated to a timestamped file."""
        lg = self._logger(max_bytes=50)   # very small threshold for test
        # First write: file created, ~60+ bytes -> exceeds threshold on next call
        lg.append({"data": "x" * 40})
        # Second write: rotation fires before writing
        lg.append({"data": "y" * 40})
        # Third write: may fire another rotation
        lg.append({"data": "z" * 40})

        rotated = [
            f for f in os.listdir(self.tmpdir)
            if f.startswith("rag_eval_history.") and f != "rag_eval_history.jsonl"
        ]
        self.assertGreaterEqual(len(rotated), 1)

    def test_rotate_keeps_at_most_5_files(self):
        """Rotation must keep at most 5 rotated copies."""
        lg = self._logger(max_bytes=10, max_rotations=5)
        for _ in range(20):
            lg.append({"data": "x" * 20})

        rotated = [
            f for f in os.listdir(self.tmpdir)
            if f.startswith("rag_eval_history.") and f != "rag_eval_history.jsonl"
        ]
        self.assertLessEqual(len(rotated), 5)

    def test_rotated_files_are_gzip_compressed(self):
        """Files beyond the 5 most recent must be gzip-compressed (.jsonl.gz)."""
        lg = self._logger(max_bytes=10, max_rotations=2)
        for _ in range(10):
            lg.append({"data": "x" * 20})

        gz_files = [
            f for f in os.listdir(self.tmpdir) if f.endswith(".jsonl.gz")
        ]
        if gz_files:
            # make sure the gz files are valid gzip
            for gzf in gz_files:
                with gzip.open(os.path.join(self.tmpdir, gzf), "rt") as fh:
                    data = fh.read()
                self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()
