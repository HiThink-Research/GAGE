from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.utils import cleanup as cleanup_mod


class CleanupRegistrationTests(unittest.TestCase):
    def test_install_signal_cleanup_registers_atexit_once_and_can_unregister(self) -> None:
        callback_a = MagicMock()
        callback_b = MagicMock()

        with ExitStack() as stack:
            register = stack.enter_context(patch("gage_eval.utils.cleanup.atexit.register"))
            stack.enter_context(patch("gage_eval.utils.cleanup.signal.signal"))
            stack.enter_context(patch.object(cleanup_mod, "_CALLBACKS", []))
            stack.enter_context(patch.object(cleanup_mod, "_INSTALLED_SIGNALS", set()))
            stack.enter_context(patch.object(cleanup_mod, "_ATEXIT_INSTALLED", False))
            stack.enter_context(patch.object(cleanup_mod, "_CLEANED_UP", False))

            unregister = cleanup_mod.install_signal_cleanup(callback_a)
            cleanup_mod.install_signal_cleanup(callback_b)

            register.assert_called_once_with(cleanup_mod._run_callbacks)
            self.assertEqual([entry.callback for entry in cleanup_mod._CALLBACKS], [callback_a, callback_b])

            unregister()

            self.assertEqual([entry.callback for entry in cleanup_mod._CALLBACKS], [callback_b])

    def test_run_callbacks_is_idempotent(self) -> None:
        callback = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("gage_eval.utils.cleanup.atexit.register"))
            stack.enter_context(patch("gage_eval.utils.cleanup.signal.signal"))
            stack.enter_context(patch.object(cleanup_mod, "_CALLBACKS", []))
            stack.enter_context(patch.object(cleanup_mod, "_INSTALLED_SIGNALS", set()))
            stack.enter_context(patch.object(cleanup_mod, "_ATEXIT_INSTALLED", False))
            stack.enter_context(patch.object(cleanup_mod, "_CLEANED_UP", False))

            cleanup_mod.install_signal_cleanup(callback)
            cleanup_mod._run_callbacks()
            cleanup_mod._run_callbacks()

            callback.assert_called_once_with()
            self.assertEqual(cleanup_mod._CALLBACKS, [])
            self.assertTrue(cleanup_mod._CLEANED_UP)


if __name__ == "__main__":
    unittest.main()
