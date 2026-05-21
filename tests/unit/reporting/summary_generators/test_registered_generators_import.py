from __future__ import annotations

import pytest


@pytest.mark.fast
def test_registered_generator_modules_import_after_v2_cutover() -> None:
    import gage_eval.reporting.summary_generators.appworld  # noqa: F401
    import gage_eval.reporting.summary_generators.arena  # noqa: F401
    import gage_eval.reporting.summary_generators.external_harness  # noqa: F401
    import gage_eval.reporting.summary_generators.gomoku  # noqa: F401
    import gage_eval.reporting.summary_generators.swebench  # noqa: F401
    import gage_eval.reporting.summary_generators.tau2  # noqa: F401
