from __future__ import annotations

import json
from pathlib import Path

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "gage_eval"
SWEBENCH_PRO_ANSIBLE_TASK = (
    "instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d"
    "-v0f01c69f1e2528b935359cfe578530722bca2c59"
)


def test_external_harness_kit_modules_are_provider_grouped() -> None:
    kit_root = SRC / "external_harness_kits"
    harbor_root = kit_root / "harbor"

    assert not (SRC / "external_harness").exists()
    assert (kit_root / "base.py").is_file()
    assert (kit_root / "errors.py").is_file()
    assert (kit_root / "secret_redaction.py").is_file()
    assert harbor_root.is_dir()
    for module in ("environment.py", "launcher.py", "observability.py", "results.py"):
        assert (harbor_root / module).is_file()


def test_harbor_dataset_loader_lives_with_dataset_loaders() -> None:
    manifest = json.loads((SRC / "registry" / "manifests" / "datasets.json").read_text(encoding="utf-8"))
    harbor_entries = {
        entry["name"]: entry
        for entry in manifest["entries"]
        if entry["name"] in {"harbor_registry", "harbor_local_path"}
    }

    assert not (SRC / "assets" / "datasets" / "harbor.py").exists()
    assert (SRC / "assets" / "datasets" / "loaders" / "harbor.py").is_file()
    assert set(harbor_entries) == {"harbor_registry", "harbor_local_path"}
    for entry in harbor_entries.values():
        assert entry["module"] == "gage_eval.assets.datasets.loaders.harbor"
        assert entry["declared_in"] == "src/gage_eval/assets/datasets/loaders/harbor.py"


def test_harbor_role_adapter_follows_flat_adapter_layout() -> None:
    old_adapter_namespace = ".".join(["gage_eval", "role", "adapters", "external"])

    assert (SRC / "role" / "adapters" / "harbor.py").is_file()
    assert not (SRC / "role" / "adapters" / "external").exists()

    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert old_adapter_namespace not in text, str(path)


def test_sample_artifact_writer_is_provider_neutral() -> None:
    writer_path = SRC / "pipeline" / "sample_artifact_writer.py"
    text = writer_path.read_text(encoding="utf-8").lower()

    assert "harbor" not in text


def test_custom_config_contains_harbor_terminal_bench2_entrypoint() -> None:
    config_path = ROOT / "config" / "custom" / "external_harness_kits" / "harbor_terminal_bench2_lmstudio_1case.yaml"
    assert config_path.is_file()
    text = config_path.read_text(encoding="utf-8")
    assert "HARBOR_TB2_TASK_PATH" in text
    assert "/Users/" not in text
    assert "tests/data/external_harness_kits/terminal_bench/gpt2-codegolf" in text
    assert "class_path: gage_eval.role.adapters.harbor:HarborAdapter" in text
    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))
    assert config.role_adapters[0].class_path == "gage_eval.role.adapters.harbor:HarborAdapter"


def test_custom_config_contains_harbor_swebench_pro_entrypoint() -> None:
    config_path = ROOT / "config" / "custom" / "external_harness_kits" / "harbor_swebench_pro_lmstudio_1case.yaml"
    assert config_path.is_file()
    text = config_path.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "tests/data/external_harness_kits/swebench_pro" not in text
    assert "HARBOR_SWEBENCH_PRO_TASK_PATH" not in text
    assert "loader: harbor_registry" in text
    assert "ref: swebenchpro@1.0" in text
    assert "tests/data/external_harness_kits/registries/harbor_swebenchpro_ansible_1case.json" in text
    assert SWEBENCH_PRO_ANSIBLE_TASK in text
    assert "name: swe-agent" in text
    assert "kind: installed_client" in text

    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))
    assert config.datasets[0].loader == "harbor_registry"
    assert config.datasets[0].params["ref"] == "swebenchpro@1.0"
    assert (
        config.datasets[0].params["registry_path"]
        == "tests/data/external_harness_kits/registries/harbor_swebenchpro_ansible_1case.json"
    )
    assert config.datasets[0].params["task_names"] == [SWEBENCH_PRO_ANSIBLE_TASK]
    assert config.datasets[0].params["n_tasks"] == 1
    assert config.role_adapters[0].class_path == "gage_eval.role.adapters.harbor:HarborAdapter"
    assert config.role_adapters[0].params["harness"]["agent"]["kind"] == "installed_client"
    assert config.role_adapters[0].params["harness"]["agent"]["name"] == "swe-agent"


def test_harbor_one_case_datasets_live_under_tests_data() -> None:
    from harbor.models.task.paths import TaskPaths

    tb2_task = ROOT / "tests" / "data" / "external_harness_kits" / "terminal_bench" / "gpt2-codegolf"
    swe_task = (
        ROOT
        / "tests"
        / "data"
        / "external_harness_kits"
        / "swebench_pro"
        / SWEBENCH_PRO_ANSIBLE_TASK
    )

    assert TaskPaths(tb2_task).is_valid()
    assert TaskPaths(swe_task).is_valid()


def test_harbor_one_case_datasets_are_not_collected_as_pytest_tests() -> None:
    pytest_ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")

    assert "tests/data" in pytest_ini
