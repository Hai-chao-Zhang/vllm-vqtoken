# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib.metadata import entry_points
from types import ModuleType

from vllm_vqtoken import MODEL_ARCHITECTURE, MODEL_ENTRY_POINT, register


def test_installed_distribution_exposes_general_plugin_entry_point():
    matches = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="vllm.general_plugins")
        if entry_point.name == "vqtoken_model"
    }

    assert matches == {"vqtoken_model": "vllm_vqtoken:register"}


def test_register_is_lazy_and_idempotent(monkeypatch):
    class FakeRegistry:
        models: dict[str, str] = {}

        @classmethod
        def get_supported_archs(cls):
            return set(cls.models)

        @classmethod
        def register_model(cls, architecture: str, entry_point: str):
            cls.models[architecture] = entry_point

    fake_vllm = ModuleType("vllm")
    fake_vllm.ModelRegistry = FakeRegistry
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    register()
    register()

    assert FakeRegistry.models == {MODEL_ARCHITECTURE: MODEL_ENTRY_POINT}
