from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict

from core.support.file import load_file


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class OpenAIReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: ReasoningEffort


class OpenAINodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    use_responses_api: bool = True
    prompt_cache_key: str | None = None
    temperature: float | None = None
    reasoning: OpenAIReasoningConfig | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


class OpenAIConfigCollection(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config item not found: {name}") from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.keys()))





def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return [to_namespace(item) for item in value]

    return value


def _parse_model_session(session_name: str, data: Any) -> OpenAIConfigCollection:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session config: {session_name}")

    configs = OpenAIConfigCollection()
    for node_name, node_data in data.items():
        if not isinstance(node_data, dict) or "model_name" not in node_data:
            raise ValueError(f"Invalid node config: {session_name}.{node_name}")
        configs[node_name] = OpenAINodeConfig.model_validate(node_data)
    return configs


def _apply_test_model_names(config: Any) -> None:
    if isinstance(config, OpenAINodeConfig):
        if not config.model_name.endswith("-mini"):
            config.model_name = f"{config.model_name}-mini"
        return

    if isinstance(config, dict):
        for item in config.values():
            _apply_test_model_names(item)


@lru_cache(maxsize=None)
def load_model_config(
    config_path: str | Path = "configs/models.yaml",
) -> OpenAIConfigCollection:
    session_config = load_file(config_path) or {}

    configs = OpenAIConfigCollection()
    for session_name, session_data in session_config.items():
        configs[session_name] = _parse_model_session(session_name, session_data)

    return configs


def load_general_config(
    config_path: str | Path = "configs/general.yaml",
):

    config = load_file(config_path) or {}
    return to_namespace(config)


def load_config(
    config_dir: str | Path = "configs", is_test: bool = False
) -> SimpleNamespace:
    config_dir = Path(config_dir)

    general_config_path: str | Path = config_dir / "general.yaml"
    model_config_path: str | Path = config_dir / "models.yaml"

    general_config = (
        load_general_config(general_config_path)
        if general_config_path is not None
        else SimpleNamespace()
    )

    model_config = (
        load_model_config(model_config_path)
        if model_config_path is not None
        else OpenAIConfigCollection()
    )
    model_config = deepcopy(model_config)

    if is_test:
        _apply_test_model_names(model_config)

    return SimpleNamespace(
        **vars(general_config),
        models=model_config,
    )
