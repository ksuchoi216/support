from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

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


class OpenAIConfigCollection(dict[str, OpenAINodeConfig]):
    def __getattr__(self, name: str) -> OpenAINodeConfig:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config item not found: {name}") from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.keys()))


def _load_config(
    config_path: str | Path,
) -> dict[str, dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config root in {path}: expected dict")
    return data


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return [to_namespace(item) for item in value]

    return value


@lru_cache(maxsize=None)
def load_model_config(
    config_path: str | Path = "configs/models.yaml",
) -> OpenAIConfigCollection:
    session_config = _load_config(config_path)

    configs = OpenAIConfigCollection()
    for node_name, node_data in session_config.items():
        if not isinstance(node_data, dict):
            raise ValueError(f"Invalid node config: {node_name}")
        configs[node_name] = OpenAINodeConfig.model_validate(node_data)

    return to_namespace(configs)


def load_general_config(
    config_path: str | Path = "configs/general.yaml",
):

    config = _load_config(config_path)
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

    if is_test:
        for node_config in vars(model_config).values():
            if not node_config.model_name.endswith("-mini"):
                node_config.model_name = f"{node_config.model_name}-mini"

    return SimpleNamespace(
        **vars(general_config),
        models=model_config,
    )
