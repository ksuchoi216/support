"""Shared node helpers for LangGraph sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from loguru import logger

from .chains import build_chain
from core.support.config import OpenAINodeConfig

load_dotenv(find_dotenv(usecwd=True))


def _normalize_output_value(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalize_output_value(value.model_dump())
    if isinstance(value, Mapping):
        return {
            _normalize_output_value(key): _normalize_output_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, str) and type(value) is not str:
        return str(value)
    return value


class GeneralNode:
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        override_input: Callable[[dict], dict[str, Any] | None],
        state_save_key: str,
        prompt_key: str,
        local_prompt: bool = False,
        local_prompt_dir: str | Path | None = None,
        update_state_for_input: Callable[[dict], dict[str, Any]] | None = None,
        update_state_from_output: Callable[[Any, dict], dict[str, Any]] | None = None,
        node_name: str | None = None,
        output_parser=None,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        state_type: Literal["dict", "list", "direct"] = "direct",
        state_dict_key: str | None = None,
        iter_key: str | None = None,
        return_parallel: bool = False,
    ) -> None:
        self.model_config = model_config
        self.prompt_key = prompt_key
        self.local_prompt = local_prompt
        self.local_prompt_dir = local_prompt_dir
        self.override_input = override_input
        self.update_state_for_input = update_state_for_input
        self.update_state_from_output = update_state_from_output
        self.output_parser = output_parser
        self.tools = tools
        self.tool_choice = tool_choice
        self.state_type = state_type
        self.state_dict_key = state_dict_key
        self.state_save_key = state_save_key
        self.node_name = node_name or self.prompt_key or "general_node"
        self.return_parallel = return_parallel
        self.iter_key = iter_key

    def _preprocess(self) -> None:
        self.prompt, self.parser, self.chain = build_chain(
            model_config=self.model_config,
            prompt_key=self.prompt_key,
            local_prompt=self.local_prompt,
            local_prompt_dir=self.local_prompt_dir,
            output_parser=self.output_parser,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )

    def _update_state_for_input(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.update_state_for_input is None:
            return state

        state_update = self.update_state_for_input(state)
        if not isinstance(state_update, Mapping):
            raise TypeError("update_state_for_input must return a mapping.")
        state.update(state_update)
        return state

    def _update_state_from_output(
        self,
        *,
        state: dict[str, Any],
        output: Any,
    ) -> dict[str, Any]:
        output = _normalize_output_value(output)

        if self.update_state_from_output is not None:
            state_update = self.update_state_from_output(output, state)
            if not isinstance(state_update, Mapping):
                raise TypeError("update_state_from_output must return a mapping.")
            logger.info("AI Answer:\n{}\n", output)
            state.update(state_update)
            return state

        logger.info("AI Answer:\n{}\n", output)

        if self.return_parallel:
            return {self.state_save_key: output}

        if self.state_type == "list":
            state.setdefault(self.state_save_key, []).append(output)
        elif self.state_type == "dict":
            state.setdefault(self.state_save_key, {})[self.state_dict_key] = output
        else:
            state[self.state_save_key] = output
        return state

    def __call__(self):
        self._preprocess()

        def node(
            state: dict[str, Any],
            config: RunnableConfig | None = None,
        ) -> dict[str, Any]:
            state = self._update_state_for_input(state)
            logger.info("============= {} ==============", self.node_name)

            inputs = self.override_input(state) or {}
            if not isinstance(inputs, Mapping):
                raise TypeError("override_input must return a mapping or None.")

            resolved_inputs = dict(inputs)
            if isinstance(self.parser, PydanticOutputParser):
                resolved_inputs["format_instructions"] = (
                    self.parser.get_format_instructions()
                )

            output = self.chain.invoke(resolved_inputs, config=config)
            if self.iter_key:
                state[self.iter_key] += 1
                logger.info("Iter: {} from {}", state[self.iter_key], self.iter_key)

            return self._update_state_from_output(state=state, output=output)

        return node
