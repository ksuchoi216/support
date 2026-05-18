from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from core.support.config import OpenAINodeConfig
from .langfuse import load_prompt
from .models import REASONING_MODELS, ModelNames

ParserType = PydanticOutputParser | StrOutputParser


def validate_model_name(model_name: str) -> ModelNames:
    try:
        return ModelNames(model_name)
    except ValueError as exc:
        supported_models = ", ".join(model.value for model in ModelNames)
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Supported models: {supported_models}"
        ) from exc


def build_llm(model_config: OpenAINodeConfig) -> ChatOpenAI:
    model_name = validate_model_name(model_config.model_name)

    llm_kwargs: dict[str, Any] = {
        "model": model_config.model_name,
        "use_responses_api": model_config.use_responses_api,
    }
    if model_name in REASONING_MODELS:
        if model_config.temperature is not None:
            raise ValueError("temperature is only supported for non-reasoning models.")
        if model_config.reasoning is not None:
            llm_kwargs["reasoning"] = model_config.reasoning.model_dump()
        if model_config.verbosity is not None:
            if model_config.use_responses_api:
                llm_kwargs["model_kwargs"] = {
                    "text": {"verbosity": model_config.verbosity}
                }
            else:
                llm_kwargs["verbosity"] = model_config.verbosity
    else:
        if model_config.reasoning is not None or model_config.verbosity is not None:
            raise ValueError(
                "reasoning and verbosity are only supported for reasoning models."
            )
        if model_config.temperature is not None:
            llm_kwargs["temperature"] = model_config.temperature

    if model_config.prompt_cache_key:
        llm_kwargs["extra_body"] = {"prompt_cache_key": model_config.prompt_cache_key}
        logger.info("Using prompt_cache_key: %s", model_config.prompt_cache_key)
    return ChatOpenAI(**llm_kwargs)


def build_bound_llm(
    model_config: OpenAINodeConfig,
    *,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
):
    llm = build_llm(model_config)
    if not tools:
        return llm

    bind_kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        bind_kwargs["tool_choice"] = tool_choice
    return llm.bind_tools(list(tools), **bind_kwargs)


def build_output_parser(output_parser) -> ParserType:
    if isinstance(output_parser, type) and issubclass(output_parser, BaseModel):
        logger.info(
            "Using PydanticOutputParser with model: {}",
            output_parser.__name__,
        )
        return PydanticOutputParser(pydantic_object=output_parser)
    logger.info("Using StrOutputParser for output parsing.")
    return StrOutputParser()


def build_chain(
    *,
    model_config: OpenAINodeConfig,
    prompt_key: str,
    local_prompt: bool = False,
    local_prompt_dir: str | Path | None = None,
    output_parser=None,
    is_chat: bool = False,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
):
    prompt = load_prompt(
        prompt_key,
        local_prompt=local_prompt,
        prompt_dir=local_prompt_dir,
    )
    llm = build_bound_llm(
        model_config,
        tools=tools,
        tool_choice=tool_choice,
    )
    parser = build_output_parser(output_parser)

    if is_chat:
        chain = llm | parser
    else:
        prompter = PromptTemplate.from_template(prompt)
        chain = prompter | llm | parser

    return prompt, parser, chain


# def build_chain_chat(self, prompt, llm, parser, is_chat: bool = False):
