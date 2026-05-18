from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger
from pandas import read_table
from pydantic import BaseModel

from .chains import build_chain
from .langfuse import load_prompt
from core.support.config import OpenAINodeConfig

load_dotenv(find_dotenv(usecwd=True))


class PromptGenerator:
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        prompt_key: str,
        local_prompt_dir: str | Path | None = None,
        output_parser: type[BaseModel] | None = None,
        node_name: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.prompt_key = prompt_key
        self.local_prompt_dir: str | Path | None = local_prompt_dir
        self.node_name = node_name or self.prompt_key
        self.output_parser = output_parser

        self.initialize()

    def initialize(self):
        self.prompt, self.parser, self.chain = build_chain(
            model_config=self.model_config,
            prompt_key=self.prompt_key,
            local_prompt_dir=self.local_prompt_dir,
            output_parser=self.output_parser,
        )

    # OVERRIDE POSSIBLELY
    def override_input(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any] | list[SystemMessage | HumanMessage]:
        return input_data

    # OVERRIDE POSSIBLELY
    def override_inputs(
        self, input_data: dict[str, Any]
    ) -> list[dict[str, Any]] | list[list[SystemMessage | HumanMessage]]:
        return [input_data]

    def _build_prompt_input(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any] | list[SystemMessage | HumanMessage]:
        prompt_input = self.override_input(input_data)
        if isinstance(prompt_input, list):
            return prompt_input

        if not isinstance(prompt_input, dict):
            raise TypeError("override_input() must return a dict or list.")

        if isinstance(self.parser, PydanticOutputParser):
            prompt_input = {
                **prompt_input,
                "format_instructions": self.parser.get_format_instructions(),
            }
        return prompt_input

    def _build_prompt_inputs(
        self, input_data: dict[str, Any]
    ) -> list[dict[str, Any]] | list[list[SystemMessage | HumanMessage]]:
        batch_input_data = self.override_inputs(input_data)
        if not isinstance(batch_input_data, list) or not all(
            isinstance(item, (dict, list)) for item in batch_input_data
        ):
            raise TypeError("override_inputs() must return a list of dicts or lists.")

        return [self._build_prompt_input(item) for item in batch_input_data]

    def _serialize_output(self, output: Any) -> Any:
        if isinstance(self.parser, PydanticOutputParser) and hasattr(
            output, "model_dump"
        ):
            return output.model_dump()
        return output

    def invoke(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> Any:
        prompt_input = self._build_prompt_input(input_data)
        logger.info("Invoking Generator.")
        output = self.chain.invoke(prompt_input, config=config)
        # if output_parser is a PydanticOutputParser, it will use dump
        return self._serialize_output(output)

    def batch(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> list[Any]:
        batch_inputs = self._build_prompt_inputs(input_data)
        logger.info(
            "Batch invoking Generator with {} prompt inputs.",
            len(batch_inputs),
        )
        outputs = self.chain.batch(batch_inputs, config=config)
        # print outputs

        return [self._serialize_output(output) for output in outputs]


class ChatGenerator(PromptGenerator):
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        system_prompt_key: str,
        human_prompt_key: str,
        local_prompt_dir: str | Path | None = None,
        output_parser: type[BaseModel] | None = None,
        node_name: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.system_prompt_key = system_prompt_key
        self.human_prompt_key = human_prompt_key
        self.local_prompt_dir: str | Path | None = local_prompt_dir
        self.node_name = node_name or f"{system_prompt_key}_{human_prompt_key}"
        self.output_parser = output_parser

        self.initialize()

    def initialize(self):
        """
        prompt_key는 system_prompt와 human_prompt로 구성되어 있음.
        self.prompt_key = "organizer_for_3rd_grade"인 경우, prompt_key는 system_prompt와 human_prompt로 구성되어 있음.
        """
        self.prompt = load_prompt(
            f"{self.system_prompt_key}",
            prompt_dir=self.local_prompt_dir,
        )

        self.human_prompt = load_prompt(
            f"{self.human_prompt_key}",
            prompt_dir=self.local_prompt_dir,
        )

        _, self.parser, self.chain = build_chain(
            model_config=self.model_config,
            prompt_key=f"{self.system_prompt_key}",
            local_prompt_dir=self.local_prompt_dir,
            output_parser=self.output_parser,
            is_chat=True,
        )

    def _format_template_variables(self, template, variables):
        try:
            return template.format(**variables)
        except KeyError as error:
            missing_key = error.args[0]
            raise ValueError

    def _append_format_instructions(self, template):
        if isinstance(self.parser, PydanticOutputParser):
            return (
                template + "\n# Output Format\n" + self.parser.get_format_instructions()
            )
        return template

    def _initialize_page_separator_template(self):
        if not hasattr(self, "page_separator_template"):
            self.page_separator_template: str = (
                "\n---\n다음은 문서 페이지 {page} 입니다. "
                "총 {total_pages} 페이지 중 {page} 페이지입니다.\n---\n"
            )

    def _build_image_message_contents(self, images):
        total_pages = len(images)
        image_content = []
        for page, image in enumerate(images, start=1):
            image_content.append(
                {
                    "type": "text",
                    "text": self.page_separator_template.format(
                        page=page, total_pages=total_pages
                    ),
                }
            )
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                        "detail": "auto",
                    },
                }
            )
        return image_content

    def override_input(
        self, input_data: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        input_data = dict(input_data)

        # Support images from any of the common keys ('images', 'images64')
        images = None
        for key in ["images", "images64"]:
            if key in input_data and isinstance(input_data[key], list):
                images = input_data.pop(key)
                break

        system_content = self._append_format_instructions(self.prompt)
        system_message = SystemMessage(content=system_content)

        human_prompt = self.human_prompt
        try:
            human_prompt = self._format_template_variables(human_prompt, input_data)
        except ValueError:
            pass

        human_content = [{"type": "text", "text": human_prompt}]
        if images is not None:
            logger.info("Processing {} images in ChatGenerator input.", len(images))
            self._initialize_page_separator_template()
            image_contents = self._build_image_message_contents(images)
            human_content.extend(image_contents)

        human_message = HumanMessage(content=human_content)
        return [system_message, human_message]
