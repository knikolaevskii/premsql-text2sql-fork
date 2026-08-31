import re
import time
from typing import Any, Dict, Optional

import requests

from premsql.generators.base import Text2SQLGeneratorBase
from premsql.logger import setup_console_logger

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")

logger = setup_console_logger(name="[WIKISQL-API-GENERATOR]")


class WikiSQLText2SQLGeneratorAPI(Text2SQLGeneratorBase):
    """
    Same OpenAI-compatible local-API generator as Text2SQLGeneratorAPI, plus a
    WikiSQL-specific postprocessing step: WikiSQL's gold queries lowercase
    string literals inside quotes, so generated queries are normalized the
    same way before comparison to avoid spurious mismatches.
    """

    def lowercase_quoted_strings(self, sql_string: str) -> str:
        def replace_quoted(match):
            return f"'{match.group(1).lower()}'"

        pattern = r"'([^'\\]*(\\.[^'\\]*)*)'"
        return re.sub(pattern, replace_quoted, sql_string)

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        type: str,
        api_base_url: str,
        experiment_folder: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_base_url = api_base_url

        super().__init__(
            experiment_folder=experiment_folder,
            experiment_name=experiment_name,
            type=type,
        )

    @property
    def load_client(self):
        return OpenAI(base_url=self.api_base_url, api_key="dummy-key")

    @property
    def load_tokenizer(self):
        return None

    @property
    def model_name_or_path(self):
        return self.model_name

    def generate(
        self,
        data_blob: dict,
        temperature: Optional[float] = 0.0,
        max_new_tokens: Optional[int] = 256,
        postprocess: Optional[bool] = True,
        retries: int = 3,
        retry_delay: float = 2.0,
        use_extended_api: bool = False,
        **kwargs,
    ) -> str:
        prompt = data_blob["prompt"]
        generation_config = {
            **kwargs,
            **{"temperature": temperature, "max_tokens": max_new_tokens},
        }

        attempt = 0
        while attempt < retries:
            try:
                if use_extended_api:
                    completion = self._call_direct_api(prompt=prompt, generation_config=generation_config)
                else:
                    completion = self._call_openai_client(prompt=prompt, generation_config=generation_config)

                if postprocess:
                    completion = self.postprocess(output_string=completion)

                return self.lowercase_quoted_strings(completion)

            except Exception as e:
                attempt += 1
                logger.error(f"[Attempt {attempt}] Error generating SQL with model {self.model_name}: {e}")

                if attempt >= retries:
                    raise
                time.sleep(retry_delay)

    def _call_openai_client(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        standard_params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": generation_config.get("temperature", 0.0),
            "max_tokens": generation_config.get("max_tokens", 256),
            "top_p": generation_config.get("top_p"),
            "stop": generation_config.get("stop"),
            "stream": generation_config.get("stream", False),
        }
        standard_params = {k: v for k, v in standard_params.items() if v is not None}
        return self.client.completions.create(**standard_params).choices[0].text

    def _call_direct_api(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        request_params = {
            "model": self.model_name,
            "prompt": prompt,
            **generation_config,
        }
        request_params = {k: v for k, v in request_params.items() if v is not None}

        response = requests.post(
            f"{self.api_base_url}/completions",
            json=request_params,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        return response.json()["choices"][0]["text"]
