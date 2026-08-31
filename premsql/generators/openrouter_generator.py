import os
import time
from typing import Optional

from premsql.generators.base import Text2SQLGeneratorBase
from premsql.logger import setup_console_logger

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Module openai is not installed")

logger = setup_console_logger(name="[OPENROUTER-GENERATOR]")


class Text2SQLGeneratorOpenRouter(Text2SQLGeneratorBase):
    """
    OpenRouter API generator for Text2SQL benchmarking. Uses OpenRouter's
    OpenAI-compatible API to reach multiple hosted model providers through a
    single key.
    """

    MODEL_MAPPING = {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "claude-3-sonnet": "anthropic/claude-3.5-sonnet",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "deepseek-v3": "deepseek/deepseek-chat",
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    }

    SYSTEM_PROMPTS = {
        "sqlite": (
            "You are an expert SQLite developer. Your role is to convert user questions into "
            "accurate, efficient SQL queries based on the provided database schema. Always return "
            "only the SQL query without any explanations or formatting."
        ),
        "postgresql": (
            "You are an expert PostgreSQL developer. Your role is to convert user questions into "
            "accurate, efficient SQL queries based on the provided database schema. Always return "
            "only the SQL query without any explanations or formatting."
        ),
        "wikisql": (
            "You are an expert SQLite developer. Your role is to convert user questions into "
            "accurate, efficient SQL queries based on the provided database schema. Always return "
            "only the SQL query without any explanations or formatting and use lowercase in WHERE "
            "clauses and finish the query with ; . col0, col1, col2 are the actual column names in "
            "the database, so use them in the query"
        ),
    }

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        type: str,
        experiment_folder: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        data_base_type: Optional[str] = "sqlite",
    ):
        if data_base_type not in self.SYSTEM_PROMPTS:
            raise ValueError(f"Invalid database type: {data_base_type}")

        self.data_base_type = data_base_type
        self._api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as parameter or OPENROUTER_API_KEY environment variable"
            )

        self.original_model_name = model_name
        self.model_name = self.MODEL_MAPPING.get(model_name, model_name)

        super().__init__(
            experiment_folder=experiment_folder,
            experiment_name=experiment_name,
            type=type,
        )

    @property
    def load_client(self):
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self._api_key)

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
        **kwargs,
    ) -> str:
        prompt = data_blob["prompt"]
        generation_config = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            **kwargs,
        }
        system_prompt = self.SYSTEM_PROMPTS[self.data_base_type]

        attempt = 0
        while attempt < retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **generation_config,
                )
                generated_text = completion.choices[0].message.content
                return self.postprocess(output_string=generated_text) if postprocess else generated_text

            except Exception as e:
                attempt += 1
                logger.error(f"[Attempt {attempt}] Error generating SQL with model {self.model_name}: {e}")

                if attempt >= retries:
                    raise
                time.sleep(retry_delay)

    def get_available_models(self):
        return list(self.MODEL_MAPPING.keys())

    def add_model_mapping(self, short_name: str, openrouter_model_id: str):
        self.MODEL_MAPPING[short_name] = openrouter_model_id
