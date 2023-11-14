import logging
from typing import Optional


logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(
        self,
        framework: str,
        model_name: str,
        run_ts: str,
        model_dtype: str,
        temperature: float,
        quantization_bits: Optional[str] = None,
    ):
        self.framework = framework
        self.model_name = model_name
        self.run_ts = run_ts
        self.model_dtype = model_dtype
        self.temperature = temperature
        self.quantization_bits = quantization_bits

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, value):
        if value not in ["transformers", "gguf", "hf-tgi"]:
            raise ValueError("framework must be either 'transformers' or 'gguf'")
        self._framework = value

    @property
    def load_in_4bit(self) -> bool:
        return self.quantization_bits == "4bit" if self.quantization_bits is not None else False

    @property
    def load_in_8bit(self) -> bool:
        return self.quantization_bits == "8bit" if self.quantization_bits is not None else False

    def to_dict(self):
        return {
            "framework": self.framework,
            "model_name": self.model_name,
            "run_ts": self.run_ts,
            "model_dtype": self.model_dtype,
            "temperature": self.temperature,
            "quantization_bits": self.quantization_bits,
        }
