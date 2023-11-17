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
        quantization_method: Optional[str] = None,
        quantization_bits: Optional[str] = None,
        misc: dict = {},
    ):
        self.framework = framework
        self.model_name = model_name
        self.run_ts = run_ts
        self.model_dtype = model_dtype
        self.temperature = temperature
        self.quantization_method = quantization_method
        self.quantization_bits = quantization_bits
        self.misc = misc

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, value):
        if value not in ["transformers", "gguf", "hf-tgi"]:
            raise ValueError("framework must be either 'transformers' or 'gguf'")
        self._framework = value

    @property
    def quantization_method(self):
        return self._quantization_method

    @quantization_method.setter
    def quantization_method(self, value):
        if value not in ["bitsandbytes", "gptq", None]:
            raise ValueError(f"quant method must be 'bitsandbytes', 'gptq', or None. Got {value}")
        self._quantization_method = value

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
            "quantization_method": self.quantization_method,
            "quantization_bits": self.quantization_bits,
            "misc": self.misc,
        }


class MongoConfig:
    def __init__(self, uri: str, db: str, collection: str):
        """
        Initialize the MongoDB configuration.
        """
        self.uri = uri
        self.db = db
        self.collection = collection
