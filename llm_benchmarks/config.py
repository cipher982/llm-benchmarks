from typing import Optional


class ModelConfig:
    def __init__(self, run_ts: str, torch_dtype: str, temperature: float, quantization_bits: Optional[str] = None):
        self.run_ts = run_ts
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.quantization_bits = quantization_bits

    @property
    def load_in_4bit(self) -> bool:
        return self.quantization_bits == "4bit" if self.quantization_bits is not None else False

    @property
    def load_in_8bit(self) -> bool:
        return self.quantization_bits == "8bit" if self.quantization_bits is not None else False
