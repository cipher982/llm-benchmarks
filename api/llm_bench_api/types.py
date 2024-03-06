from typing import Optional

from pydantic import BaseModel


class BenchmarkRequest(BaseModel):
    provider: str
    model: str
    query: str
    max_tokens: int = 256
    temperature: float = 0.1
    run_always: bool = False
    debug: bool = False


class BenchmarkResponse(BaseModel):
    status: str
    metrics: dict
    reason: Optional[str] = None
