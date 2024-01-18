def get_quant_type(file: str) -> str:
    """Get quantization type from file name."""
    if "f16" in file:
        return "f16"
    elif "int8" in file:
        return "8bit"
    elif "int4" in file:
        return "4bit"
    else:
        raise ValueError(f"Unknown quant type for file: {file}")
