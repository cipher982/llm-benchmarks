# LLM Benchmarks Operations

Production discovery and health checks run through Sauron jobs on `clifford`:

- `llm-bench-discovery`
- `llm-bench-provider-discovery`
- `llm-bench-health`

This repo no longer carries systemd timer units for those jobs. Use Sauron for
scheduled operations and this directory only for manual helper scripts.

## Manual Error Classification

`classify-errors.sh` classifies unclassified error rollups in MongoDB:

```bash
./ops/classify-errors.sh
./ops/classify-errors.sh --max 500
./ops/classify-errors.sh --all
./ops/classify-errors.sh --use-openai
```

Required environment variables:

- `MONGODB_URI`
- `MONGODB_DB`
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

For scheduled health email behavior, edit the Sauron job implementation rather
than adding timers back to this repo.
