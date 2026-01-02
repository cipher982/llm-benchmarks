# LLM Benchmarks Operations

Scripts and systemd units for monitoring the benchmark service.

## Daily Health Check

AI-powered analysis of benchmark errors, sent via email daily.

### What it does

1. Queries MongoDB for last 24h of errors and successes
2. Aggregates by error kind, provider, and model
3. Sends summary to OpenAI (gpt-4o-mini) for analysis
4. Emails you the assessment with:
   - Overall status (healthy/warning/critical)
   - Models likely deprecated
   - Code fixes needed (capability mismatches)
   - Provider-wide issues
   - Recommended actions

### Local testing

```bash
# From llm-benchmarks directory
cd /path/to/llm-benchmarks

# Dry run (prints email, doesn't send)
uv run python ops/daily-health-check.py --dry-run

# Look back further
uv run python ops/daily-health-check.py --days 7 --dry-run

# Actually send email (requires msmtp configured)
uv run python ops/daily-health-check.py
```

Required environment variables:
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DB` - Database name (default: llm-bench)
- `OPENAI_API_KEY` - OpenAI API key for analysis
- `NOTIFY_EMAIL` - Email recipient (default: david010@gmail.com)

### Deploy to clifford

```bash
# SSH to server
ssh clifford

# Copy systemd units
sudo cp /opt/llm-benchmarks/ops/llm-health-check.service /etc/systemd/system/
sudo cp /opt/llm-benchmarks/ops/llm-health-check.timer /etc/systemd/system/
sudo cp /opt/llm-benchmarks/ops/llm-health-check-failure@.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable llm-health-check.timer
sudo systemctl start llm-health-check.timer

# Verify timer is active
systemctl list-timers | grep llm-health

# Test the service manually
sudo systemctl start llm-health-check.service
journalctl -u llm-health-check.service -f
```

### Check status

```bash
# Timer status
systemctl status llm-health-check.timer

# Last run
journalctl -u llm-health-check.service -n 50

# Next scheduled run
systemctl list-timers llm-health-check.timer
```

### Environment on clifford

The service reads from `/opt/llm-benchmarks/.env`. Make sure these are set:

```bash
# Check existing env
cat /opt/llm-benchmarks/.env

# Add OpenAI key if missing (get from ~/git/me/mytech for the key location)
echo 'OPENAI_API_KEY=sk-...' | sudo tee -a /opt/llm-benchmarks/.env
```

## Model Discovery

Daily fetching of OpenRouter catalog to discover new models.

### What it does

1. Fetches OpenRouter's `/api/v1/models` API (free, no auth)
2. Stores catalog in MongoDB `openrouter_catalog` collection
3. Tracks `first_seen_at` and `last_seen_at` timestamps
4. Use `discovery.cli report` to see new models to add

### Local testing

```bash
# Fetch OpenRouter catalog
uv run python -m api.llm_bench.discovery.cli fetch

# View discovery report (copy-paste commands to add models)
uv run python -m api.llm_bench.discovery.cli report --max-matches 20

# Stats about catalog
uv run python -m api.llm_bench.discovery.cli stats
```

### Deploy to clifford

```bash
# SSH to server
ssh clifford

# Copy systemd units
sudo cp /opt/llm-benchmarks/ops/llm-discovery.service /etc/systemd/system/
sudo cp /opt/llm-benchmarks/ops/llm-discovery.timer /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable llm-discovery.timer
sudo systemctl start llm-discovery.timer

# Verify timer is active
systemctl list-timers | grep llm-discovery

# Test the service manually
sudo systemctl start llm-discovery.service
journalctl -u llm-discovery.service -f
```

## Files

| File | Purpose |
|------|---------|
| `daily-health-check.py` | Main script - collects data, calls OpenAI, sends email |
| `llm-health-check.service` | systemd service unit |
| `llm-health-check.timer` | systemd timer (daily at 08:00 UTC) |
| `llm-health-check-failure@.service` | Crash notification service |
| `llm-discovery.service` | systemd service for model discovery |
| `llm-discovery.timer` | systemd timer (daily at 07:00 UTC, 1hr before health check) |
