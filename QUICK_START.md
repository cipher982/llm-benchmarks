# Quick Start: Add Claude Sonnet 4.5

## 🚀 TL;DR

```bash
# 1. Get the exact model ID from Anthropic docs
#    https://docs.anthropic.com/en/docs/models-overview

# 2. Set environment variables (if not already set)
export MONGODB_URI="your-mongodb-uri"
export MONGODB_DB="llm-bench"
export ANTHROPIC_API_KEY="your-api-key"

# 3. Add the model (replace MODEL_ID with actual ID)
python scripts/add_model_to_db.py \
  --provider anthropic \
  --model-id claude-4-sonnet-20250929

# 4. Test it
python api/bench_headless.py --providers anthropic --limit 1
```

## ✅ That's it!

---

## 📋 Checklist

- [ ] Get exact model ID from Anthropic
- [ ] Set `MONGODB_URI` environment variable
- [ ] Set `ANTHROPIC_API_KEY` environment variable
- [ ] Run the add_model_to_db.py script
- [ ] Test with a benchmark run
- [ ] Check for errors

---

## 📚 Full Documentation

For detailed information, see:
- **RESEARCH_SUMMARY.md** - Complete research and all methods
- **ADD_SONNET_4.5_INSTRUCTIONS.md** - Step-by-step guide
- **scripts/add_model_to_db.py** - Python script with `--help`

---

## 🔍 Verify Model ID

The model ID likely follows this pattern:
```
claude-4-sonnet-YYYYMMDD
```

Where YYYYMMDD is the release date (e.g., 20250929 for Sep 29, 2025).

**Check Anthropic's docs to confirm!**

---

## ❓ Common Questions

**Q: Do I need local server access?**  
A: Only if your MongoDB is running locally. If it's remote, you can run from anywhere.

**Q: Will this work?**  
A: Yes! The system is designed for this. Just need the correct model ID.

**Q: What if the model ID is wrong?**  
A: The benchmark will fail with an API error. Update the ID in the database.

**Q: Can I test without adding to database?**  
A: No, the system reads models from MongoDB only.

---

## 🆘 Need Help?

1. Check `/workspace/RESEARCH_SUMMARY.md` for complete details
2. Run with `--help`: `python scripts/add_model_to_db.py --help`
3. List existing models: `python scripts/add_model_to_db.py --list --provider anthropic`