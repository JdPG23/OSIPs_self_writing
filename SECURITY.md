# Security

## API Keys
- Never commit `.env` files — they contain API keys
- The `.gitignore` excludes `.env` by default
- Use environment variables, not hardcoded keys

## Generated Content
- Proposals in `outputs/` may contain sensitive technical ideas
- Review before sharing or publishing
- Do not commit proposals with proprietary institutional information

## LLM-as-Judge
- The scorer model is separate from the generator to avoid self-reinforcement
- Scoring prompts are fixed in `scorer.py` to prevent gaming
- If you suspect the pipeline is "overfitting" to the scorer, check for
  prompt injection patterns in generated proposals

## Reporting
If you find a security issue, open a private issue or contact the maintainer.
