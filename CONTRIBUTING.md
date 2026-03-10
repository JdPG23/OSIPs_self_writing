# Contributing

## Ways to Contribute

### 1. Expand the Corpus (highest impact)
The system is only as good as its reference data.

**Add reference OSIPs** to `corpus/references/`:
```json
{
  "title": "ACRONYM — Full Title",
  "institution": "University/Company",
  "country": "ES",
  "year": 2024,
  "domain": "debris-removal",
  "trl_entry": 2,
  "trl_target": 4,
  "budget_eur": 75000,
  "status": "approved",
  "content": "Full proposal text or detailed summary..."
}
```

**Update ESA priorities** in `corpus/esa_priorities.md` when new calls are published.

**Add rejection patterns** to `corpus/rejection_patterns.md` from real feedback.

### 2. Improve the Scoring Rubric
Edit `scorer.py` rubrics to be more calibrated. If you have access to real
ESA evaluator feedback, use it to refine the scoring criteria.

### 3. Experiment with pipeline.py
The core activity. Create a branch, iterate, log results:
```bash
git checkout -b osip-research/your-tag
# modify pipeline.py
python run.py --topic "Your topic" > run.log 2>&1
# check results, keep/discard, repeat
```

### 4. Add New Scoring Dimensions
The current 4 dimensions may not capture everything. Proposals welcome for:
- Feasibility scoring (budget/timeline realism)
- Team-fit scoring (implicit credibility signals)
- Readability metrics (automated, no LLM needed)

## Guidelines

- Keep `prepare.py` and `scorer.py` stable — changes there invalidate all prior results
- One logical change per commit
- Always log experiments to `results.tsv`
- Don't commit API keys or generated proposals with real proprietary content
