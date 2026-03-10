# Experiment Log & Insights

Running notes on what works and what doesn't. Updated after each research run.

## Hypothesis Backlog

Ideas to test, ordered by expected impact. The meta-agent picks from here
when it runs out of its own ideas.

### High Priority
- [ ] Add few-shot examples of approved OSIPs to Writer Agent prompt
- [ ] Inject specific ESA mission names into Context Agent output
- [ ] Force Writer to include exact TRL numbers with justification
- [ ] Add "ESA Alignment Checker" as a separate agent between Writer and Critic
- [ ] Test structured output (XML sections) vs free-form markdown

### Medium Priority
- [ ] Vary number of ideation angles (1 vs 3 vs 5)
- [ ] Test 0 vs 1 vs 2 vs 3 revision rounds
- [ ] Add budget breakdown template to Writer prompt
- [ ] Experiment with chain-of-thought before writing vs direct generation
- [ ] Test Haiku for Critic (cheaper) vs Sonnet (better critique)

### Low Priority / Exploratory
- [ ] Inject rejected OSIP patterns as negative examples
- [ ] Add a "Simplifier" agent that removes filler after revision
- [ ] Test parallel ideation (3 Writers, pick best) vs sequential
- [ ] Domain-specific glossary injection

## Completed Experiments

*Updated automatically or manually after runs.*

<!-- Example:
### Run: mar10
- Branch: osip-research/mar10
- Topic: "Autonomous GNC for debris removal"
- Baseline: 62.0
- Best: 78.5 (+16.5 over baseline)
- Key finding: Adding ESA mission names to context improved alignment by +7 pts
- Key finding: 2 revision rounds optimal; 3rd round adds tokens but <1pt gain
-->

## Patterns Discovered

*Stable patterns confirmed across multiple runs.*

<!-- Example:
- TRL explicit mention: +5-8 pts on structure score consistently
- Few-shot examples: +3-5 pts on quality score
- Budget breakdown: +2-4 pts on structure score
-->
