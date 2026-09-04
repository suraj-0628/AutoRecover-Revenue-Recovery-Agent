"""Working revenue at risk in batches, rather than one case at a time.

The pieces, and what each is allowed to do:

  tiers.py     amount bands, parsed from the merchant's dunning policy
  plan.py      a BatchPlan and its budget — what a whole band will be offered
  executor.py  applies a plan to one case, deterministically, no LLM
  run.py       the BatchRun: what was worked, what it cost, what came back

The split exists because the decision unit and the execution unit are not the
same. A batch shares a cause, so the *decision* can be made once for the whole
band; each case still needs its own side effects and its own eligibility check,
so *execution* is per case. Running a full agent session per case instead makes
a 200-case batch take two hours of rate-limit waiting, and buys nothing — the
reasoning would be identical every time.
"""
