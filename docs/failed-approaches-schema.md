# Record genuine unsuccessful approaches after the real experiments

Create `docs/failed-approaches.json` only from measured approved-dataset runs. The freeze command requires four distinct actual completed run IDs and at least two entries marked substantive. Each diagnosis should reference curves, configuration and validation artefacts, and explain a plausible mechanism and any uncertainty.

```json
{
  "attempts": [
    {
      "run_id": "REPLACE_WITH_AN_ACTUAL_COMPLETED_RUN_ID",
      "hypothesis": "The specific improvement this run tested",
      "evidence": "Observed comparison, curve or error evidence and its artifact path",
      "diagnosis": "What failed, why the evidence supports that interpretation, and remaining uncertainty",
      "substantive": true
    }
  ]
}
```

The example is a schema, not an experiment record. Do not copy it as evidence. Four actual attempts are required overall; at least two must be substantial failed approaches, not four syntax mistakes or four deliberately sabotaged configurations. Ties or improvements in a secondary measure can require nuanced interpretation rather than automatically declaring failure.
