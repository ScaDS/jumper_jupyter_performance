# Rubric: resource effect predictions

You are **not** grading anything here. Read each suggestion's description in
`output/suggestions.json` and write down what it predicts, so the harness can
check it against what was measured.

For every resource a description makes a claim about, add one entry to
`predictions`:

- `suggestion_index` - 1-based, matching the suggestion's `index`.
- `resource` - one of `cpu`, `gpu`, `memory`.
- `direction` - what the description says will happen to that resource:
  - `down` - it will be used less, or relieved
  - `up` - it will be used more, or taken on
  - `none` - explicitly unchanged
- `quote` - the words the prediction comes from.

Only what the description actually claims. "Move the loop to the GPU" predicts
`gpu: up` and `cpu: down`. "Vectorise with numpy" predicts `cpu: down` and
says nothing about the GPU - so do not add a GPU entry for it. Inventing a
prediction the description did not make would make the agreement rate a
measure of your guesses.

Do not consult any measurements. They are deliberately not in this packet.
