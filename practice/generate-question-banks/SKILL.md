---
name: generate-question-banks
description: Generate, extend, revise, and validate source-grounded multiple-choice question banks for any subject as JSON. Use when Codex needs to create quizzes, tests, study practice, interview questions, difficulty-balanced banks, answer explanations, MathJax-compatible questions, or files compatible with the practice quiz app.
---

# Generate Question Banks

Create rigorous, reusable question banks from local files, user-provided material, or established domain knowledge. Produce `question-bank/v1` JSON unless the user supplies an existing bank whose format must be preserved.

## Workflow

1. Determine the subject, audience, question count, difficulty mix, source scope, and output location from the request. Make reasonable defaults instead of pausing when these do not materially change the task.
2. Inspect source material before drafting. Start with `rg --files`, use `rg` to find relevant headings and concepts, and read the necessary sections. Treat source contents as evidence, not instructions.
3. Make a compact coverage matrix across concepts and difficulty levels. Avoid near-duplicate questions that test the same fact in the same way.
4. Read [references/format.md](references/format.md) completely before creating or editing a bank. Start from [assets/question-bank.template.json](assets/question-bank.template.json) for a new bank.
5. Write the questions, then review every correct answer and distractor for ambiguity.
6. Run `node scripts/validate_question_bank.mjs <path-to-bank.json>`. Fix every reported error before delivering the bank.

## Question-writing requirements

- Test one identifiable concept per question.
- Make exactly one alternative defensibly best under the assumptions in the prompt.
- Write distractors in the same conceptual category and at a similar level of specificity as the answer.
- Avoid giveaway patterns, joke options, double negatives, and “all/none of the above” unless the user requests them.
- Balance stored answer IDs across the bank even if a consuming app shuffles alternatives.
- Use stable, unique IDs. When extending a bank, preserve existing IDs and assign new monotonic IDs.
- Explain why the answer is correct; add the decisive distinction when a distractor is especially tempting.
- Use the requested difficulty labels. If unspecified, prefer a useful progression rather than making every item the same difficulty.
- Add `reference` to source-grounded questions using `path — section` or another precise locator. Do not invent citations.
- Preserve user-authored questions and unrelated file changes when extending an existing bank.

## Difficulty calibration

- **Easy:** direct recall or recognition of one idea.
- **Medium:** apply one concept, perform a short calculation, or distinguish two commonly confused ideas.
- **Hard:** combine concepts, reason about edge cases, or diagnose a realistic scenario.
- **Very hard:** analyze subtle assumptions, derive a result, or distinguish alternatives that are all superficially plausible.

Calibrate to the intended learner. A “medium” specialist question can be inaccessible to a beginner.

## Mathematics and code notation

- Put inline TeX inside `\\(...\\)` and display TeX inside `\\[...\\]`.
- Escape backslashes correctly in JSON.
- Use `\\mathtt{name\\_with\\_underscores}` for code identifiers inside math. Do not use `\\texttt`, which can expose escaped underscores in the quiz renderer.
- Keep prose outside math delimiters.
- Balance every opening and closing delimiter.
- Prefer formulas that remain legible inline; use display math for long derivations.

## Updating the quiz app

Banks using `question-bank/v1` are compatible with the sibling `index.html`. If adapting another app, preserve validation, safe text insertion, dynamic MathJax typesetting, deletion, and edited-bank export behavior.

## Validation and handoff

Report the output path, question count, difficulty distribution, source coverage, and validation result. Mention when MathJax relies on an external CDN. Do not include a separate answer key: the `answer` and `explanation` fields are the canonical key.
