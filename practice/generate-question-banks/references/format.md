# Question bank format

## Required top-level fields

```json
{
  "format": "question-bank/v1",
  "title": "Probability practice",
  "description": "A medium-to-hard probability question bank.",
  "questions": []
}
```

- `format`: use `question-bank/v1` for new banks.
- `title`: non-empty human-readable title.
- `description`: concise scope and audience statement.
- `questions`: non-empty array of question objects.

The validator also accepts the legacy `ml-interview-quiz/v1` format when maintaining an existing bank.

## Question object

```json
{
  "id": "prob-001",
  "topic": "Conditional probability",
  "difficulty": "medium",
  "prompt": "If \\(P(A)=0.4\\) and \\(P(B\\mid A)=0.5\\), what is \\(P(A\\cap B)\\)?",
  "choices": [
    { "id": "A", "text": "\\(0.20\\)" },
    { "id": "B", "text": "\\(0.40\\)" },
    { "id": "C", "text": "\\(0.50\\)" },
    { "id": "D", "text": "\\(0.90\\)" }
  ],
  "answer": "A",
  "explanation": "Use \\(P(A\\cap B)=P(B\\mid A)P(A)=0.5\\times0.4=0.20\\).",
  "reference": "notes/probability.md — Conditional probability"
}
```

Required fields:

- `id`: stable unique string.
- `topic`: concept or subject area.
- `difficulty`: non-empty label such as `easy`, `medium`, `hard`, or `very hard`.
- `prompt`: complete question with all necessary assumptions.
- `choices`: at least two `{id, text}` objects with unique IDs.
- `answer`: exactly one choice ID.
- `explanation`: concise reasoning for the correct answer.

Optional fields:

- `reference`: local path and section, URL, publication locator, or other verifiable source pointer.
- Additional metadata may be included if it does not replace required fields.

## TeX in JSON

Use MathJax delimiters and JSON escaping:

```json
{
  "prompt": "For \\(f(x)=x^2\\), what is \\(f'(x)\\)?",
  "choices": [
    { "id": "A", "text": "\\(2x\\)" },
    { "id": "B", "text": "\\(x\\)" }
  ]
}
```

Use `\\mathtt{max\\_features}` for code identifiers rendered within mathematics. Keep ordinary code names outside TeX when no mathematical expression surrounds them.

## Quality checks

- Each correct answer is unique and defensible.
- Alternatives are mutually exclusive under the prompt's assumptions.
- Stored answer positions are reasonably balanced.
- No question duplicates another question's learning objective and framing.
- Difficulty matches the target audience.
- Source-grounded claims have accurate references.
- TeX delimiters are balanced and supported by MathJax.
- The deterministic validator passes.
