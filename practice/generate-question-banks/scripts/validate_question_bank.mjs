#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const supportedFormats = new Set(["question-bank/v1", "ml-interview-quiz/v1"]);
const filePath = process.argv[2];

if (!filePath) {
  console.error("Usage: node validate_question_bank.mjs <question-bank.json>");
  process.exit(2);
}

const errors = [];
const answerCounts = new Map();

function requireString(value, path) {
  if (typeof value !== "string" || !value.trim()) {
    errors.push(`${path} must be a non-empty string`);
    return false;
  }
  return true;
}

function countToken(text, token) {
  return text.split(token).length - 1;
}

function validateMath(text, path) {
  if (typeof text !== "string") return;
  const inlineOpen = countToken(text, "\\(");
  const inlineClose = countToken(text, "\\)");
  const displayOpen = countToken(text, "\\[");
  const displayClose = countToken(text, "\\]");

  if (inlineOpen !== inlineClose) {
    errors.push(`${path} has unbalanced \\( ... \\) delimiters`);
  }
  if (displayOpen !== displayClose) {
    errors.push(`${path} has unbalanced \\[ ... \\] delimiters`);
  }
  if (text.includes("\\texttt")) {
    errors.push(`${path} uses \\texttt; use \\mathtt for MathJax-safe code identifiers`);
  }
}

let bank;
try {
  bank = JSON.parse(await readFile(resolve(filePath), "utf8"));
} catch (error) {
  console.error(`Cannot read valid JSON from ${filePath}: ${error.message}`);
  process.exit(1);
}

if (!bank || typeof bank !== "object" || Array.isArray(bank)) {
  errors.push("bank must be a JSON object");
} else {
  if (!supportedFormats.has(bank.format)) {
    errors.push(`format must be one of: ${[...supportedFormats].join(", ")}`);
  }
  requireString(bank.title, "title");
  requireString(bank.description, "description");

  if (!Array.isArray(bank.questions) || bank.questions.length === 0) {
    errors.push("questions must be a non-empty array");
  } else {
    const questionIds = new Set();

    bank.questions.forEach((question, questionIndex) => {
      const path = `questions[${questionIndex}]`;
      if (!question || typeof question !== "object" || Array.isArray(question)) {
        errors.push(`${path} must be an object`);
        return;
      }

      ["id", "topic", "difficulty", "prompt", "answer", "explanation"].forEach((field) => {
        requireString(question[field], `${path}.${field}`);
      });

      if (typeof question.id === "string") {
        if (questionIds.has(question.id)) errors.push(`${path}.id duplicates ${question.id}`);
        questionIds.add(question.id);
      }

      validateMath(question.prompt, `${path}.prompt`);
      validateMath(question.explanation, `${path}.explanation`);

      if (!Array.isArray(question.choices) || question.choices.length < 2) {
        errors.push(`${path}.choices must contain at least two alternatives`);
      } else {
        const choiceIds = new Set();
        question.choices.forEach((choice, choiceIndex) => {
          const choicePath = `${path}.choices[${choiceIndex}]`;
          if (!choice || typeof choice !== "object" || Array.isArray(choice)) {
            errors.push(`${choicePath} must be an object`);
            return;
          }
          requireString(choice.id, `${choicePath}.id`);
          requireString(choice.text, `${choicePath}.text`);
          validateMath(choice.text, `${choicePath}.text`);
          if (typeof choice.id === "string") {
            if (choiceIds.has(choice.id)) errors.push(`${choicePath}.id duplicates ${choice.id}`);
            choiceIds.add(choice.id);
          }
        });

        if (typeof question.answer === "string" && !choiceIds.has(question.answer)) {
          errors.push(`${path}.answer does not match a choice ID`);
        }
      }

      if (typeof question.answer === "string") {
        answerCounts.set(question.answer, (answerCounts.get(question.answer) ?? 0) + 1);
      }
      if (question.reference !== undefined) {
        requireString(question.reference, `${path}.reference`);
      }
    });
  }
}

if (errors.length > 0) {
  console.error(`Question bank validation failed with ${errors.length} error(s):`);
  errors.forEach((error) => console.error(`- ${error}`));
  process.exit(1);
}

const distribution = [...answerCounts.entries()]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([id, count]) => `${id}:${count}`)
  .join(", ");

console.log(`Valid question bank: ${bank.questions.length} questions`);
console.log(`Format: ${bank.format}`);
console.log(`Answer distribution: ${distribution}`);
