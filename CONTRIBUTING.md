# Contributing Guide

Welcome to the Customer Intelligence Platform project.

Please follow these guidelines when contributing to ensure a consistent development workflow.

---

## Development Workflow

1. Pull the latest changes from `main`.
2. Create a new feature branch.
3. Implement your changes.
4. Test your code locally.
5. Commit using meaningful commit messages.
6. Push your branch to GitHub.
7. Create a Pull Request.
8. Wait for code review before merging.

---

## Branch Naming

Use one of the following conventions:

feature/<feature-name>
bugfix/<issue-name>
docs/<document-name>
refactor/<module-name>

### Examples

feature/data-ingestion

feature/churn-label

feature/api-predict

bugfix/sqlite-join

docs/readme-update

---

## Commit Message Convention

Use descriptive commit messages.

Examples:

feat: add SQLite ingestion pipeline

feat: implement churn prediction endpoint

fix: handle missing customer records

refactor: simplify feature engineering module

docs: update project README

test: add unit tests for API

Avoid commit messages like:

update

changes

final

test

abc

---

## Pull Request Checklist

Before creating a Pull Request, ensure:

- Code builds successfully
- Code has been tested locally
- No unnecessary files are committed
- Documentation has been updated (if required)
- Commit history is clean

---

## Coding Standards

- Follow PEP 8
- Use meaningful variable and function names
- Keep functions focused and reusable
- Add comments only when necessary
- Handle exceptions appropriately
- Remove unused imports and dead code

---

## Project Structure

Each intern owns their assigned module.

data/
features/
ml/
segmentation/
api/

Do not modify another module without discussing it first.

---

## Code Review

All changes must be submitted through a Pull Request.

Do not merge your own Pull Request without approval.

Address review comments before requesting another review.

---

## Questions

If you're blocked or unsure about an implementation, ask before making major architectural changes.