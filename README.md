# custome# Customer Intelligence Platform

A backend platform for customer intelligence, churn prediction, customer segmentation, and targeted marketing built using Python, FastAPI, SQLite, and Machine Learning.

---

## Project Overview

This project analyzes customer purchasing behavior using the Olist E-Commerce dataset to:

- Predict customer churn
- Generate customer risk scores
- Segment customers
- Produce targeted marketing campaign lists
- Expose predictions through REST APIs

---

## Tech Stack

- Python 3.11
- FastAPI
- SQLite
- Pandas
- Scikit-learn
- LightGBM
- SHAP
- Pydantic
- Pytest

---

## Team

| Module | Owner |
|---------|-------|
| Data Engineering | Yashaswi |
| Feature Engineering | Lohit |
| Machine Learning | Kalyan |
| Segmentation & Campaigns | Kuushalie |
| FastAPI & Integration | Rajeswari |

---

## Repository Structure

```
customer-intelligence-platform/

├── .github/
├── app/
├── docs/
├── tests/
├── scripts/

├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── Makefile
├── README.md
└── requirements.txt
```

---

## Getting Started

### Clone Repository

```bash
git clone <repository-url>
cd customer-intelligence-platform
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

Windows

```bash
.venv\Scripts\activate
```

macOS/Linux

```bash
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Environment Variables

Copy the example configuration.

```bash
cp .env.example .env
```

Update the values if required.

---

## Development Workflow

1. Pull the latest changes from `main`
2. Create a feature branch
3. Implement your changes
4. Test locally
5. Commit your changes
6. Push your branch
7. Open a Pull Request
8. Wait for code review before merging

---

## Branch Naming

```
feature/<feature-name>

bugfix/<bug-name>

docs/<document-name>
```

Example:

```
feature/churn-model

feature/customer-api

bugfix/sqlite-join
```

---

## Commit Message Convention

```
feat: add churn prediction endpoint

fix: resolve SQLite foreign key issue

docs: update README

refactor: simplify feature engineering

test: add API unit tests
```

---

## Running the Project

Commands will be added as the project progresses.

Examples:

```bash
make install
make test
make serve
```

---

## Testing

Run all tests:

```bash
pytest
```

---

## Documentation

Additional project documentation is available in the `docs/` directory.

---

## Contributing

Please read the `CONTRIBUTING.md` file before submitting a Pull Request.

---

## License

This repository is intended for internship training and internal learning purposes.r-intelligence-platform