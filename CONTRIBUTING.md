# Contributing to EngramVault

Thank you for your interest in contributing to **EngramVault**! We welcome community contributions, bug fixes, adapters for new AI providers, and UI improvements.

## Code of Conduct

Please treat everyone with respect and empathy. We are committed to providing a friendly, welcoming, and harassment-free environment for everyone.

---

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/gaskb/engramvault.git
cd engramvault
```

### 2. Set Up a Virtual Environment

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install pytest httpx
```

Or using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### 3. Run the Test Suite

Before making any changes, ensure all tests pass:
```bash
pytest
```
EngramVault has a zero-regression policy: all 100+ tests must pass.

---

## Development Guidelines

### Architecture & Epistemic Hierarchy
- Keep the separation between raw data (`conversations`, `messages`, `chunks`), extracted suggestions (`claims`, `relations`), and verified knowledge (`facts`, `decisions`).
- Respect the **Zero External Heavy Dependencies** philosophy: EngramVault relies on SQLite (FTS5 + vectors) and pure Python/NumPy for default operations. Avoid adding external heavyweight database engines as mandatory requirements.

### Coding Standards
- **Python:** 3.10+ compatible. Use type annotations where practical.
- **Frontend:** Pure HTML/CSS/Vanilla JS (served via Jinja2 templates) without complex frontend build chains. Keep it lightweight and fast.
- **Formatting & Linting:** Run `ruff check .` before submitting.

---

## Submitting a Pull Request

1. Create a descriptive branch:
   ```bash
   git checkout -b feat/add-new-adapter
   ```
2. Write unit/integration tests for your changes in `tests/`.
3. Verify that `pytest` passes with 100% success.
4. Commit your changes with clear semantic commit messages (e.g. `feat: add mistral adapter`, `fix: handle edge case in zip import`).
5. Open a Pull Request against the `main` branch on GitHub.

Thank you for helping make AI memory accessible, private, and verifiable!
