# CI/CD Pipeline Documentation

## Overview

This document describes the automated CI/CD workflows for the Weft project. All workflows are configured to run on GitHub Actions and help maintain code quality, security, and reliability.

## Workflows

### 1. **CI Pipeline** (`ci.yml`)
**Triggers:** Push to `main`/`develop`, Pull Requests

**What it does:**
- Runs tests on Python 3.11 and 3.12
- Sets up PostgreSQL with pgvector for database testing
- Runs pytest with coverage reporting (minimum 50% coverage)
- Uploads coverage reports to Codecov
- Archives HTML coverage reports as artifacts

**Key Features:**
- Multi-version Python testing
- Database service setup and health checks
- Database migrations validation
- Test timeouts (10s per test)
- Coverage thresholds

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage ≥ 50%
- ✅ No timeouts
- ✅ Database migrations successful

---

### 2. **Linting & Code Quality** (`lint.yml`)
**Triggers:** Push to `main`/`develop`, Pull Requests

**What it does:**
- Checks import sorting (isort)
- Validates code formatting (Black)
- Runs static analysis (Ruff)
- Performs type checking (mypy)

**Configuration:**
- Black line length: 88 characters
- Mypy checks with ignore-missing-imports enabled
- Ruff checks for PEP8 compliance
- isort uses Black-compatible profile

**Note:** Warnings are non-blocking (continue-on-error: true) to allow gradual improvements

---

### 3. **Security Scanning** (`security.yml`)
**Triggers:** Push to `main`/`develop`, Pull Requests, Weekly schedule (Sundays 2 AM UTC)

**What it does:**
- Runs Bandit for security vulnerability detection
- Checks for known vulnerable dependencies (Safety)
- Generates JSON reports for detailed review
- Fails on HIGH/CRITICAL security issues

**Severity Levels:**
- 🔴 **CRITICAL/HIGH** → CI fails (blocks merge)
- 🟡 **MEDIUM** → Warning only
- 🟢 **LOW** → Logged but non-blocking

**Reports:**
- Uploaded as artifacts for 30 days
- Available for security review

---

### 4. **Code Formatting Auto-Fix** (`format.yml`)
**Triggers:** Push to `develop`

**What it does:**
- Automatically formats code on develop branch
- Runs isort, Black, and Ruff
- Commits fixes back to the branch
- Useful for keeping code clean

**Behavior:**
- Only runs if changes are detected
- Uses bot account for commits
- Commits message: "style: auto-format code..."

---

### 5. **Pull Request Validation** (`pr-validation.yml`)
**Triggers:** Pull Request opened/updated

**What it does:**
- Validates commit message format (conventional commits)
- Checks for accidentally committed secrets
- Warns about large files (>50MB)
- Provides validation summary

**Commit Format:**
```
<type>(<scope>): <description>

Examples:
- feat(storage): add pgvector embeddings
- fix(core): handle missing data gracefully
- docs: update README with setup instructions
- chore(deps): upgrade langchain-core
```

---

### 6. **Release & Deploy Validation** (`release.yml`)
**Triggers:** Push to `main`, tagged releases

**What it does:**
- Validates semantic versioning format (v1.2.3)
- Verifies all database migrations are valid
- Checks that required documentation exists
- Creates GitHub releases automatically

**Tag Format:**
```
v1.0.0              # Release version
v2.1.0-alpha        # Pre-release
v2.1.0-beta.1       # Beta release
```

---

### 7. **Dependabot Auto-Merge** (`dependabot.yml` config + `dependabot.yml` workflow)

**Configuration:**
- Checks for dependency updates weekly (Mondays 3 AM UTC)
- Limits to 5 open PRs
- Labels: `dependencies`, `automated`
- Auto-approves updates

**Behavior:**
- 🟢 Direct production dependencies: Auto-approved
- 🟡 Transitive dependencies: Requires manual review

---

## Local Development Setup

### Pre-commit Hooks

Install pre-commit hooks to catch issues before pushing:

```bash
pip install pre-commit
pre-commit install
```

This will:
- ✅ Auto-format code (Black, isort)
- ✅ Check for secrets
- ✅ Validate YAML/JSON
- ✅ Remove trailing whitespace
- ✅ Run type checks

Run hooks manually:
```bash
pre-commit run --all-files
```

### Running Tests Locally

```bash
# Start PostgreSQL with pgvector
docker compose up -d postgres

# Wait for database to be ready
sleep 5

# Run migrations
alembic upgrade head

# Run tests
pytest tests/ --cov=Weft --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Code Quality Checks

```bash
# Format code
black Weft/ tests/
isort Weft/ tests/

# Lint
ruff check Weft/ tests/

# Type checking
mypy Weft/ --ignore-missing-imports

# Security scanning
bandit -r Weft/
safety check
```

---

## GitHub Repository Settings

For optimal CI/CD experience, configure:

1. **Branch Protection Rules** (Settings → Branches):
   - Require status checks to pass:
     - CI Pipeline
     - Linting & Code Quality
     - Security Scanning
   - Require code reviews: 1+ reviews
   - Dismiss stale reviews: enabled
   - Require branches to be up to date: enabled

2. **Actions Permissions**:
   - Allow all actions and reusable workflows
   - Enable artifact retention

3. **Dependabot Settings**:
   - Alerts: Enabled
   - Security updates: Enabled
   - Version updates: Enabled (see `.github/dependabot.yml`)

---

## Troubleshooting

### Tests Fail Locally But Pass in CI

**Common causes:**
1. Database not running: `docker compose up -d postgres`
2. Migrations not applied: `alembic upgrade head`
3. Dependencies out of sync: `pip install -r requirements.txt`
4. Python version mismatch: Use Python 3.11+

### CI Pipeline Timeout

**Solutions:**
- Check for infinite loops in code
- Increase timeout in `ci.yml` (default: 10s per test)
- Mark slow tests with `@pytest.mark.slow`

### Security Scan Failures

**Review the reports:**
1. Download `bandit-report.json` from workflow artifacts
2. Check for false positives
3. Update `.github/workflows/security.yml` to ignore if necessary

### Dependabot Not Creating PRs

**Check:**
- `.github/dependabot.yml` exists and is valid YAML
- GitHub Actions is enabled
- Branch `develop` exists (configured branch)

---

## Coverage Goals

Current minimum coverage: **50%**

For better test suite quality, consider:
- Target: 70-80% coverage
- Focus on critical paths and error handling
- Use `--cov-report=html` to identify gaps

```bash
pytest tests/ --cov=Weft --cov-report=html
open htmlcov/index.html
```

---

## Contributing with CI/CD

### Before Creating a PR:

1. ✅ Run tests locally: `pytest tests/`
2. ✅ Format code: `black . && isort .`
3. ✅ Check types: `mypy Weft/`
4. ✅ Lint: `ruff check .`
5. ✅ Use conventional commits: `feat(module): description`

### PR Review Checklist:

- [ ] All CI checks pass
- [ ] Code coverage maintained/improved
- [ ] No security issues detected
- [ ] Commit messages are clear
- [ ] Documentation updated
- [ ] At least 1 approval

---

## Next Steps

1. **Customize severity thresholds** in `.github/workflows/security.yml`
2. **Set branch protection rules** as described above
3. **Configure Codecov** (optional) for detailed coverage tracking
4. **Add issue templates** in `.github/ISSUE_TEMPLATE/`
5. **Create pull request template** in `.github/pull_request_template.md`

---

## Contact & Support

For questions about CI/CD setup, refer to:
- GitHub Actions documentation: https://docs.github.com/actions
- Ruff documentation: https://docs.astral.sh/ruff/
- Pytest documentation: https://docs.pytest.org/