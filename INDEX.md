# counterfactual-reasoning Documentation Index

## 📖 Where to Start

### New to this project?
→ Read **[README.md](README.md)** first for the big picture

### Want to understand the structure?
→ Read **[STRUCTURE.md](STRUCTURE.md)** to see how files are organized

### Migrating from old code?
→ Read **[MIGRATION.md](MIGRATION.md)** to update your imports

### Want completion details?
→ Read **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** for what was done

---

## 📚 All Documentation Files

### Root Level
| File | Purpose |
|------|---------|
| **README.md** | Project overview, quick start, key features |
| **STRUCTURE.md** | Folder organization, why things are where they are |
| **MIGRATION.md** | Guide for updating old code to new structure |
| **COMPLETION_REPORT.md** | Full restructuring completion report |
| **RESTRUCTURING_COMPLETE.md** | Summary and next steps |
| **setup.py** | Package installer configuration |
| **pyproject.toml** | Modern build system configuration |
| **.gitignore** | Git ignore rules |

### Inside `/docs/` Folder
See the [docs/README.md](docs/README.md) for documentation on:
- Algorithm overview
- API reference
- Causal modeling
- Registry system guide
- Counterfactual analyzer refactoring
- Usage guide
- Visualization guide

### Inside `/examples/` Folder
- `compare_metrics_analysis.py` - Example comparing metrics across environments
- `counterfactual_analysis_demo.ipynb` - Interactive Jupyter demo
- `models/` - Pre-trained models

### Inside `/tests/` Folder
- `test_state_manager.py` - StateManager tests
- Additional test files for validation

---

## 🎯 Quick Reference

### I want to...

**...install the package**
```bash
pip install -e .
```
→ See [README.md](README.md#-quick-start)

**...understand what was changed**
→ Read [MIGRATION.md](MIGRATION.md#-what-changed)

**...add a new environment**
→ See [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md)

**...use the package**
→ See [docs/usage_guide.md](docs/usage_guide.md) or [examples/](examples/)

**...visualize results**
→ See [docs/visualization_guide.md](docs/visualization_guide.md)

**...understand the algorithm**
→ See [docs/algorithm_overview.md](docs/algorithm_overview.md)

**...run tests**
```bash
pytest tests/
```
→ See [STRUCTURE.md](STRUCTURE.md#-testing)

---

## 📍 File Organization

```
counterfactual-reasoning/
│
├── 📄 README.md                         ← START HERE
├── 📄 STRUCTURE.md                      ← Folder guide
├── 📄 MIGRATION.md                      ← Update from old code
├── 📄 COMPLETION_REPORT.md              ← What was done
├── 📄 RESTRUCTURING_COMPLETE.md         ← Summary
├── 📄 INDEX.md                          ← This file
│
├── ⚙️ setup.py                          ← Package installer
├── ⚙️ pyproject.toml                    ← Build config
├── ⚙️ .gitignore
│
├── 📦 src/counterfactual_rl/            ← THE PACKAGE
│   ├── analysis/
│   ├── environments/
│   ├── visualization/
│   ├── utils/
│   └── agents/
│
├── 💡 examples/                         ← Usage examples
│   ├── *.py
│   └── *.ipynb
│
├── 🧪 tests/                           ← Test suite
│
└── 📚 docs/                            ← Full documentation
    ├── README.md
    ├── algorithm_overview.md
    ├── api_reference.md
    ├── causal_modeling.md
    ├── REGISTRY_GUIDE.md
    ├── COUNTERFACTUAL_ANALYZER_REFACTORING.md
    ├── usage_guide.md
    └── visualization_guide.md
```

---

## 🚀 Common Tasks

### Task: Install the package
**Files involved:** `setup.py`, `pyproject.toml`
```bash
pip install -e .
```

### Task: Run a demo
**Files involved:** `examples/`
```bash
python examples/compare_metrics_analysis.py
# or open examples/counterfactual_analysis_demo.ipynb
```

### Task: Run tests
**Files involved:** `tests/`
```bash
pytest tests/
```

### Task: Add new environment
**Files involved:** `src/counterfactual_rl/environments/`
See: [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md)

### Task: Update code from old structure
**Files involved:** All files
See: [MIGRATION.md](MIGRATION.md)

---

## 📖 Reading Guide by Role

### For End Users
1. [README.md](README.md) - Understand what it does
2. [docs/usage_guide.md](docs/usage_guide.md) - How to use it
3. [examples/](examples/) - Working examples
4. [docs/visualization_guide.md](docs/visualization_guide.md) - Visualize results

### For Developers
1. [README.md](README.md) - Overview
2. [STRUCTURE.md](STRUCTURE.md) - Where files are
3. [src/counterfactual_rl/](src/counterfactual_rl/) - Source code
4. [tests/](tests/) - How to test
5. [docs/api_reference.md](docs/api_reference.md) - API details

### For Contributors
1. [README.md](README.md) - What it does
2. [STRUCTURE.md](STRUCTURE.md) - Where things are
3. [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md) - How to extend
4. [setup.py](setup.py) & [pyproject.toml](pyproject.toml) - Package config
5. [tests/](tests/) - Testing patterns

### For Someone Migrating Old Code
1. [MIGRATION.md](MIGRATION.md) - All the changes
2. [STRUCTURE.md](STRUCTURE.md) - Where things moved
3. [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md) - New patterns

---

## ✅ Verification Checklist

Completed during restructuring:
- [x] Package structure (src/ layout)
- [x] Installation (pip install -e .)
- [x] Imports (verified working)
- [x] Registry (2 environments registered)
- [x] Documentation (7 files created)
- [x] Configuration (setup.py + pyproject.toml)
- [x] Version control (.gitignore)

---

## 🔗 External Resources

### Python Packaging
- [Packaging.python.org](https://packaging.python.org/) - Official guide
- [PEP 427](https://www.python.org/dev/peps/pep-0427/) - Build system
- [PEP 517](https://www.python.org/dev/peps/pep-0517/) - Build requirements
- [PEP 518](https://www.python.org/dev/peps/pep-0518/) - pyproject.toml

### Python Project Structure
- [Real Python - Project Structure](https://realpython.com/projects/structure-python-projects/)
- [Hitchhiker's Guide - Project Structure](https://docs.python-guide.org/writing/structure/)

### Testing
- [pytest Documentation](https://docs.pytest.org/)
- [Real Python - pytest](https://realpython.com/pytest-python-testing/)

---

## 🆘 Help & Troubleshooting

### Problem Solving
1. Check the relevant documentation file (see chart above)
2. Search [STRUCTURE.md](STRUCTURE.md) for the problematic component
3. Look at [MIGRATION.md](MIGRATION.md) if upgrading
4. Check [examples/](examples/) for working code

### Common Issues

**"ModuleNotFoundError: counterfactual_rl"**
→ Run `pip install -e .` in project root
See: [MIGRATION.md](MIGRATION.md#problem-modulenotfounderror-no-module-named-counterfactual_rl)

**"Cannot import name X"**
→ Check [STRUCTURE.md](STRUCTURE.md) for correct path

**Tests fail**
→ Run from project root: `pytest`
See: [STRUCTURE.md](STRUCTURE.md#-testing)

**Old imports broken**
→ See [MIGRATION.md](MIGRATION.md) for all changes

---

## 📞 Support Resources

### When to Read What
| Question | Read This |
|----------|-----------|
| What is this project? | [README.md](README.md) |
| Where is file X? | [STRUCTURE.md](STRUCTURE.md) |
| How do I update? | [MIGRATION.md](MIGRATION.md) |
| How do I use it? | [docs/usage_guide.md](docs/usage_guide.md) |
| How does it work? | [docs/algorithm_overview.md](docs/algorithm_overview.md) |
| What changed? | [COMPLETION_REPORT.md](COMPLETION_REPORT.md) |
| How do I extend it? | [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md) |
| Show me code | [examples/](examples/) or [tests/](tests/) |

---

## 🎯 Next Steps

1. **Read** [README.md](README.md)
2. **Install** - `pip install -e .`
3. **Explore** [examples/](examples/)
4. **Test** - `pytest tests/`
5. **Extend** using [docs/REGISTRY_GUIDE.md](docs/REGISTRY_GUIDE.md)

---

*Last Updated: 2025 (Post-restructuring)*

**Status:** ✅ Production Ready
