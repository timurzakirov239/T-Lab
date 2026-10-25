# Portable submission verification

Run this after checking out the `research` branch or unpacking the release ZIP. The GitHub `main` branch contains only the README; the verification code and compact snapshots are on `research` and in the release ZIP.

The entry point checks the compact P001, P006, P007, and P008 evidence. It uses Python 3.12 and NumPy. It does not run the P006 runner, prepare data, fit models, or score queries. The P001 verifier is run in a temporary tree containing only its verifier and compact snapshot so the optional local archive of frozen raw inputs is not opened. P006 and P008 read compact snapshot files. P007 runs `report_expanded_search.py --verify`, which verifies the compact snapshot and independently recomputes cohort rank metrics from saved evidence.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.4
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_submission.py
```

The default prints one JSON receipt to stdout. To save a receipt, specify a new path; an existing file is never overwritten:

```powershell
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_submission.py --output task_01_auto_features/report/submission_verification.json
```

All checks must report `PASS`. The receipt includes each verifier's parsed JSON result. P001 and P006 checks rely on their existing snapshot verifiers; P007 invokes its compact snapshot verifier and recomputes metrics; P008 invokes `report_sampled_training.py --verify` against the compact P008 snapshot. These checks establish integrity and recompute compact rank summaries; they do not reproduce training or validate raw-source semantics.
