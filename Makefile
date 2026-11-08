PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: scout width-cloud gate-c sites download-iv qc-iv qc-rules harden-manifest site-rules-iv check test lint nocomments all step-01 step-02 step-03 step-04 step-05 step-06 step-07 step-08 step-09

check: lint test nocomments

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

nocomments:
	$(PY) tools/check_no_comments.py

step-01:
	$(PY) pipeline/01_fetch_usgs.py

step-02:
	$(PY) pipeline/02_extract_gee.py

step-03:
	$(PY) pipeline/03_build_table.py

step-04:
	$(PY) pipeline/04_audit_sample.py

step-05:
	$(PY) pipeline/05_run_validation.py

step-06:
	$(PY) pipeline/06_final_models.py

step-07:
	$(PY) pipeline/07_explain.py

step-08:
	$(PY) pipeline/08_regimes_uncertainty.py

step-09:
	$(PY) pipeline/09_figures_tables.py

all: step-01 step-02 step-03 step-04 step-05 step-06 step-07 step-08 step-09

scout:
	$(PY) pipeline/scout_stations.py

width-cloud:
	$(PY) pipeline/width_cloud.py

gate-c:
	$(PY) pipeline/gate_c.py

sites:
	$(PY) pipeline/freeze_sites.py
	$(PY) pipeline/freeze_sites.py --check-only

download-iv:
	$(PY) pipeline/download_usgs_iv.py

qc-iv:
	$(PY) pipeline/qc_usgs_iv.py --run-id $(RUN_ID)

qc-rules:
	$(PY) pipeline/qc_rule_counts.py --run-id $(RUN_ID)

harden-manifest:
	$(PY) pipeline/harden_manifest.py --run-id $(RUN_ID)

site-rules-iv:
	$(PY) pipeline/site_rules_iv.py
