# DTI - HN SPIDERplan Scorecard

Streamlit prototype for head and neck SPIDERplan scorecard review using RT Plan/RP, RT Structure/RS, and RT Dose/RD DICOM files.

## Current features

- Single-plan SPIDERplan scorecard
- Optional two-plan comparison mode
- Side-by-side final grades and scores
- Overlapping overall SPIDERplan radar chart
- Overlapping target-volume radar comparison
- Overlapping OAR radar comparison
- Final metric scorecard with color-coded status:
  - Green = passed/achieved
  - Yellow = marginal
  - Red = failed
- Per-target Rx assignment for PTV/CTV/GTV structures
- `_eval` structure support for V105% review
- Highest-dose PTV V105% evaluation without requiring `_eval`
- Global max hotspot review retained separately
- Helper-contour exclusions including z* structures, LN helper structures, and target optimization contours ending in `opti`

## Required files

Each scored plan requires:

- RT Plan / RP
- RT Structure / RS
- RT Dose / RD

Upload Plan A first. Upload Plan B to activate comparison mode.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub notes

Do not upload patient DICOM files, uploads folders, virtual environments, or `__pycache__` folders.
