import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv


load_dotenv()


# Lightweight phrase dictionary for common chest X-ray Spanish report wording.
# This keeps artifact builds offline and deterministic.
SPANISH_TO_ENGLISH_PHRASES = [
    ("sin hallazg", "no significant findings"),
    ("sin hallazgos", "no significant findings"),
    ("dentr normal", "within normal limits"),
    ("dentro normal", "within normal limits"),
    ("cardiomegali", "cardiomegaly"),
    ("infiltr", "infiltrate"),
    ("neumoni", "pneumonia"),
    ("pinzamient", "blunting"),
    ("sen costofren", "costophrenic angle"),
    ("izquierd", "left"),
    ("derech", "right"),
    ("bilateral", "bilateral"),
    ("aortic", "aortic"),
    ("elongacion", "elongation"),
    ("sin cambi", "without significant change"),
    ("respect estudi previ", "compared with previous study"),
    ("normal", "normal"),
]

TRANSLATION_CACHE: Dict[str, str] = {}


def parse_list_field(value: Any) -> List[str]:
    """Parse metadata fields that may be list-like strings into a clean list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Try JSON/Python list parsing first (e.g., "['a', 'b']" or "[\"a\", \"b\"]").
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                continue

    # Fall back to split delimiters used in some datasets.
    for sep in ("|", ";", ","):
        if sep in text:
            return [chunk.strip() for chunk in text.split(sep) if chunk.strip()]

    return [text]


def translate_report_to_english(report_text: str) -> str:
    """Apply a lightweight Spanish-to-English translation pass for report snippets."""
    text = str(report_text).strip()
    if not text:
        return ""

    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()

    for src, dst in SPANISH_TO_ENGLISH_PHRASES:
        lowered = lowered.replace(src, dst)

    lowered = re.sub(r"\s+", " ", lowered).strip()
    if lowered:
        lowered = lowered[0].upper() + lowered[1:]
    return lowered


def _looks_spanish(text: str) -> bool:
    lower = text.lower()
    spanish_markers = [
        " sin ",
        " con ",
        " para ",
        " del ",
        " de ",
        " y ",
        " pulmon",
        " hallaz",
        " cardiomegali",
        " izquierd",
        " derech",
        " normal",
        " radiolog",
        " estudi",
    ]
    if any(marker in f" {lower} " for marker in spanish_markers):
        return True
    return bool(re.search(r"[áéíóúñ]", lower))


def _clean_translation_output(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(
        r"^the translation from spanish to english is:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def translate_report_with_api_fallback(
    report_text: str,
    enable_api_fallback: bool = False,
    translation_model: str = "gpt-4o-mini",
) -> str:
    """
    Translate report text to English with a two-step strategy:
    1) fast dictionary normalization
    2) optional OpenAI fallback for remaining Spanish text
    """
    normalized = translate_report_to_english(report_text)

    if not enable_api_fallback:
        return normalized

    if not normalized or not _looks_spanish(normalized):
        return normalized

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return normalized

    cache_key = f"{translation_model}::{normalized}"
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "Translate the following chest X-ray report text to clear medical English. "
            "Preserve clinical meaning and uncertainty. Return only translated report text.\n\n"
            f"Text: {normalized}"
        )
        response = client.responses.create(
            model=translation_model,
            input=prompt,
            temperature=0.0,
        )
        translated = _clean_translation_output(getattr(response, "output_text", ""))
        final_text = translated if translated else normalized
        TRANSLATION_CACHE[cache_key] = final_text
        return final_text
    except Exception as exc:
        if os.getenv("TRANSLATION_DEBUG") == "1":
            print(f"[translation-fallback] API call failed: {type(exc).__name__}: {exc}")
        return normalized


def serialize_list_field(items: List[str]) -> str:
    """Serialize lists as compact JSON for durable CSV storage."""
    return json.dumps(items, ensure_ascii=True)


def _is_numeric_token(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _build_root_candidates(images_root: Optional[str], csv_parent: Path) -> List[Path]:
    raw_candidates: List[Path] = []
    if images_root and str(images_root).strip():
        raw_candidates.append(Path(images_root))
    else:
        raw_candidates.extend([Path("."), csv_parent, csv_parent / "sample", csv_parent / "images"])

    deduped: List[Path] = []
    seen = set()
    for item in raw_candidates:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_image_path(
    image_id: Any,
    image_dir: Optional[Any],
    root_candidates: Optional[List[Path]] = None,
) -> Path:
    image_name = str(image_id).strip()
    if not image_name:
        return Path("")

    if Path(image_name).is_absolute():
        return Path(image_name)

    roots = root_candidates if root_candidates else [Path(".")]
    dir_name = str(image_dir).strip() if image_dir is not None else ""

    # Prefer paths that already exist to prevent propagating bad path guesses.
    for base_dir in roots:
        fallback_path = base_dir / image_name
        if fallback_path.exists():
            return fallback_path

    if dir_name:
        for base_dir in roots:
            dir_path = base_dir / dir_name / image_name
            if dir_path.exists():
                return dir_path

    # If nothing exists yet, avoid using purely numeric ImageDir values such as "53.0".
    if dir_name and not _is_numeric_token(dir_name):
        return roots[0] / dir_name / image_name
    return roots[0] / image_name


def load_padchest_metadata(
    csv_path: str,
    images_root: Optional[str],
    image_id_col: str = "ImageID",
    image_dir_col: str = "ImageDir",
    report_col: str = "Report",
    labels_col: str = "Labels",
    locations_col: str = "Localizations",
    translate_reports_to_english: bool = True,
    enable_report_translation_api_fallback: bool = False,
    report_translation_model: str = "gpt-4o-mini",
    drop_missing_image_ids: bool = True,
) -> pd.DataFrame:
    """
    Load and normalize PadChest metadata.

    Output columns:
    - image_id
    - image_path
    - report
    - labels (python list[str])
    - locations (python list[str])
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_file)
    required = [image_id_col, labels_col, locations_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    root_candidates = _build_root_candidates(images_root=images_root, csv_parent=csv_file.parent)

    image_dirs = df[image_dir_col] if image_dir_col in df.columns else [None] * len(df)

    if report_col in df.columns:
        reports = df[report_col]
    else:
        reports = [""] * len(df)

    rows: List[Dict[str, Any]] = []
    for image_id, image_dir, report, labels, locations in zip(
        df[image_id_col],
        image_dirs,
        reports,
        df[labels_col],
        df[locations_col],
    ):
        image_id_text = str(image_id).strip()
        if drop_missing_image_ids and not image_id_text:
            continue

        path = build_image_path(
            image_id=image_id_text,
            image_dir=image_dir,
            root_candidates=root_candidates,
        )
        rows.append(
            {
                "image_id": image_id_text,
                "image_path": str(path),
                "report": (
                    ""
                    if pd.isna(report)
                    else (
                        translate_report_with_api_fallback(
                            str(report).strip(),
                            enable_api_fallback=enable_report_translation_api_fallback,
                            translation_model=report_translation_model,
                        )
                        if translate_reports_to_english
                        else str(report).strip()
                    )
                ),
                "labels": parse_list_field(labels),
                "locations": parse_list_field(locations),
            }
        )

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError("No usable rows after metadata normalization.")

    return output_df


def save_retrieval_metadata(
    metadata_df: pd.DataFrame,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
) -> None:
    if not output_csv and not output_json:
        raise ValueError("At least one of output_csv or output_json must be provided.")

    if output_csv:
        out_csv = Path(output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        write_df = metadata_df.copy()
        write_df["labels"] = write_df["labels"].apply(serialize_list_field)
        write_df["locations"] = write_df["locations"].apply(serialize_list_field)
        write_df.to_csv(out_csv, index=False)

    if output_json:
        out_json = Path(output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        records = metadata_df.to_dict(orient="records")
        out_json.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")


def _normalize_retrieval_metadata_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    expected = {"image_id", "labels", "locations"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"Retrieval metadata from {source_label} must include {sorted(expected)}. "
            f"Found: {list(df.columns)}"
        )

    if "image_path" not in df.columns:
        df["image_path"] = ""
    else:
        df["image_path"] = df["image_path"].fillna("").astype(str)

    if "report" not in df.columns:
        df["report"] = ""
    else:
        df["report"] = df["report"].fillna("").astype(str)

    # Optional locator fields for resolving images from a local dataset root.
    optional_remote_cols = ["source_zip", "train_folder", "image_filename"]
    for col in optional_remote_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    df["labels"] = df["labels"].apply(parse_list_field)
    df["locations"] = df["locations"].apply(parse_list_field)
    return df


def load_retrieval_metadata(metadata_path: str) -> pd.DataFrame:
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"Retrieval metadata file not found: {metadata_path}")

    if meta_file.suffix.lower() == ".json":
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Retrieval metadata JSON must be a list of records: {metadata_path}")
        df = pd.DataFrame(raw)
    else:
        df = pd.read_csv(meta_file)

    return _normalize_retrieval_metadata_df(df=df, source_label=str(meta_file))
