import os
import re
from collections import Counter
from typing import Any, Dict, List

from dotenv import load_dotenv


load_dotenv()


LABEL_EXPLANATIONS = {
    "normal": "No major abnormal chest finding was identified in similar reference images.",
    "cardiomegaly": "The heart silhouette appears larger than usual on similar X-rays.",
    "opacity": "An area appears denser/whiter than expected, which can reflect several causes.",
    "infiltrates": "Patchy increased lung density is seen in similar cases.",
    "consolidation": "A region of lung appears more solid or dense in similar images.",
    "pleural effusion": "Fluid is present in the pleural space in similar reference studies.",
    "interstitial pattern": "Fine linear or reticular markings are more prominent in similar cases.",
    "alveolar pattern": "Air-space density pattern appears in similar X-rays.",
    "costophrenic angle blunting": "The lower outer lung angle appears less sharp, often related to fluid or scarring.",
    "nodule": "A small rounded lung spot is seen in similar images.",
    "pulmonary fibrosis": "Scarring-like chronic lung changes are present in similar reference cases.",
    "pneumonia": "An infection-like lung opacity pattern appears in similar studies.",
    "air trapping": "Some lung regions may retain extra air on breathing phases in similar cases.",
    "aortic elongation": "The aorta contour appears elongated in similar images.",
    "supra aortic elongation": "The upper aortic arch contour appears elongated in similar cases.",
    "vascular hilar enlargement": "The central hilar vascular shadows look more prominent in similar images.",
    "vascular redistribution": "Pulmonary blood flow appears redistributed in similar reference cases.",
    "heart insufficiency": "A heart-failure-like radiographic pattern appears in similar studies.",
    "laminar atelectasis": "A thin band-like area of partial lung collapse is present in similar images.",
    "diaphragmatic eventration": "A focal elevated contour of the diaphragm is seen in similar cases.",
    "dual chamber device": "A two-lead cardiac rhythm device is visible in similar images.",
    "pacemaker": "A cardiac pacing device is visible in similar cases.",
    "nsg tube": "A nasogastric tube is visible in similar images.",
    "rib fracture": "A rib break is seen in similar reference X-rays.",
    "vertebral degenerative changes": "Age-related spine wear-and-tear changes are present in similar images.",
    "kyphosis": "An increased forward curvature of the thoracic spine appears in similar studies.",
    "copd signs": "Hyperinflation/airway-related chronic obstructive pattern appears in similar images.",
    "unchanged": "Compared reports in similar cases describe no significant interval change.",
}


def _normalize_label_key(label: str) -> str:
    key = label.strip().lower()
    key = re.sub(r"\s+", " ", key)
    return key


def _get_label_explanation(label: str) -> str:
    normalized = _normalize_label_key(label)
    if normalized in LABEL_EXPLANATIONS:
        return LABEL_EXPLANATIONS[normalized]
    return (
        "This label appears in similar reference images; interpretation depends on full "
        "clinical context and radiologist review."
    )


def _build_label_explanations(labels: List[str]) -> List[Dict[str, str]]:
    return [{"label": label, "explanation": _get_label_explanation(label)} for label in labels]


def build_rag_context(retrieved_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build structured context from retrieved labels and anatomical locations."""
    label_counter: Counter = Counter()
    location_counter: Counter = Counter()
    report_snippets: List[str] = []

    for case in retrieved_cases:
        label_counter.update(case.get("labels", []))
        location_counter.update(case.get("locations", []))
        report_text = str(case.get("report", "")).strip()
        if report_text:
            report_snippets.append(report_text)

    top_labels = [item for item, _ in label_counter.most_common(5)]
    top_locations = [item for item, _ in location_counter.most_common(5)]

    case_lines = []
    for case in retrieved_cases:
        case_lines.append(
            {
                "image_id": case.get("image_id", "unknown"),
                "report": case.get("report", ""),
                "labels": case.get("labels", []),
                "locations": case.get("locations", []),
                "distance": case.get("distance", None),
            }
        )

    summary_text = (
        f"Most frequent similar findings: {', '.join(top_labels) if top_labels else 'none'}. "
        f"Most frequent anatomical regions: {', '.join(top_locations) if top_locations else 'none'}."
    )

    top_report_snippets = report_snippets[:3]

    label_explanations = _build_label_explanations(top_labels)

    return {
        "top_labels": top_labels,
        "top_locations": top_locations,
        "top_report_snippets": top_report_snippets,
        "label_explanations": label_explanations,
        "cases": case_lines,
        "summary_text": summary_text,
    }


def _placeholder_explanation(context: Dict[str, Any]) -> str:
    labels = context.get("top_labels", [])
    locations = context.get("top_locations", [])
    top_report_snippets = context.get("top_report_snippets", [])
    label_explanations = context.get("label_explanations", [])

    label_text = ", ".join(labels[:3]) if labels else "non-specific chest findings"
    location_text = ", ".join(locations[:3]) if locations else "chest regions"

    details_lines = []
    for item in label_explanations[:5]:
        details_lines.append(f"- {item['label']}: {item['explanation']}")
    details_text = "\n".join(details_lines)

    base = (
        "This X-ray appears similar to prior cases in the dataset with "
        f"findings such as {label_text}, often noted in {location_text}. "
        "This is a similarity-based summary and not a medical diagnosis. "
        "Please consult a qualified clinician for interpretation."
    )
    report_section = ""
    if top_report_snippets:
        quoted = "\n".join([f"- {snippet}" for snippet in top_report_snippets])
        report_section = f"\n\nReference report snippets from similar cases:\n{quoted}"

    if details_text:
        return f"{base}{report_section}\n\nLabel-wise explanation:\n{details_text}"
    return f"{base}{report_section}"


def _openai_explanation(context: Dict[str, Any], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = (
"You are a helpful assistant explaining chest X-ray similarities to a patient in the simplest way possible.\n\n"
"Your role is NOT to diagnose. You are only explaining what similar cases showed.\n\n"

"Guidelines:\n"
"- Use very simple, everyday language (avoid medical jargon or explain it clearly).\n"
"- Be calm, reassuring, and easy to understand.\n"
"- Do NOT say anything is confirmed.\n"
"- Do NOT give treatment advice.\n"
"- Clearly mention that this is based on similar images, not a final result.\n"
"- If medical terms are used (like cardiomegaly), explain them in brackets.\n\n"

"Output format:\n\n"

"1) Summary (5 short sentences):\n"
"- Explain what the X-ray looks similar to.\n"
"- Mention general areas (like heart or lungs).\n\n"

"2) Label-wise explanation:( in detail very clearly like explaining to non medical people)\n"
"- Use bullet points.\n"
"- Each point should:\n"
"  • Name the finding\n"
"  • Explain it in one simple sentence\n"
"  • Add a short meaning in brackets if needed\n\n"

"Example style:\n"
"- Cardiomegaly: The heart looks slightly larger than usual (this can sometimes happen due to strain on the heart).\n\n"

"3) Safety note (1 sentence):\n"
"- Clearly say this is NOT a diagnosis and a doctor should confirm.\n\n"

"Tone:\n"
"- Friendly\n"
"- Simple\n"
"- Reassuring\n"
"- No technical complexity\n"

)

    user_prompt = f"Retrieved context:\n{context}"

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        temperature=0.2,
    )
    return response.output_text.strip()


def generate_explanation(
    context: Dict[str, Any],
    use_openai_if_available: bool = True,
    openai_model: str = "gpt-4o-mini",
) -> str:
    """Generate final patient-friendly explanation with safe fallback behavior."""
    key_available = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai_if_available and key_available:
        try:
            return _openai_explanation(context=context, model=openai_model)
        except Exception:
            return _placeholder_explanation(context)
    return _placeholder_explanation(context)
