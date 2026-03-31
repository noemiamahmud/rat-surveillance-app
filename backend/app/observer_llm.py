from __future__ import annotations

from typing import Any

from .config import settings


DRUG_CONDITION_CONTEXT = {
    "meth": "Methamphetamine conditions typically increase locomotor bursts, stereotypy, and reduce resting. Flag stereotypy patterns as expected but clinically relevant.",
    "methamphetamine": "Methamphetamine conditions typically increase locomotor bursts, stereotypy, and reduce resting. Flag stereotypy patterns as expected but clinically relevant.",
    "alcohol": "Alcohol conditions may increase hesitation, resting, and reduce exploration. Ataxic movement patterns and freezing are of interest.",
    "ethanol": "Ethanol conditions may increase hesitation, resting, and reduce exploration. Ataxic movement patterns and freezing are of interest.",
    "amphetamine": "Amphetamine conditions increase locomotor activity and may produce stereotypy at higher doses. Compare exploration and stereotypy ratios to baseline.",
    "cocaine": "Cocaine conditions produce acute locomotor activation followed by stereotypy. Monitor for rapid behavioral state transitions.",
    "baseline": "This is a baseline/control session. Behavior distribution serves as the reference for drug condition comparisons.",
    "saline": "This is a saline control session. Behavior distribution serves as the reference for drug condition comparisons.",
}


def llm_available() -> bool:
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY.strip())
    if provider == "llama_cpp":
        return bool(settings.LLM_BASE_URL.strip())
    return False


def llm_descriptor() -> str:
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "openai":
        return settings.OPENAI_MODEL
    if provider == "llama_cpp":
        return f"{settings.LLAMA_CPP_MODEL} via llama.cpp"
    return "None"


def _build_llm_client():
    try:
        from openai import OpenAI
    except Exception:
        return None, None

    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "openai":
        return OpenAI(api_key=settings.OPENAI_API_KEY), settings.OPENAI_MODEL
    elif provider == "llama_cpp":
        return OpenAI(
            api_key="local-llama",
            base_url=settings.LLM_BASE_URL.rstrip("/") + "/v1",
        ), settings.LLAMA_CPP_MODEL
    return None, None


def summarize_with_llm(
    *,
    dominant_behavior: str,
    review_priority: str,
    bouts: list[dict[str, Any]],
    behavior_share: dict[str, float],
    pipeline_mode: str,
    condition: str | None = None,
) -> str | None:
    if not llm_available():
        return None

    client, model = _build_llm_client()
    if client is None:
        return None

    top_bouts = [
        {
            "label": bout["label"],
            "start_s": round(float(bout["start_s"]), 2),
            "end_s": round(float(bout["end_s"]), 2),
            "confidence": round(float(bout["confidence"]), 2),
        }
        for bout in bouts[:8]
    ]

    condition_context = ""
    if condition:
        condition_key = condition.strip().lower()
        condition_context = DRUG_CONDITION_CONTEXT.get(condition_key, f"Experimental condition: {condition}.")
        condition_context = f" Condition context: {condition_context}"

    system_prompt = (
        "You are an expert rodent behavior analyst summarizing a rat behavior analysis session "
        "from a drug self-administration experiment. Be precise about behavioral states and their "
        "implications for the experimental condition. Write 2-3 concise sentences."
    )
    user_prompt = (
        f"Pipeline mode: {pipeline_mode}. "
        f"Dominant behavior: {dominant_behavior}. "
        f"Review priority: {review_priority}. "
        f"Behavior share: {behavior_share}. "
        f"Top bouts: {top_bouts}."
        f"{condition_context}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.4,
        )
        text = response.choices[0].message.content
        return (text or "").strip() or None
    except Exception:
        return None
