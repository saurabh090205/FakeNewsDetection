import hashlib
import time
from collections import OrderedDict
from fastapi import APIRouter, HTTPException

from app.core.config import DATASET_STATS_PATH, METRICS_DIR, MAPPING_PATH, MINIMUM_ARTICLE_WORDS, MODEL_NAMES, read_json
from app.ml.predictor import PredictionError, predict_news, predict_texts, validate_text_inputs
from app.core.config import UNCERTAINTY_MARGIN
from app.preprocessing.text import combine_news
from app.schemas import PredictRequest
from app.services.artifacts import model_inventory

router = APIRouter(prefix="/api")

# Bounded in-memory LRU cache for LIME explanations (max 64 items)
_EXPLANATION_CACHE: OrderedDict[str, dict] = OrderedDict()
_MAX_CACHE_SIZE = 64


def _get_cached_explanation(key: str) -> dict | None:
    if key in _EXPLANATION_CACHE:
        _EXPLANATION_CACHE.move_to_end(key)
        return _EXPLANATION_CACHE[key]
    return None


def _set_cached_explanation(key: str, value: dict):
    _EXPLANATION_CACHE[key] = value
    if len(_EXPLANATION_CACHE) > _MAX_CACHE_SIZE:
        _EXPLANATION_CACHE.popitem(last=False)


@router.get("/health")
def health():
    inventory = model_inventory()
    return {
        "status": "ok",
        "platform": "Veritas AI — Fake News Detection & Explainability Platform",
        "available_models": sum(item["available"] for item in inventory),
        "models": inventory,
    }


@router.get("/models")
def models():
    return {"models": model_inventory(), "minimum_article_words": MINIMUM_ARTICLE_WORDS}


@router.get("/metrics")
def metrics():
    all_metrics = {name: read_json(METRICS_DIR / f"{name}.json") for name in MODEL_NAMES}
    comparison = read_json(METRICS_DIR / "comparison.json")
    experiments = read_json(METRICS_DIR / "experiment_curves.json")
    return {"models": {key: value for key, value in all_metrics.items() if value is not None}, "comparison": comparison, "experiments": experiments}


@router.post("/predict")
def predict(request: PredictRequest):
    try:
        return predict_news(request.headline, request.article, request.model)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc


@router.post("/compare")
def compare(request: PredictRequest):
    try:
        validate_text_inputs(request.headline, request.article)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc
    outcomes = []
    for item in model_inventory():
        if item["available"]:
            try:
                outcomes.append(predict_news(request.headline, request.article, item["name"]))
            except PredictionError as exc:
                outcomes.append({"model": item["name"], "error": str(exc), "code": exc.code})
        else:
            outcomes.append({"model": item["name"], "error": "Model not trained yet.", "code": "MODEL_UNAVAILABLE"})
    if not outcomes:
        raise HTTPException(status_code=503, detail={"error": "No trained models are available.", "code": "NO_MODELS_AVAILABLE"})
    return {"results": outcomes}


@router.get("/model/{name}/metrics")
def model_metrics(name: str):
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    result = read_json(METRICS_DIR / f"{name}.json")
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "No saved evaluation results for this model.", "code": "METRICS_UNAVAILABLE"})
    return result


@router.get("/model/{name}/confusion-matrix")
def confusion_matrix(name: str):
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    result = read_json(METRICS_DIR / f"{name}.json")
    if result is None or "confusion_matrix" not in result:
        raise HTTPException(status_code=404, detail={"error": "No saved confusion matrix for this model.", "code": "METRICS_UNAVAILABLE"})
    return {"model": name, "confusion_matrix": result["confusion_matrix"]}


@router.get("/dataset/statistics")
def dataset_statistics():
    result = read_json(DATASET_STATS_PATH)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": "Dataset statistics are not available until the dataset is prepared.", "code": "DATASET_UNAVAILABLE"})
    return result


@router.post("/explain")
def explain(request: PredictRequest):
    if request.model not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail={"error": "Choose a supported model.", "code": "INVALID_MODEL"})
    try:
        validate_text_inputs(request.headline, request.article)
        text = combine_news(request.headline, request.article)
        cache_key = hashlib.sha256(f"{request.model}:{text}".encode("utf-8")).hexdigest()
        cached = _get_cached_explanation(cache_key)
        if cached:
            result = dict(cached)
            result["cached"] = True
            return result

        from lime.lime_text import LimeTextExplainer
        label_mapping = read_json(MAPPING_PATH)
        if not label_mapping or "class_to_index" not in label_mapping:
            raise PredictionError("Label mapping is unavailable.", "MAPPING_UNAVAILABLE", 503)

        index_to_class = {int(index): label for label, index in label_mapping["class_to_index"].items()}
        class_names = [index_to_class[index] for index in range(len(index_to_class))]

        # Run real model prediction for target text
        start_time = time.perf_counter()
        initial_probs = predict_texts(request.model, [text])[0]
        class_index = int(initial_probs.argmax())
        predicted_class = index_to_class[class_index]
        confidence = float(initial_probs[class_index])

        explainer = LimeTextExplainer(
            class_names=class_names,
            random_state=42,
            split_expression=r"\W+",
            bow=True,
        )

        # Perturb and explain specifically for the predicted class
        exp = explainer.explain_instance(
            text,
            lambda samples: predict_texts(request.model, samples),
            labels=(class_index,),
            num_features=14,
            num_samples=160,
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 1)
        raw_words = exp.as_list(label=class_index)

        words_list = []
        pos_words = []
        neg_words = []
        for word, weight in raw_words:
            w_float = float(weight)
            item = {
                "word": str(word),
                "weight": round(w_float, 4),
                "direction": "supports" if w_float >= 0 else "opposes",
                "abs_weight": round(abs(w_float), 4),
            }
            words_list.append(item)
            if w_float >= 0:
                pos_words.append(item)
            else:
                neg_words.append(item)

        pos_words.sort(key=lambda x: x["abs_weight"], reverse=True)
        neg_words.sort(key=lambda x: x["abs_weight"], reverse=True)

        # Build informative, plain-English summary
        top_pos_str = ", ".join(f"'{item['word']}'" for item in pos_words[:4]) or "none identified"
        top_neg_str = ", ".join(f"'{item['word']}'" for item in neg_words[:3]) or "none identified"
        plain_summary = (
            f"The model predicted {predicted_class} ({confidence * 100:.1f}% confidence). "
            f"Linguistic features that most strongly supported this prediction were: {top_pos_str}. "
            f"Features that opposed this prediction were: {top_neg_str}. "
            "These word weights represent local statistical importance within the trained model, not verified factual authenticity."
        )

        response_payload = {
            "model": request.model,
            "prediction": predicted_class,
            "prediction_label": f"Predicted {predicted_class.capitalize()}",
            "confidence": round(confidence, 4),
            "class_index": class_index,
            "class_names": class_names,
            "words": words_list,
            "positive_words": pos_words,
            "negative_words": neg_words,
            "plain_english_summary": plain_summary,
            "note": "Local surrogate explanation (LIME); influential words are approximate statistical cues and do not establish objective factual accuracy.",
            "methodology": "LIME perturbs the input text by selectively omitting words, evaluates the changes in model prediction scores, and fits an interpretable ridge regression surrogate locally around the instance.",
            "sample_count": 160,
            "cached": False,
            "execution_time_ms": duration_ms,
        }

        _set_cached_explanation(cache_key, response_payload)
        return response_payload

    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code}) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail={"error": "The LIME explanation dependency is not installed.", "code": "EXPLAINER_UNAVAILABLE"}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"LIME explanation generation failed: {str(exc)}", "code": "EXPLANATION_ERROR"}) from exc

