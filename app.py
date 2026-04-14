import json
import re
import time
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.ensemble import IsolationForest
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="AutoEDA Chatbot", layout="wide")

MODEL_ID = "google/flan-t5-small"
MAX_ROWS_FOR_PREVIEW = 200
DEFAULT_CORR_THRESHOLD = 0.6
CHAT_CONTEXT_CHAR_LIMIT = 1500


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    df = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except Exception:
            df = None

    if df is None:
        raise ValueError("Unable to read the CSV file. Please upload a standard CSV.")

    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) > 0 and (
                "date" in col.lower()
                or "time" in col.lower()
                or sample.str.contains(
                    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}",
                    regex=True
                ).mean() > 0.5
            ):
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notna().mean() > 0.6:
                        df[col] = converted
                except Exception:
                    pass

    return df


@st.cache_resource(show_spinner=False)
def load_model(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def generate_text(prompt: str, max_new_tokens: int = 90) -> str:
    tokenizer, model, device = load_model()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.08,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def get_profile(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols
    }


def get_datetime_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    for col in datetime_cols:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            summary[col] = {
                "min": str(non_null.min()),
                "max": str(non_null.max()),
                "unique_values": int(non_null.nunique())
            }

    return summary


def get_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    missing_counts = df.isnull().sum().to_dict()
    missing_percent = ((df.isnull().mean()) * 100).round(2).to_dict()

    return {
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "missing_percent": {k: float(v) for k, v in missing_percent.items()},
        "duplicate_rows": int(df.duplicated().sum())
    }


def get_correlations(df: pd.DataFrame, corr_threshold: float) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return {"correlation_matrix": {}, "strong_pairs": []}

    corr_matrix = numeric_df.corr(numeric_only=True).round(3)
    strong_pairs = []

    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_value = corr_matrix.iloc[i, j]
            if pd.notna(corr_value) and abs(corr_value) >= corr_threshold:
                strong_pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": float(round(corr_value, 3))
                })

    strong_pairs = sorted(strong_pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "strong_pairs": strong_pairs
    }


def get_iqr_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    outlier_summary = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            lower_bound = q1
            upper_bound = q3
            outlier_count = 0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        outlier_summary[col] = {
            "outlier_count": int(outlier_count),
            "outlier_percent": float(round((outlier_count / len(series)) * 100, 2)),
            "lower_bound": float(round(lower_bound, 3)) if pd.notna(lower_bound) else None,
            "upper_bound": float(round(upper_bound, 3)) if pd.notna(upper_bound) else None
        }

    return outlier_summary


def get_isolation_forest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] == 0 or len(numeric_df) < 10:
        return {
            "num_anomalies": 0,
            "anomaly_percent": 0.0,
            "status": "Not enough numeric data for Isolation Forest."
        }

    filled = numeric_df.fillna(numeric_df.median(numeric_only=True))

    try:
        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42
        )
        preds = model.fit_predict(filled)
        anomaly_count = int((preds == -1).sum())

        return {
            "num_anomalies": anomaly_count,
            "anomaly_percent": float(round((anomaly_count / len(filled)) * 100, 2)),
            "status": "Success"
        }
    except Exception as e:
        return {
            "num_anomalies": 0,
            "anomaly_percent": 0.0,
            "status": f"Isolation Forest failed: {str(e)}"
        }


def get_feature_engineering_suggestions(df: pd.DataFrame) -> List[str]:
    suggestions = []
    profile = get_profile(df)
    dq = get_data_quality(df)

    if profile["datetime_columns"]:
        suggestions.append("Extract date parts such as year, month, weekday, or quarter from datetime columns.")

    high_missing = [col for col, pct in dq["missing_percent"].items() if pct > 20]
    if high_missing:
        suggestions.append(f"Consider imputing, flagging, or dropping high-missingness columns: {', '.join(high_missing)}.")

    if len(profile["numeric_columns"]) >= 2:
        suggestions.append("Consider scaling numeric variables if you plan to use distance-based or gradient-based models.")

    if profile["categorical_columns"]:
        suggestions.append("Categorical columns may need one-hot encoding, frequency encoding, or target-aware encoding for downstream modeling.")

    if profile["numeric_columns"]:
        suggestions.append("Investigate skewed numeric features for possible log transformations.")

    return suggestions


def build_eda_summary(df: pd.DataFrame, corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> Dict[str, Any]:
    total_start = time.perf_counter()

    start = time.perf_counter()
    profile = get_profile(df)
    profile_time = time.perf_counter() - start

    start = time.perf_counter()
    datetime_summary = get_datetime_summary(df)
    datetime_time = time.perf_counter() - start

    start = time.perf_counter()
    data_quality = get_data_quality(df)
    quality_time = time.perf_counter() - start

    start = time.perf_counter()
    correlations = get_correlations(df, corr_threshold)
    corr_time = time.perf_counter() - start

    start = time.perf_counter()
    outliers_iqr = get_iqr_outliers(df)
    iqr_time = time.perf_counter() - start

    start = time.perf_counter()
    anomalies_isolation_forest = get_isolation_forest_summary(df)
    iso_time = time.perf_counter() - start

    start = time.perf_counter()
    feature_engineering_suggestions = get_feature_engineering_suggestions(df)
    feat_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    return {
        "profile": profile,
        "datetime_summary": datetime_summary,
        "data_quality": data_quality,
        "correlations": correlations,
        "outliers_iqr": outliers_iqr,
        "anomalies_isolation_forest": anomalies_isolation_forest,
        "feature_engineering_suggestions": feature_engineering_suggestions,
        "performance_metrics": {
            "profile_seconds": round(profile_time, 3),
            "datetime_seconds": round(datetime_time, 3),
            "data_quality_seconds": round(quality_time, 3),
            "correlation_seconds": round(corr_time, 3),
            "iqr_seconds": round(iqr_time, 3),
            "isolation_forest_seconds": round(iso_time, 3),
            "feature_suggestion_seconds": round(feat_time, 3),
            "total_eda_seconds": round(total_time, 3)
        }
    }


def plot_missing_values(df: pd.DataFrame):
    missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_percent = missing_percent[missing_percent > 0]

    if missing_percent.empty:
        st.info("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    missing_percent.plot(kind="bar", ax=ax)
    ax.set_ylabel("Missing Percentage")
    ax.set_title("Missing Values by Column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    series = df[col].dropna()
    if series.empty:
        st.info("This column has no non-null values to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series, bins=30)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def generate_rule_based_report(eda_summary: Dict[str, Any]) -> str:
    profile = eda_summary["profile"]
    dq = eda_summary["data_quality"]
    corr = eda_summary["correlations"]
    iso = eda_summary["anomalies_isolation_forest"]
    suggestions = eda_summary["feature_engineering_suggestions"]

    missing_cols = [col for col, cnt in dq["missing_counts"].items() if cnt > 0]
    strong_pairs = corr["strong_pairs"][:5]

    lines = []
    lines.append(f"This dataset contains {profile['num_rows']} rows and {profile['num_columns']} columns.")
    lines.append(f"Numeric columns: {len(profile['numeric_columns'])}.")
    lines.append(f"Categorical columns: {len(profile['categorical_columns'])}.")
    lines.append(f"Datetime columns: {len(profile['datetime_columns'])}.")
    lines.append(f"Duplicate rows: {dq['duplicate_rows']}.")

    if missing_cols:
        lines.append(f"Missing values were found in {len(missing_cols)} columns: {', '.join(missing_cols[:8])}.")
    else:
        lines.append("No missing values were found.")

    if strong_pairs:
        pair_text = "; ".join([f"{p['feature_1']} and {p['feature_2']} ({p['correlation']})" for p in strong_pairs])
        lines.append(f"Strong correlations include: {pair_text}.")
    else:
        lines.append("No strong correlations were detected using the current threshold.")

    lines.append(
        f"Isolation Forest flagged {iso['num_anomalies']} potential anomalies ({iso['anomaly_percent']}% of rows)."
    )

    if suggestions:
        lines.append("Feature engineering suggestions:")
        lines.extend([f"- {s}" for s in suggestions])

    return "\n".join(lines)


def top_missing_columns(eda_summary: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    dq = eda_summary["data_quality"]
    rows = []
    for col, cnt in dq["missing_counts"].items():
        if cnt > 0:
            rows.append({
                "column": col,
                "missing_count": cnt,
                "missing_percent": dq["missing_percent"][col]
            })
    rows = sorted(rows, key=lambda x: x["missing_percent"], reverse=True)
    return rows[:limit]


def top_outlier_columns(eda_summary: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    outliers = []
    for col, stats in eda_summary["outliers_iqr"].items():
        outliers.append({
            "column": col,
            "outlier_count": stats.get("outlier_count", 0),
            "outlier_percent": stats.get("outlier_percent", 0.0)
        })
    outliers = sorted(outliers, key=lambda x: x["outlier_count"], reverse=True)
    return outliers[:limit]


def build_chat_context(eda_summary: Dict[str, Any]) -> str:
    profile = eda_summary["profile"]
    dq = eda_summary["data_quality"]
    corr = eda_summary["correlations"]
    iso = eda_summary["anomalies_isolation_forest"]

    context = {
        "rows": profile["num_rows"],
        "columns": profile["num_columns"],
        "numeric_columns": profile["numeric_columns"][:10],
        "categorical_columns": profile["categorical_columns"][:10],
        "datetime_columns": profile["datetime_columns"][:10],
        "duplicate_rows": dq["duplicate_rows"],
        "top_missing_columns": top_missing_columns(eda_summary, limit=5),
        "top_correlations": corr["strong_pairs"][:5],
        "anomalies": iso,
        "top_outlier_columns": top_outlier_columns(eda_summary, limit=5),
        "feature_engineering_suggestions": eda_summary["feature_engineering_suggestions"][:4]
    }

    text = json.dumps(context, indent=2, default=str)
    return text[:CHAT_CONTEXT_CHAR_LIMIT]


def is_shape_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in [
        "how big", "shape", "how many rows", "how many columns", "size of the dataset",
        "dataset size", "dimensions"
    ])


def is_missing_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ["missing", "null", "na", "empty values", "blank values"])


def is_duplicate_question(question: str) -> bool:
    q = question.lower()
    return "duplicate" in q


def is_correlation_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ["correlation", "correlated", "relationship", "related columns"])


def is_anomaly_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ["anomaly", "anomalies", "outlier", "outliers", "unusual"])


def is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in [
        "summary", "summarize", "overview", "describe the dataset", "main findings",
        "main issues", "biggest issues", "what stands out", "insights",
        "recommendations", "what should i do next", "key problems",
        "biggest risks", "main patterns", "important findings"
    ])


def is_feature_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ["feature engineering", "features", "transform", "preprocess", "prepare the data"])


def is_column_type_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ["numeric columns", "categorical columns", "column types", "data types", "dtypes"])


def answer_rule_based(question: str, eda_summary: Dict[str, Any]) -> str:
    profile = eda_summary["profile"]
    dq = eda_summary["data_quality"]
    corr = eda_summary["correlations"]
    iso = eda_summary["anomalies_isolation_forest"]
    suggestions = eda_summary["feature_engineering_suggestions"]

    if is_shape_question(question):
        return (
            f"The dataset has {profile['num_rows']} rows and {profile['num_columns']} columns. "
            f"It includes {len(profile['numeric_columns'])} numeric columns, "
            f"{len(profile['categorical_columns'])} categorical columns, and "
            f"{len(profile['datetime_columns'])} datetime columns."
        )

    if is_missing_question(question):
        missing = top_missing_columns(eda_summary, limit=10)
        if not missing:
            return "I do not see any missing values in this dataset."
        lines = [f"- {m['column']}: {m['missing_count']} missing values ({m['missing_percent']}%)" for m in missing]
        return "Here are the columns with missing values:\n" + "\n".join(lines)

    if is_duplicate_question(question):
        return f"I found {dq['duplicate_rows']} duplicate rows in the dataset."

    if is_correlation_question(question):
        strong_pairs = corr["strong_pairs"][:8]
        if not strong_pairs:
            return "I did not find any strong correlations using the current threshold."
        lines = [f"- {p['feature_1']} and {p['feature_2']}: correlation {p['correlation']}" for p in strong_pairs]
        return "These are the strongest correlations I found:\n" + "\n".join(lines)

    if is_anomaly_question(question):
        outliers = top_outlier_columns(eda_summary, limit=5)
        lines = [f"- {o['column']}: {o['outlier_count']} outliers ({o['outlier_percent']}%)" for o in outliers if o["outlier_count"] > 0]
        summary = (
            f"Isolation Forest flagged {iso['num_anomalies']} potential anomalies "
            f"({iso['anomaly_percent']}% of rows)."
        )
        if lines:
            return summary + "\nTop columns with IQR-based outliers:\n" + "\n".join(lines)
        return summary + "\nI did not see notable IQR outlier counts in numeric columns."

    if is_feature_question(question):
        if suggestions:
            return "Here are the main feature engineering ideas I suggest:\n" + "\n".join([f"- {s}" for s in suggestions])
        return "I do not have any strong feature engineering suggestions for this dataset."

    if is_column_type_question(question):
        return (
            f"Numeric columns: {', '.join(profile['numeric_columns']) if profile['numeric_columns'] else 'None'}.\n\n"
            f"Categorical columns: {', '.join(profile['categorical_columns']) if profile['categorical_columns'] else 'None'}.\n\n"
            f"Datetime columns: {', '.join(profile['datetime_columns']) if profile['datetime_columns'] else 'None'}."
        )

    return ""


def llm_chat_answer(question: str, eda_summary: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    context = build_chat_context(eda_summary)

    recent = []
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        recent.append(f"{role}: {msg['content']}")
    history_text = "\n".join(recent)

    prompt = f"""
You are a chatbot for exploratory data analysis.
You are talking to a user about their uploaded CSV dataset.

Rules:
- Answer like a helpful chatbot, not like raw JSON.
- Use only the EDA context below.
- Do not invent facts.
- If the context does not support something, say so clearly.
- Prefer plain English.
- Keep the answer focused and useful.

EDA CONTEXT:
{context}

RECENT CHAT HISTORY:
{history_text}

USER QUESTION:
{question}

CHATBOT ANSWER:
""".strip()

    answer = generate_text(prompt, max_new_tokens=170).strip()

    bad_patterns = [
        r"lower_bound\s*:",
        r"upper_bound\s*:",
        r"feature_1\s*:",
        r"feature_2\s*:",
        r"correlation\s*:",
        r"outlier_count\s*:"
    ]
    if any(re.search(p, answer.lower()) for p in bad_patterns):
        return ""

    return answer


def answer_question(question: str, eda_summary: Dict[str, Any], use_llm: bool, chat_history: List[Dict[str, str]]) -> str:
    q = question.lower().strip()
    rule_answer = answer_rule_based(question, eda_summary)

    exact_fact_question = any(p in q for p in [
        "how big", "shape", "how many rows", "how many columns", "dataset size",
        "missing", "null", "duplicate", "correlation", "correlated",
        "anomaly", "anomalies", "outlier", "outliers", "column types", "data types", "dtypes"
    ])

    interpretive_question = any(p in q for p in [
        "main issues", "biggest issues", "what stands out", "main findings",
        "summary", "summarize", "overview", "insights", "recommendations",
        "what should i do next", "biggest risks", "important findings", "key problems"
    ])

    if rule_answer and exact_fact_question:
        return rule_answer

    if interpretive_question:
        if use_llm:
            context = build_chat_context(eda_summary)
            prompt = f"""
You are an EDA chatbot.

Use only the context below.
Answer briefly and clearly in plain English.
Do not invent facts.
Do not change numeric values.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

            try:
                llm_answer = generate_text(prompt, max_new_tokens=90).strip()
                bad_patterns = [
                    "lower_bound:",
                    "upper_bound:",
                    "feature_1:",
                    "feature_2:",
                    "outlier_count:",
                    "missing_count:"
                ]
                if llm_answer and not any(p in llm_answer.lower() for p in bad_patterns):
                    return llm_answer
            except Exception:
                pass

        return generate_rule_based_report(eda_summary)

    if rule_answer:
        return rule_answer

    if use_llm:
        try:
            llm_answer = llm_chat_answer(question, eda_summary, chat_history)
            if llm_answer:
                return llm_answer
        except Exception:
            pass

    return (
        "I can help answer questions about dataset size, missing values, duplicates, correlations, "
        "anomalies, column types, summaries, and feature engineering ideas. "
        "Try asking something like 'What are the main issues?' or 'Are there any strong correlations?'"
    )

st.title("AutoEDA Chatbot")
st.caption("A chatbot-first EDA assistant for uploaded CSV datasets.")

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    corr_threshold = st.slider(
        "Strong correlation threshold",
        0.3,
        0.95,
        DEFAULT_CORR_THRESHOLD,
        0.05
    )
    use_llm_for_chat = st.checkbox("Use generative model for open-ended responses", value=False)
    use_llm_for_report = st.checkbox("Use LLM for narrative report", value=False)
    st.write("Model:", MODEL_ID)
    st.caption("Correlation threshold controls which numeric column pairs count as strongly related.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    eda_summary = build_eda_summary(df, corr_threshold=corr_threshold)

    top_row = st.columns([1, 1, 1, 1])
    top_row[0].metric("Rows", eda_summary["profile"]["num_rows"])
    top_row[1].metric("Columns", eda_summary["profile"]["num_columns"])
    top_row[2].metric("Duplicate Rows", eda_summary["data_quality"]["duplicate_rows"])
    top_row[3].metric("EDA Runtime (s)", eda_summary["performance_metrics"]["total_eda_seconds"])

    main_tab, profile_tab, quality_tab, corr_tab, anomaly_tab, chart_tab = st.tabs([
        "Chatbot",
        "Profile",
        "Data Quality",
        "Correlations",
        "Anomalies",
        "Charts"
    ])

    with main_tab:
        st.subheader("Talk to your dataset")
        st.write("Suggested prompts:")
        st.markdown("""
- Summarize this dataset.
- How big is the dataset?
- Which columns have missing values?
- Are there any strong correlations?
- What anomalies stand out?
- What feature engineering ideas do you suggest?
- What are the column types?
""")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Clear chat history", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            if st.button("Ask for automatic summary", use_container_width=True):
                starter = "Summarize this dataset."
                st.session_state.chat_history.append({"role": "user", "content": starter})
                answer = answer_question(
                    starter,
                    eda_summary,
                    use_llm=use_llm_for_chat,
                    chat_history=st.session_state.chat_history
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.write(
                    "Hi — I’m your AutoEDA chatbot. Upload a CSV and ask me questions about "
                    "missing values, correlations, anomalies, dataset size, and feature engineering ideas."
                )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_prompt = st.chat_input("Ask a question about the dataset")

        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.write(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        user_prompt,
                        eda_summary,
                        use_llm=use_llm_for_chat,
                        chat_history=st.session_state.chat_history
                    )
                    st.write(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(MAX_ROWS_FOR_PREVIEW), use_container_width=True)

        st.subheader("Narrative Report")
        if st.button("Generate dataset report"):
            with st.spinner("Generating report..."):
                report = generate_rule_based_report(eda_summary)
                if use_llm_for_report:
                    prompt = f"""
Rewrite the following EDA report in a polished, student-friendly way.
Do not add facts.

REPORT:
{report}

REWRITTEN REPORT:
""".strip()
                    report = generate_text(prompt, max_new_tokens=120)
            st.text_area("Dataset Report", report, height=260)
            st.download_button(
                "Download report as text",
                data=report,
                file_name="autoeda_report.txt",
                mime="text/plain"
            )

        st.subheader("Download Results")
        st.download_button(
            "Download EDA summary as JSON",
            data=json.dumps(eda_summary, indent=2, default=str),
            file_name="eda_summary.json",
            mime="application/json"
        )

    with profile_tab:
        profile = eda_summary["profile"]

        st.subheader("Dataset overview")
        st.write(f"Rows: **{profile['num_rows']}**")
        st.write(f"Columns: **{profile['num_columns']}**")

        dtype_df = pd.DataFrame(
            list(profile["dtypes"].items()),
            columns=["Column", "Data Type"]
        )
        st.subheader("Column types")
        st.dataframe(dtype_df, use_container_width=True)

        a, b, c = st.columns(3)
        with a:
            st.subheader("Numeric")
            st.write(", ".join(profile["numeric_columns"]) if profile["numeric_columns"] else "None")
        with b:
            st.subheader("Categorical")
            st.write(", ".join(profile["categorical_columns"]) if profile["categorical_columns"] else "None")
        with c:
            st.subheader("Datetime")
            st.write(", ".join(profile["datetime_columns"]) if profile["datetime_columns"] else "None")

        if eda_summary["datetime_summary"]:
            st.subheader("Datetime summary")
            st.json(eda_summary["datetime_summary"])

    with quality_tab:
        dq = eda_summary["data_quality"]

        st.subheader("Missing values")
        missing_df = pd.DataFrame({
            "Column": list(dq["missing_counts"].keys()),
            "Missing Count": list(dq["missing_counts"].values()),
            "Missing Percent": list(dq["missing_percent"].values())
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0]

        if not missing_df.empty:
            st.dataframe(
                missing_df.sort_values("Missing Percent", ascending=False),
                use_container_width=True
            )
        else:
            st.success("No missing values found.")

        st.subheader("Duplicate rows")
        st.write(f"Duplicate rows detected: **{dq['duplicate_rows']}**")

        st.subheader("Missing value chart")
        plot_missing_values(df)

    with corr_tab:
        st.subheader("Strong correlations")
        strong_pairs = eda_summary["correlations"]["strong_pairs"]

        if strong_pairs:
            st.dataframe(pd.DataFrame(strong_pairs), use_container_width=True)
        else:
            st.info("No strong correlations found with the current threshold.")

    with anomaly_tab:
        st.subheader("Isolation Forest summary")
        iso = eda_summary["anomalies_isolation_forest"]
        st.write(f"Potential anomalies detected: **{iso['num_anomalies']}**")
        st.write(f"Percentage of rows flagged: **{iso['anomaly_percent']}%**")
        st.write(f"Status: **{iso['status']}**")

        st.subheader("IQR outlier summary")
        outlier_df = pd.DataFrame.from_dict(
            eda_summary["outliers_iqr"],
            orient="index"
        ).reset_index().rename(columns={"index": "Column"})

        if not outlier_df.empty:
            st.dataframe(
                outlier_df.sort_values("outlier_count", ascending=False),
                use_container_width=True
            )
        else:
            st.info("No numeric columns available for outlier analysis.")

    with chart_tab:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("Numeric distribution explorer")
        if numeric_cols:
            selected_col = st.selectbox("Choose a numeric column", numeric_cols)
            plot_numeric_distribution(df, selected_col)
        else:
            st.info("No numeric columns available for charting.")

        st.subheader("Feature engineering suggestions")
        suggestions = eda_summary["feature_engineering_suggestions"]
        if suggestions:
            for item in suggestions:
                st.write(f"- {item}")
        else:
            st.write("No feature engineering suggestions were generated.")

else:
    st.info("Upload a CSV file from the sidebar to begin.")
