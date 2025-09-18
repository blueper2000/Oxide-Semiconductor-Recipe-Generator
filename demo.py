from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
from io import BytesIO

import streamlit as st
import os
from experiment.predict import RAGRecipePredictor, RecipePredictor
from litellm import embedding
import litellm
from pdf2recipe import pdf_bytelist_to_recipes
from litellm import completion

st.set_page_config(
    page_title="OxideSC Materials Synthesis Recipe Recommender",
    page_icon=":microscope:",
    layout="wide",
)

st.title("OxideSC Materials Synthesis Recipe Recommender")
st.sidebar.title("Input Parameters")

# --- API KEY ---
openai_api_key = st.sidebar.text_input("OpenAI API Key (필수 입력)", "", type="password")
if not openai_api_key:
    st.sidebar.error("OpenAI API Key를 반드시 입력해야 합니다.")
    st.stop()

update_key = st.sidebar.button("Update Key")
if update_key:
    st.session_state.openai_key = openai_api_key
    litellm.openai_key = openai_api_key
    litellm.api_key = openai_api_key
    st.toast("API Key updated successfully")
    st.rerun()

# --- Sidebar form ---
with st.sidebar, st.form("recipe_form"):
    material_name = st.text_input("Deposited materials", "In2O3")
    synthesis_technique = st.text_input("Key Deposition/Process Method", "ALD")

    # ✅ 분리: 이동도 / 안정성
    mobility_target  = st.text_input("Target Mobility",  "> 20 cm²/V·s")
    stability_target = st.text_input("Target Stability", "< 200 mV")

    other_contstraints = st.text_area("Other Constraints", "")

    model = st.selectbox(
        "Model",
        [
            "gpt-5-mini",
            "gpt-5-high",
            "gpt-5",
            "gpt-5-low",
            "o4-mini-high",
            "o4-mini",
            "o4-mini-low",
        ],
    )

    top_k = st.slider("Number of Retrievals", 0, 10, 5)

    files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    if files:
        for file in files:
            st.write(file.name)

    generate_btn = st.form_submit_button("Recommend")

clear_btn = st.sidebar.button("Clear Conversation")
if clear_btn:
    st.session_state.pop("messages", None)
    st.session_state.pop("references", None)
    st.rerun()

if not generate_btn and not hasattr(st.session_state, "messages"):
    st.write(
        "This is a demo of the Materials Synthesis Recipe Recommender. "
        "Please enter the desired material properties and click on the 'Recommend' button "
        "to get a list of materials synthesis recipes that can be used to synthesize materials with the desired properties."
    )
    st.stop()

use_rag = top_k >= 1
output_filename = "data/recipes.jsonl"

# ---------------- Model helpers ----------------
def get_model_config(model_name):
    """모델별 설정을 반환하는 통합 함수"""
    if model_name.startswith("gpt-5"):
        base_model = "gpt-5" if model_name != "gpt-5-mini" else "gpt-5-mini"
        reasoning_effort = None
        if "high" in model_name:
            reasoning_effort = "high"
        elif "low" in model_name:
            reasoning_effort = "low"
        return base_model, reasoning_effort, True  # max_completion_tokens 사용

    elif model_name.startswith("o4"):
        base_model = "o4-mini"
        reasoning_effort = None
        if "high" in model_name:
            reasoning_effort = "high"
        elif "low" in model_name:
            reasoning_effort = "low"
        return base_model, reasoning_effort, True

    elif model_name.startswith("o1") or model_name.startswith("o3"):
        base_model = model_name.replace("-high", "").replace("-low", "")
        reasoning_effort = "high" if "high" in model_name else "low" if "low" in model_name else None
        return base_model, reasoning_effort, True

    else:
        return model_name, None, False

def call_completion_model(messages, model_name):
    """통합된 모델 호출 함수"""
    base_model, reasoning_effort, use_completion_tokens = get_model_config(model_name)

    if use_completion_tokens:
        kwargs = {
            "model": base_model,
            "messages": messages,
            "max_completion_tokens": 16384,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs = {
            "model": base_model,
            "messages": messages,
            "max_tokens": 4096,
        }

    return completion(**kwargs)

# ---------------- JSON → Tables utils ----------------
def _json_to_tables(json_text: str):
    """
    LLM이 반환한 JSON(또는 JSON이 포함된 텍스트)을 받아
    scorecard(dict)와 {섹션명: DataFrame} 딕셔너리를 리턴.
    유효하지 않으면 (None, {}) 반환.
    """
    if not isinstance(json_text, str) or not json_text.strip():
        return None, {}

    raw = json_text.strip()

    # 1) 코드펜스 제거
    if "```" in raw:
        try:
            fence_blocks = raw.split("```")
            candidate = None
            for i in range(1, len(fence_blocks), 2):
                block = fence_blocks[i + 1] if (i + 1) < len(fence_blocks) else ""
                if "{" in block and "}" in block:
                    candidate = block
                    break
            if candidate:
                raw = candidate.strip()
        except Exception:
            pass

    # 2) 첫 '{' ~ 마지막 '}' 추출
    if "{" in raw and "}" in raw:
        try:
            start = raw.index("{")
            end = raw.rindex("}")
            raw = raw[start : end + 1]
        except Exception:
            pass

    # 3) JSON 로드
    try:
        data = json.loads(raw)
    except Exception:
        return None, {}

    if not isinstance(data, dict):
        return None, {}

    score = data.get("scorecard", None)
    tables = data.get("tables", {})

    dfs = {}
    if isinstance(tables, dict):
        for section, rows in tables.items():
            if isinstance(rows, list):
                try:
                    dfs[section] = pd.DataFrame(rows)
                except Exception:
                    pass

    return score, dfs

def _render_json_tables_block(
    text: str,
    base_name: str = "prediction_tables",
    expanded: bool = True,
    title: str = "📦 Parse JSON to tables",
):
    """어떤 텍스트든 JSON 테이블을 찾아 렌더링 + 단일 CSV/Excel 다운로드 제공"""
    with st.expander(title, expanded=expanded):
        score, dfs = _json_to_tables(text)
        if not dfs:
            st.info("응답에서 유효한 JSON 테이블 구조를 찾지 못했습니다. 템플릿 준수 여부를 확인하세요.")
            return

        if score:
            _render_scorecard(score)
            st.markdown("---")

        show_details = st.checkbox("🔎 Show rationale & Ref IDs", value=False)

        st.subheader("Tables")
        for section, df in dfs.items():
            st.markdown(f"**{section}**")
            df_show = df if show_details else df.drop(columns=["Rationale", "Ref IDs"], errors="ignore")
            st.dataframe(df_show, use_container_width=True)

        all_df_full = _concat_tables(dfs)
        all_df_simple = all_df_full.drop(columns=["Rationale", "Ref IDs"], errors="ignore")
        _offer_downloads_unified(all_df_simple, all_df_full, base_name=base_name)

def _render_scorecard(score: dict):
    """Streamlit에 scorecard를 예쁘게 렌더링."""
    if not isinstance(score, dict):
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence (0–5)", f"{score.get('confidence', 'N/A')}")
    with c2:
        st.metric("Similar refs", f"{score.get('similar_refs', 'N/A')}")
    
    st.write("**Why**:", score.get("why", ""))
    risks = score.get("risks", [])
    if isinstance(risks, list) and len(risks) == 2:
        st.write(f"**Risks:** 1) {risks[0]}  2) {risks[1]}")

def _concat_tables(dfs: dict) -> pd.DataFrame:
    """섹션별 DataFrame(dict) → 단일 DataFrame(Section 컬럼 포함)로 병합"""
    rows = []
    for section, df in dfs.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "Section", section)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _offer_downloads_unified(
    all_df_simple: pd.DataFrame,
    all_df_full: pd.DataFrame,
    base_name: str = "prediction_tables",
):
    """단일 CSV/Excel 다운로드 제공 (simple / full)"""
    if all_df_full is None or all_df_full.empty:
        return

    st.subheader("Download")

    csv_simple = all_df_simple.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV — All tables (simple)",
        data=csv_simple,
        file_name=f"{base_name}__all_simple.csv",
        mime="text/csv",
        use_container_width=True,
    )

    csv_full = all_df_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV — All tables (with rationale & refs)",
        data=csv_full,
        file_name=f"{base_name}__all_full.csv",
        mime="text/csv",
        use_container_width=True,
    )

    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        writer.book.use_zip64()
        all_df_simple.to_excel(writer, index=False, sheet_name="tables_simple")
        all_df_full.to_excel(writer, index=False, sheet_name="tables_full")
    st.download_button(
        label="⬇️ Download Excel — All tables",
        data=xbuf.getvalue(),
        file_name=f"{base_name}__all.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ---------------- Prompt ----------------
PREDICTION_PROMPT = """## Key Contributions
- **Deposited material**: {material_name}
- **Key Deposition/Process Method**: {deposition_method}
- **Target Mobility**: {mobility_target}
- **Target Stability**: {stability_target}
""".strip()

def get_embedding(contributions):
    response = embedding(model="text-embedding-3-large", input=[contributions])
    emb = response["data"][0]["embedding"]
    return emb

@st.cache_resource
def get_predictors():
    # RAGRecipePredictor 내부에서 자동으로 iknow-lab/oxideSC-recipe_embeddings 데이터셋을 로드합니다
    rag_predictor = RAGRecipePredictor(
        model=model,
        prompt_filename="experiment/prompts/rag.txt",
        rag_topk=top_k,
        retrieval_split="all",
        api_key=openai_api_key,
    )
    base_predictor = RecipePredictor(
        model=model,
        prompt_filename="experiment/prompts/prediction.txt",
        api_key=openai_api_key,
    )
    return rag_predictor, base_predictor

rag_predictor, base_predictor = get_predictors()

# ---------------- Prediction ----------------
def predict_recipe(
    material_name,
    synthesis_technique,
    mobility_target,
    stability_target,
    other_contstraints,
    top_k,
    model,
    use_rag,
    files=None,
):
    # 개선된 prompt 형식 사용 - 변수명 일치
    contributions = PREDICTION_PROMPT.format(
        material_name=material_name,
        deposition_method=synthesis_technique,
        mobility_target=mobility_target,
        stability_target=stability_target,
    )

    if use_rag or files:
        predictor = rag_predictor
        emb = get_embedding(contributions)
    else:
        predictor = base_predictor
        emb = None

    if files:
        with st.spinner("Extracting recipes from PDFs..."):
            references = pdf_bytelist_to_recipes([file.read() for file in files])
    else:
        references = None

    predictor.base_references = references
    predictor.model = model

    if other_contstraints:
        contributions += f"\n\n## Other Constraints\n{other_contstraints}"

    batch = [
        {
            "contribution": contributions,
            "recipe": "",
            "contributions_embedding": emb,
        }
    ]

    # predictor.predict는 (idx, output) 제너레이터라고 가정
    output = ""
    for _, output in predictor.predict(batch):
        pass

    if use_rag or files:
        ref_outputs = []
        if references:
            ref_outputs.extend(references)

        ref_df = predictor.search(emb, k=top_k, return_rows=True)
        if ref_df is not None and len(ref_df) > 0:
            limit = min(top_k, len(ref_df))
            for i in range(limit):
                rid = ref_df["id"][i]
                contribution = ref_df["contribution"][i]
                recipe_txt = ref_df["recipe"][i]
                ref_output = f"Semantic Scholar: [{rid}](https://www.semanticscholar.org/paper/{rid})\n"
                ref_output += f"{contribution}\n\n{recipe_txt}"
                ref_outputs.append(ref_output)

        references = ref_outputs if ref_outputs else None
    else:
        references = None

    prompt = predictor.build_prompt(batch[0])[0]["content"]
    return output, references, prompt

# ---------------- Initial run ----------------
if not hasattr(st.session_state, "messages"):
    st.session_state.messages = []

    with st.spinner("Generating recipes..."):
        recipe, references, user_prompt = predict_recipe(
            material_name,
            synthesis_technique,
            mobility_target,
            stability_target,
            other_contstraints,
            top_k,
            model,
            use_rag,
            files=files,
        )
        st.session_state.references = references
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        st.session_state.messages.append({"role": "assistant", "content": recipe})
else:
    recipe = st.session_state.messages[1]["content"]
    references = st.session_state.get("references")
    user_prompt = st.session_state.messages[0]["content"]

# ---------------- Render assistant block ----------------
with st.chat_message("assistant"):
    st.header("Predicted Recipes")

    score0, dfs0 = _json_to_tables(recipe)
    if dfs0:
        _render_json_tables_block(
            recipe,
            base_name="prediction_tables__initial",
            expanded=True,
            title="📦 Predicted tables",
        )
    else:
        st.markdown(recipe)

    if use_rag and references:
        st.header("References")
        for i, ref in enumerate(references):
            with st.expander(f"Reference {i + 1}", expanded=False):
                st.markdown(ref)

# ---------------- History ----------------
if len(st.session_state.messages) > 2:
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ---------------- Chat input ----------------
prompt = st.chat_input("Ask a question about the recipe")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating response..."):
        response = call_completion_model(st.session_state.messages, model)

    with st.chat_message("assistant"):
        asst_text = response["choices"][0]["message"]["content"]

        score1, dfs1 = _json_to_tables(asst_text)
        if dfs1:
            _render_json_tables_block(
                asst_text,
                base_name="prediction_tables__chat",
                expanded=True,
                title="📦 Parse chat JSON to tables",
            )
        else:
            st.markdown(asst_text)
