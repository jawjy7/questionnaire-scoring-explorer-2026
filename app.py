"""
Psicometria e Personalità - Streamlit analysis tool

Run with:
    streamlit run app.py

The scoring configuration is deliberately stored as a dataframe in
``st.session_state`` rather than being embedded in the scoring engine.  This
keeps the psychometric rules editable from the UI while retaining one generic
calculation path for all scales.
"""

from __future__ import annotations

import io
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - shown as a friendly UI message
    SCIPY_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False


APP_TITLE = "Psicometria e Personalità"
ATTENTION_COLUMN = "controllo2"
DEMOGRAPHIC_COLUMNS = ["Q2", "Q13", "Q10_1", "Provincia", "Q3"]


def _csv(items: Iterable[str]) -> str:
    """Serialize a sequence for the editable configuration table."""
    return ", ".join(str(x) for x in items)


def _items(prefix: str, numbers: Sequence[int]) -> List[str]:
    """Create Qualtrics column names from a prefix and question numbers."""
    return [f"{prefix}{n}" for n in numbers]


def _refs(prefix: str, refs: Sequence[str]) -> List[str]:
    """Create column names from references such as ``6r`` or ``3``."""
    return [f"{prefix}{re.sub(r'[^0-9]', '', str(ref))}" for ref in refs]


def default_config() -> pd.DataFrame:
    """Return the complete editable scoring configuration.

    Each row is one factor. ``item_columns`` and ``reverse_items`` contain the
    actual dataframe column names, so edits made in ``st.data_editor`` are
    immediately usable by the generic scoring engine.
    """
    rows: List[Dict] = []

    def add(scale, factor, cols, reverse, lo, hi, method):
        rows.append({
            "scale": scale,
            "factor": factor,
            "item_columns": _csv(cols),
            "reverse_items": _csv(reverse),
            "min_val": lo,
            "max_val": hi,
            "method": method,
        })

    # BFI-44: Q210_1...Q210_22, Q211_1...Q211_22.
    bfi = [f"Q210_{i}" for i in range(1, 23)] + [f"Q211_{i}" for i in range(1, 23)]
    bfi_rev = [bfi[i - 1] for i in [6, 21, 31, 35]] + [bfi[i - 1] for i in [9, 12, 27, 37, 43]]
    # Explicit factor indices make the mapping auditable and avoid ambiguity
    # between Q210 and Q211 question numbering.
    bfi_groups = {
        "Extroversion": ([1, 6, 11, 16, 21, 26, 31, 36], [6, 21, 31]),
        "Agreeableness": ([2, 7, 12, 17, 22, 27, 32, 37, 42], [2, 12, 27, 37]),
        "Conscientiousness": ([3, 8, 13, 18, 23, 28, 33, 38, 43], [18, 23, 43]),
        "Neuroticism": ([4, 9, 14, 19, 24, 29, 34, 39], [9, 24, 34]),
        "Openness": ([5, 10, 15, 20, 25, 30, 35, 40, 41, 44], [35, 41]),
    }
    def bfi_col(n): return bfi[n - 1]
    for factor, (nums, rev_nums) in bfi_groups.items():
        add("BFI-44", factor, [bfi_col(n) for n in nums], [bfi_col(n) for n in rev_nums], 1, 5, "mean")

    bis_groups = {
        "BIS": ([2, 8, 13, 16, 19, 22, 24], [2, 22]),
        "BAS Drive": ([3, 9, 12, 21], []),
        "BAS Fun Seeking": ([5, 10, 15, 20], []),
        "BAS Reward Responsiveness": ([4, 7, 14, 18, 23], []),
    }
    for factor, (nums, rev_nums) in bis_groups.items():
        add("BIS/BAS", factor, _items("Q66_", nums), _items("Q66_", rev_nums), 1, 4, "sum")

    study_groups = {
        "Organizzazione": ([1, 9, 11, 17], [9, 17]),
        "Elaborazione": ([2, 13, 15, 19], [2]),
        "Autovalutazione": ([3, 4, 8, 16], [8]),
        "Strategie": ([5, 10, 12, 18], [5, 10, 12]),
        "Sensibilità Metacognitiva": ([6, 7, 14, 20], []),
    }
    for factor, (nums, rev_nums) in study_groups.items():
        add("Study Approach", factor, _items("Q7.2_", nums), _items("Q7.2_", rev_nums), 1, 5, "sum")

    add("Ansia Spaziale", "Total Score", _items("Q", range(120, 128)), [], 1, 6, "sum")
    add("Piacere di Esplorare", "Piacere di Esplorare", _items("Q", [129, 130, 133, 134, 137]), [], 1, 5, "sum")
    add("Piacere di Esplorare", "Piacere del Noto", _items("Q", [131, 132, 135, 136, 138]), [], 1, 5, "sum")

    orientation = {
        "Rappresentazione Survey": ["Q2.1", "Q3.1", "Q4", "Q5", "Q8", "Q9"],
        "Visuo-spaziale / Route": ["Q6", "Q7", "Q10", "Q11"],
        "Punti Cardinali": ["Q12", "Q13.1", "Q14"],
        "Non-official Total Mean": ["Q2.1", "Q3.1", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13.1", "Q14"],
    }
    for factor, cols in orientation.items():
        add("Senso dell'orientamento", factor, cols, [], 1, 6, "mean")

    epq = {
        "Psicoticismo (P)": ([2, 6, 10, 14, 18, 22, 26, 28, 31, 35, 39, 43], [2, 6, 18, 26, 28, 35, 43]),
        "Estroversione (E)": ([3, 7, 11, 15, 19, 23, 27, 32, 36, 41, 44, 48], [27, 41]),
        "Nevroticismo (N)": ([1, 5, 9, 13, 17, 21, 25, 30, 34, 38, 42, 46], []),
        "Lie (L)": ([4, 8, 12, 20, 24, 29, 33, 37, 40, 45, 47], [8, 12, 20, 24, 29, 33, 37, 40, 47]),
    }
    for factor, (nums, rev_nums) in epq.items():
        add("EPQ-R", factor, _items("EPI_", nums), _items("EPI_", rev_nums), 0, 1, "sum")

    memory_cols = _items("SM", range(1, 8)) + _items("SM", range(8, 35))
    add("Memory Sensitivity", "Total Score", memory_cols, [], 1, 5, "sum")
    return pd.DataFrame(rows)


def parse_csv_list(value) -> List[str]:
    """Parse a comma/semicolon/newline separated cell from the config editor."""
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r"[,;\n]+", str(value)) if x.strip()]


def read_uploaded_file(uploaded) -> Tuple[pd.DataFrame, List[str]]:
    """Read an Excel/CSV upload and handle Qualtrics' descriptive first row.

    Returns the dataframe and a human-readable cleaning log. The loader first
    tries normal headers, then falls back to headerless reading when the first
    row contains question identifiers.
    """
    logs: List[str] = []
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded, dtype=str, low_memory=False)
        logs.append("已按 CSV 首行读取列名。")
    else:
        raw = pd.read_excel(uploaded, dtype=str)
        logs.append("已按 Excel 首行读取列名。")

    def has_known_columns(frame: pd.DataFrame) -> bool:
        cols = {str(c).strip() for c in frame.columns}
        return bool(cols.intersection({"Q2", "Q3", "Provincia", "controllo2", "Q210_1", "EPI_1"}))

    if not has_known_columns(raw):
        uploaded.seek(0)
        if name.endswith(".csv"):
            candidate = pd.read_csv(uploaded, header=None, dtype=str, low_memory=False)
        else:
            candidate = pd.read_excel(uploaded, header=None, dtype=str)
        known_row = None
        for idx in range(min(4, len(candidate))):
            vals = set(candidate.iloc[idx].astype(str).str.strip())
            if vals.intersection({"Q2", "Q3", "Provincia", "controllo2", "Q210_1", "EPI_1"}):
                known_row = idx
                break
        if known_row is not None:
            raw = candidate.iloc[known_row + 1:].copy()
            raw.columns = candidate.iloc[known_row].astype(str).str.strip().tolist()
            logs.append(f"检测到 Qualtrics 变量名行（原始第 {known_row + 1} 行），已跳过题干/元数据行。")
        elif len(raw) > 0:
            # When variable names cannot be recovered, preserve data and make
            # the problem visible rather than guessing column mappings.
            logs.append("未识别到 Q 编码列名，已保留原始列名；请在上传文件中确认变量名。")

    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw = raw.dropna(how="all").reset_index(drop=True)
    logs.append(f"有效答卷行数：{len(raw)}；列数：{len(raw.columns)}。")
    return raw, logs


def numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert selected questionnaire columns to numeric, preserving NaN."""
    present = [c for c in columns if c in df.columns]
    out = pd.DataFrame(index=df.index)
    for col in present:
        cleaned = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        # Qualtrics exports can contain decimal commas.
        cleaned = cleaned.str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


def clean_data(raw: pd.DataFrame, config: pd.DataFrame, exclude_attention: bool) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Create anonymous IDs, numeric item columns, and attention flags.

    Missing values are not imputed here; imputation is dimension-specific in
    ``score_data`` so the audit can show the exact rule applied to every factor.
    """
    logs: List[str] = []
    data = raw.copy().reset_index(drop=True)
    # Replace an uploaded respondent identifier rather than exposing or
    # colliding with it. The generated ID is the only identifier used below.
    if "Subject_ID" in data.columns:
        data = data.drop(columns=["Subject_ID"])
    data.insert(0, "Subject_ID", [f"ID_{i:03d}" for i in range(1, len(data) + 1)])
    # Hide common direct identifiers while keeping the uploaded dataframe
    # usable for demographic filtering.
    sensitive_patterns = re.compile(r"(name|nome|surname|cognome|email|mail|telefono|phone|id\s*respondent)", re.I)
    hidden = [c for c in data.columns if sensitive_patterns.search(str(c)) and c != "Subject_ID"]
    if hidden:
        data = data.drop(columns=hidden)
        logs.append(f"已隐藏可能的直接身份列：{', '.join(hidden)}。")

    all_items = sorted(set(sum((parse_csv_list(v) for v in config["item_columns"]), [])))
    all_items += [ATTENTION_COLUMN] if ATTENTION_COLUMN not in all_items else []
    converted = numeric_frame(data, all_items)
    for col in converted.columns:
        data[col] = converted[col]

    if ATTENTION_COLUMN in data.columns:
        data["Attention_Fail"] = data[ATTENTION_COLUMN].notna() & (data[ATTENTION_COLUMN] != 1)
        fail_n = int(data["Attention_Fail"].sum())
        logs.append(f"注意力检查：{fail_n} 人 controllo2 != 1。")
    else:
        data["Attention_Fail"] = False
        logs.append("未找到 controllo2，已跳过注意力检查。")
    flags = data[["Subject_ID", "Attention_Fail"]].copy()
    flags["Included"] = ~flags["Attention_Fail"] if exclude_attention else True
    if exclude_attention:
        data = data.loc[flags["Included"]].reset_index(drop=True)
        logs.append(f"已剔除注意力检查失败者；剩余 {len(data)} 人。")
    return data, flags, logs


def score_data(data: pd.DataFrame, config: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[str]]]:
    """Apply reverse scoring, <=20% missing-value rule, and factor scoring.

    For each factor, if missingness exceeds 20% the score is NaN. Otherwise
    missing items are replaced by the respondent's mean on that factor. The
    function returns scored wide data, cleaned item data, an audit log, and a
    factor-to-item map used by reliability and straight-line diagnostics.
    """
    scored = data[["Subject_ID"] + [c for c in DEMOGRAPHIC_COLUMNS if c in data.columns]].copy()
    item_clean = data.copy()
    logs: List[str] = []
    factor_items: Dict[str, List[str]] = {}
    for _, row in config.iterrows():
        scale = str(row.get("scale", "Scale"))
        factor = str(row.get("factor", "Factor"))
        label = f"{scale} — {factor}"
        cols = [c for c in parse_csv_list(row.get("item_columns")) if c in data.columns]
        reverse = [c for c in parse_csv_list(row.get("reverse_items")) if c in cols]
        factor_items[label] = cols
        if not cols:
            scored[label] = np.nan
            logs.append(f"{label}：未找到任何匹配题目列，得分为空。")
            continue
        vals = numeric_frame(data, cols)
        lo, hi = float(row.get("min_val", 1)), float(row.get("max_val", 5))
        out_of_range = ((vals < lo) | (vals > hi)).sum().sum()
        if out_of_range:
            logs.append(f"{label}：发现 {int(out_of_range)} 个超出 [{lo:g}, {hi:g}] 的原始值。")
        for col in reverse:
            vals[col] = (lo + hi) - vals[col]
        missing_rate = vals.isna().mean(axis=1)
        valid = vals.notna().sum(axis=1) > 0
        row_means = vals.mean(axis=1)
        filled = vals.copy()
        for col in cols:
            filled[col] = filled[col].fillna(row_means)
        filled.loc[missing_rate > 0.20, :] = np.nan
        # Keep the same dimension-level missingness rule for reliability:
        # respondents with >20% missing are not allowed to enter alpha via
        # imputed values.
        for col in cols:
            item_clean.loc[filled.index, col] = filled[col]
        if str(row.get("method", "sum")).lower() == "mean":
            scores = filled.mean(axis=1)
        else:
            scores = filled.sum(axis=1, min_count=len(cols))
        scores.loc[~valid] = np.nan
        scored[label] = scores
        logs.append(f"{label}：{len(cols)} 题，{int((missing_rate > 0.20).sum())} 人缺失率 >20%，已标记 NaN。")

    # Stable scale-level totals are useful for poster tables and are derived
    # from the edited factor configuration, never from hard-coded item lists.
    for scale, group in config.groupby("scale", sort=False):
        factor_names = {str(x) for x in group["factor"]}
        # Some scales already have an explicit configured total row (for
        # example Ansia, Memory, and the orientation non-official total).
        if "Total Score" in factor_names or "Non-official Total Mean" in factor_names:
            continue
        labels = [f"{scale} — {x}" for x in group["factor"]]
        labels = [x for x in labels if x in scored.columns]
        if not labels:
            continue
        if scale == "BFI-44":
            # Weight each of the 44 items equally; averaging the five factor
            # means would incorrectly give the 8-item factors the same weight
            # as the 10-item factor.
            bfi_items = sum((factor_items.get(label, []) for label in labels), [])
            scored[f"{scale} — Total Mean"] = item_clean[bfi_items].mean(axis=1)
        elif scale != "EPQ-R":
            scored[f"{scale} — Total Score"] = scored[labels].sum(axis=1, min_count=1)
    return scored, item_clean, logs, factor_items


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Compute classical Cronbach's alpha on complete/imputed item responses."""
    x = items.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    k = x.shape[1]
    if k < 2 or len(x) < 2:
        return np.nan
    total_var = x.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_var) or total_var == 0:
        return np.nan
    return float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / total_var))


def stats_summary(scored: pd.DataFrame, item_clean: pd.DataFrame, config: pd.DataFrame, factor_items: Dict[str, List[str]]) -> pd.DataFrame:
    """Build descriptive statistics and Cronbach alpha for every configured factor."""
    rows = []
    for _, cfg in config.iterrows():
        label = f"{cfg['scale']} — {cfg['factor']}"
        if label not in scored:
            continue
        s = pd.to_numeric(scored[label], errors="coerce").dropna()
        alpha = cronbach_alpha(item_clean[factor_items.get(label, [])]) if factor_items.get(label) else np.nan
        rows.append({
            "Scale": cfg["scale"], "Factor": cfg["factor"], "N": int(s.count()),
            "Mean": s.mean(), "SD": s.std(ddof=1), "Median": s.median(),
            "Min": s.min(), "Max": s.max(),
            "Skewness": stats.skew(s, bias=False) if SCIPY_AVAILABLE and len(s) > 2 else np.nan,
            "Kurtosis": stats.kurtosis(s, bias=False) if SCIPY_AVAILABLE and len(s) > 3 else np.nan,
            "Cronbach_alpha": alpha,
        })
    return pd.DataFrame(rows)


def anomaly_report(data: pd.DataFrame, scored: pd.DataFrame, config: pd.DataFrame, factor_items: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect bounds, z/IQR extremes, straight-lining, and high Lie scores."""
    records: List[Dict] = []
    cell_flags: List[Dict] = []
    for _, cfg in config.iterrows():
        label = f"{cfg['scale']} — {cfg['factor']}"
        cols = factor_items.get(label, [])
        if not cols:
            continue
        lo, hi = float(cfg["min_val"]), float(cfg["max_val"])
        raw_vals = numeric_frame(data, cols)
        for idx, row in raw_vals.iterrows():
            bad = row[(row < lo) | (row > hi)]
            for col in bad.index:
                cell_flags.append({"Subject_ID": data.loc[idx, "Subject_ID"], "Variable": col, "Reason": f"越界 [{lo:g}, {hi:g}]", "Value": row[col]})

        values = pd.to_numeric(scored[label], errors="coerce")
        if values.notna().sum() >= 3:
            z = (values - values.mean()) / values.std(ddof=1)
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            extreme = values.index[(z.abs() > 3) | (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
            for idx in extreme:
                why = []
                if abs(z.loc[idx]) > 3: why.append("|z|>3")
                if values.loc[idx] < q1 - 1.5 * iqr or values.loc[idx] > q3 + 1.5 * iqr: why.append("IQR")
                records.append({"Subject_ID": scored.loc[idx, "Subject_ID"], "Variable": label, "Reason": "极端值（" + ", ".join(why) + "）", "Value": values.loc[idx]})
        for idx, row in raw_vals.iterrows():
            present = row.dropna()
            if len(present) >= 3 and present.nunique() == 1:
                records.append({"Subject_ID": data.loc[idx, "Subject_ID"], "Variable": label, "Reason": "Straight-lining（题目方差为 0）", "Value": present.iloc[0]})

    lie_col = next((c for c in scored.columns if c.endswith("— Lie (L)")), None)
    if lie_col:
        for idx in scored.index[scored[lie_col] > 8]:
            records.append({"Subject_ID": scored.loc[idx, "Subject_ID"], "Variable": lie_col, "Reason": "EPQ-R Lie 高分 > 8", "Value": scored.loc[idx, lie_col]})
    return pd.DataFrame(records), pd.DataFrame(cell_flags)


def available_factor_columns(scored: pd.DataFrame) -> List[str]:
    """Return factor score columns, excluding IDs and demographic columns."""
    # ``dict.fromkeys`` preserves order while protecting downstream scipy
    # calls from duplicate labels introduced by a user-added config row.
    return list(dict.fromkeys(c for c in scored.columns if c not in {"Subject_ID", *DEMOGRAPHIC_COLUMNS}))


def download_xlsx(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Serialize multiple dataframes into an Excel workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return buffer.getvalue()


def show_sidebar() -> Tuple[bool, str]:
    """Render navigation, instructions, and the attention-filter control."""
    st.sidebar.title("导航")
    page = st.sidebar.radio("模块", [
        "1 · 数据导入与预处理", "2 · 计分配置管理", "3 · 计分引擎", 
        "4 · 统计与异常检测", "5 · 交互式探索", "6 · 假设检验与可视化",
    ])
    st.sidebar.divider()
    exclude = st.sidebar.toggle("剔除 controllo2 != 1 的被试", value=st.session_state.get("exclude_attention", False))
    st.session_state["exclude_attention"] = exclude
    with st.sidebar.expander("使用说明", expanded=False):
        st.markdown("""
1. 上传 Qualtrics 的 `.xlsx` 或 `.csv`。
2. 在“计分配置”中按需要修改题目列、反向题、范围和 sum/mean。
3. 在“计分引擎”检查清洗日志与宽表。
4. 使用统计、筛选和假设检验模块制作海报结果。
5. 最后在页面底部导出 Excel。

导出的数据始终带有自动生成的 `Subject_ID`，不会把姓名等直接身份列带入分析表。
""")
    with st.sidebar.expander("问卷计分对照指南", expanded=False):
        st.markdown("""
- 反向题：`(min_val + max_val) - Raw Score`。
- BFI-44：1–5，五个维度取均值。
- BIS/BAS：1–4，BIS/BAS 各维度求和；填充题不计分。
- Study Approach：1–5，五维度与总分求和。
- Ansia Spaziale：1–6，8题求和。
- Piacere di Esplorare：1–5，两个分量表求和。
- Senso dell'orientamento：题目取均值；总体均值标注为非官方参考分。
- EPQ-R：0/1，P/E/N/L 求和；Lie >8 标记。
- Memory Sensitivity：1–5，34题求和；`controllo2=1` 为有效注意力检查。

配置中的 `item_columns` 使用上传文件真实列名，修改后会即时重算。
""")
    return exclude, page


def page_upload() -> None:
    """Data upload, anonymization, numeric conversion, and attention review."""
    st.header("1 · 数据导入与预处理")
    uploaded = st.file_uploader("上传 Qualtrics 数据", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("请上传 Excel 或 CSV 文件开始分析。支持先进入“计分配置”查看默认规则。")
        return
    try:
        raw, read_logs = read_uploaded_file(uploaded)
        data, flags, clean_logs = clean_data(raw, st.session_state.config_df, st.session_state.exclude_attention)
        st.session_state.raw_data = raw
        st.session_state.cleaned_data = data
        st.session_state.attention_flags = flags
        st.session_state.clean_logs = read_logs + clean_logs
        st.session_state.upload_name = uploaded.name
        st.success(f"已载入 {len(data)} 名当前纳入分析的被试。")
        c1, c2, c3 = st.columns(3)
        c1.metric("原始行数", len(raw)); c2.metric("当前样本 N", len(data)); c3.metric("注意力失败", int(flags["Attention_Fail"].sum()))
        st.subheader("清洗日志")
        st.text("\n".join(st.session_state.clean_logs))
        st.subheader("注意力检查")
        st.dataframe(flags, use_container_width=True, hide_index=True)
        st.subheader("预览（已匿名化）")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"文件读取失败：{exc}")


def page_config() -> None:
    """Editable configuration table and JSON import/export."""
    st.header("2 · 计分配置管理")
    st.caption("每一行代表一个因子。题目列与反向题用逗号分隔；修改后计分引擎会使用新规则。")
    edited = st.data_editor(
        st.session_state.config_df,
        key="config_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "min_val": st.column_config.NumberColumn("min", min_value=-100, max_value=100),
            "max_val": st.column_config.NumberColumn("max", min_value=-100, max_value=100),
            "method": st.column_config.SelectboxColumn("method", options=["sum", "mean"], required=True),
        },
    )
    st.session_state.config_df = edited.copy()
    left, right = st.columns(2)
    with left:
        st.download_button("下载当前 JSON 配置", json.dumps(st.session_state.config_df.to_dict("records"), ensure_ascii=False, indent=2).encode("utf-8"), "scoring_config.json", "application/json")
    with right:
        config_file = st.file_uploader("导入 JSON 配置", type=["json"], key="config_upload")
        if config_file is not None:
            try:
                loaded = pd.DataFrame(json.load(config_file))
                required = {"scale", "factor", "item_columns", "reverse_items", "min_val", "max_val", "method"}
                if not required.issubset(loaded.columns):
                    st.error(f"配置缺少字段：{', '.join(sorted(required - set(loaded.columns)))}")
                else:
                    st.session_state.config_df = loaded
                    st.success("配置已导入，请切换模块或重新运行页面。")
            except Exception as exc:
                st.error(f"JSON 读取失败：{exc}")


def require_data() -> bool:
    """Show a consistent message when analysis pages lack an upload."""
    if "cleaned_data" not in st.session_state:
        st.warning("请先在模块 1 上传数据。")
        return False
    return True


def ensure_scores() -> None:
    """Recompute scores from the current editable configuration."""
    scored, items, logs, factor_items = score_data(st.session_state.cleaned_data, st.session_state.config_df)
    st.session_state.scored_data = scored
    st.session_state.item_clean = items
    st.session_state.score_logs = logs
    st.session_state.factor_items = factor_items
    st.session_state.summary = stats_summary(scored, items, st.session_state.config_df, factor_items)
    st.session_state.anomalies, st.session_state.cell_anomalies = anomaly_report(st.session_state.cleaned_data, scored, st.session_state.config_df, factor_items)


def page_scoring() -> None:
    """Run and display the generic scoring engine."""
    st.header("3 · 计分引擎")
    if not require_data(): return
    ensure_scores()
    st.success("计分完成：反向题、维度缺失率规则、因子分数与量表总分已生成。")
    with st.expander("计分与缺失值日志", expanded=False):
        st.text("\n".join(st.session_state.score_logs))
    st.dataframe(st.session_state.scored_data, use_container_width=True, hide_index=True)
    st.caption("分数表包含 Subject_ID、人口学变量、各因子得分及自动生成的总分。")


def page_statistics() -> None:
    """Show psychometric summary and anomaly diagnostics."""
    st.header("4 · 统计与异常检测")
    if not require_data(): return
    ensure_scores()
    st.subheader("描述统计与 Cronbach's α")
    st.dataframe(st.session_state.summary.style.format({"Mean": "{:.3f}", "SD": "{:.3f}", "Cronbach_alpha": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.subheader("异常原因汇总")
    if st.session_state.anomalies.empty:
        st.success("未检测到得分层面的极端值、straight-lining 或 Lie 高分。")
    else:
        st.dataframe(st.session_state.anomalies, use_container_width=True, hide_index=True)
    if not st.session_state.cell_anomalies.empty:
        st.subheader("原始作答越界单元格")
        st.dataframe(st.session_state.cell_anomalies, use_container_width=True, hide_index=True)
    st.caption("极端值同时使用 |z|>3 与 1.5×IQR；异常检测不自动删除记录。")


def demographic_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render demographic controls and return the filtered score dataframe."""
    filtered = df.copy()
    cols = st.columns(4)
    for i, col in enumerate(["Q2", "Q13", "Provincia"]):
        if col not in filtered.columns: continue
        values = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = cols[i].multiselect(col, values, key=f"{key_prefix}_{col}")
        if selected: filtered = filtered[filtered[col].astype(str).isin(selected)]
    if "Q3" in filtered.columns:
        q3 = pd.to_numeric(filtered["Q3"], errors="coerce")
        if q3.notna().any():
            lo, hi = float(q3.min()), float(q3.max())
            if lo < hi:
                selected_range = cols[3].slider("Q3 / CFU 范围", lo, hi, (lo, hi), key=f"{key_prefix}_q3")
                filtered = filtered[q3.between(*selected_range)]
    return filtered


def page_explorer() -> None:
    """Interactive subgroup filtering and full-vs-subgroup summaries."""
    st.header("5 · 交互式探索")
    if not require_data(): return
    ensure_scores()
    filtered = demographic_filter(st.session_state.scored_data, "explorer")
    st.metric("符合条件的子样本 N", len(filtered))
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    factor_cols = available_factor_columns(filtered)
    if factor_cols and not filtered.empty:
        sub = filtered[factor_cols].describe().T.reset_index().rename(columns={"index": "Factor"})
        full = st.session_state.scored_data[factor_cols].describe().T[["mean", "std", "count"]].reset_index().rename(columns={"index": "Factor", "mean": "Full_Mean", "std": "Full_SD", "count": "Full_N"})
        comparison = sub.merge(full, on="Factor", how="left")
        st.subheader("子组与全样本描述统计")
        st.dataframe(comparison, use_container_width=True, hide_index=True)


def correlation_analysis(df: pd.DataFrame) -> None:
    """Render Pearson/Spearman correlation matrix, p-values, heatmap, scatter."""
    # Keep one physical column per label even if a user accidentally adds a
    # duplicate factor in the editable configuration.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    factors = available_factor_columns(df)
    selected = st.multiselect("选择 2–8 个因子", factors, default=factors[:min(3, len(factors))], key="corr_factors")
    method = st.radio("相关类型", ["Pearson", "Spearman"], horizontal=True, key="corr_method")
    if len(selected) < 2: return
    corr = df[selected].corr(method=method.lower())
    pmat = pd.DataFrame(np.nan, index=selected, columns=selected)
    if SCIPY_AVAILABLE:
        for a in selected:
            for b in selected:
                # The diagonal is known exactly. Selecting [a, a] from a
                # pandas frame creates two columns with the same name and was
                # the source of SciPy's ``shapes (N,2) ...`` ValueError.
                if a == b:
                    pmat.loc[a, b] = 0.0
                    continue
                pair = pd.DataFrame({"x": pd.to_numeric(df[a], errors="coerce"), "y": pd.to_numeric(df[b], errors="coerce")}).dropna()
                if len(pair) >= 3:
                    result = stats.pearsonr(pair["x"], pair["y"]) if method == "Pearson" else stats.spearmanr(pair["x"], pair["y"])
                    pmat.loc[a, b] = result.pvalue
    valid_n = pd.DataFrame(index=selected, columns=selected, dtype=float)
    for a in selected:
        for b in selected:
            valid_n.loc[a, b] = df[[a, b]].dropna().shape[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("变量数", len(selected))
    c2.metric("最大有效 N", int(valid_n.max().max()))
    c3.metric("显著相关（p < .05）", int(((pmat < 0.05) & ~np.eye(len(selected), dtype=bool)).sum().sum() / 2))
    st.subheader("相关结果")
    result_tab1, result_tab2, result_tab3 = st.tabs(["系数矩阵", "p-value 矩阵", "有效样本 N"])
    with result_tab1:
        st.dataframe(corr.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
    with result_tab2:
        st.dataframe(pmat.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.10), use_container_width=True)
    with result_tab3:
        st.dataframe(valid_n.style.format("{:.0f}"), use_container_width=True)
    if PLOTLY_AVAILABLE:
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"{method} correlation")
        st.plotly_chart(fig, use_container_width=True)
        pair = st.selectbox("散点图变量 X", selected, key="scatter_x")
        pair_y = st.selectbox("散点图变量 Y", [x for x in selected if x != pair], key="scatter_y")
        plot_df = pd.DataFrame({pair: pd.to_numeric(df[pair], errors="coerce"), pair_y: pd.to_numeric(df[pair_y], errors="coerce")}).dropna()
        if len(plot_df) >= 3:
            fig2 = px.scatter(plot_df, x=pair, y=pair_y, trendline="ols", title=f"{pair} vs {pair_y}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("这两个变量的有效配对样本少于 3，无法绘制回归线。")


def group_analysis(df: pd.DataFrame) -> None:
    """Render independent t-test/Cohen's d or one-way ANOVA/eta-squared."""
    factors = available_factor_columns(df)
    group_options = [c for c in ["Q2", "Q13", "Q10_1", "Provincia"] if c in df.columns]
    if not group_options or not factors:
        st.info("数据中没有可用的人口学分组列或因子列。")
        return
    group_col = st.selectbox("分组变量", group_options, key="group_col")
    outcome = st.selectbox("结果变量", factors, key="group_outcome")
    work = df[[group_col, outcome]].dropna().copy()
    work[group_col] = work[group_col].astype(str)
    groups = [g for g in work[group_col].unique()]
    if len(groups) < 2:
        st.info("至少需要两个有效组。")
        return
    grouped = [work.loc[work[group_col] == g, outcome].astype(float).values for g in groups]
    if not SCIPY_AVAILABLE:
        st.error("需要 scipy 才能进行假设检验。")
        return
    if len(groups) == 2:
        t, p = stats.ttest_ind(grouped[0], grouped[1], equal_var=False, nan_policy="omit")
        pooled = np.sqrt(((len(grouped[0])-1)*np.var(grouped[0], ddof=1) + (len(grouped[1])-1)*np.var(grouped[1], ddof=1)) / (len(grouped[0])+len(grouped[1])-2))
        d = (np.mean(grouped[0]) - np.mean(grouped[1])) / pooled if pooled else np.nan
        result = pd.DataFrame([{"Test": "Welch independent t-test", "Group_1": groups[0], "Group_2": groups[1], "t": t, "p": p, "Cohen_d": d, "N_1": len(grouped[0]), "N_2": len(grouped[1])}])
    else:
        f, p = stats.f_oneway(*grouped)
        grand = work[outcome].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in grouped)
        ss_total = sum((work[outcome] - grand) ** 2)
        result = pd.DataFrame([{"Test": "One-way ANOVA", "Groups": len(groups), "F": f, "p": p, "Eta_squared": ss_between / ss_total if ss_total else np.nan}])
    st.dataframe(result.style.format({c: "{:.4f}" for c in result.columns if c not in {"Test", "Group_1", "Group_2"}}), use_container_width=True, hide_index=True)
    if PLOTLY_AVAILABLE:
        st.plotly_chart(px.box(work, x=group_col, y=outcome, points="all", title=f"{outcome} by {group_col}"), use_container_width=True)
        means = work.groupby(group_col)[outcome].agg(["mean", "sem"]).reset_index()
        fig = go.Figure([go.Bar(x=means[group_col], y=means["mean"], error_y=dict(type="data", array=means["sem"], visible=True))])
        fig.update_layout(title="分组均值 ± 标准误", xaxis_title=group_col, yaxis_title=outcome)
        st.plotly_chart(fig, use_container_width=True)


def page_hypotheses() -> None:
    """Correlation, group comparisons, and Excel export for poster preparation."""
    st.header("6 · 假设检验与可视化")
    if not require_data(): return
    ensure_scores()
    if not SCIPY_AVAILABLE:
        st.warning("当前环境未安装 scipy；相关分析和组间检验不可用。")
    st.markdown("选择一个分析卡片开始。所有检验都保留当前筛选前的全样本分数，并在结果中显示有效样本量。")
    tab1, tab2, tab3 = st.tabs(["🔗 相关分析", "⚖️ 组间比较", "📥 一键导出"])
    with tab1:
        correlation_analysis(st.session_state.scored_data)
    with tab2:
        group_analysis(st.session_state.scored_data)
    with tab3:
        sheets = {
            "Cleaned_Data": st.session_state.item_clean,
            "Scored_Wide": st.session_state.scored_data,
            "Summary": st.session_state.summary,
            "Anomalies": st.session_state.anomalies,
            "Attention_Flags": st.session_state.attention_flags,
        }
        if st.button("准备 Excel 导出", key="prepare_export"):
            try:
                st.session_state.export_bytes = download_xlsx(sheets)
                st.success("Excel 文件已生成。")
            except Exception as exc:
                st.error(f"Excel 导出失败：{exc}")
        if "export_bytes" in st.session_state:
            st.download_button("下载分析结果.xlsx", st.session_state.export_bytes, "psychometrics_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("下载配置 JSON", json.dumps(st.session_state.config_df.to_dict("records"), ensure_ascii=False, indent=2).encode("utf-8"), "scoring_config.json", "application/json")


def inject_ui_style() -> None:
    """Apply a restrained visual system so the app reads as one product."""
    st.markdown("""
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
      [data-testid="stMetric"] { background: #f7f9fc; border: 1px solid #e6ebf2; padding: .8rem 1rem; border-radius: .75rem; }
      [data-testid="stMetricValue"] { color: #17324d; }
      div[data-testid="stTabs"] button { font-weight: 600; }
      section[data-testid="stSidebar"] { border-right: 1px solid #e6ebf2; }
      h1, h2, h3 { letter-spacing: -0.02em; color: #17324d; }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    inject_ui_style()
    if "config_df" not in st.session_state:
        st.session_state.config_df = default_config()
    st.title(APP_TITLE)
    st.caption("Qualtrics 问卷清洗、心理测量计分、异常监测与海报统计工具")
    _, page = show_sidebar()
    if page.startswith("1"):
        page_upload()
    elif page.startswith("2"):
        page_config()
    elif page.startswith("3"):
        page_scoring()
    elif page.startswith("4"):
        page_statistics()
    elif page.startswith("5"):
        page_explorer()
    else:
        page_hypotheses()


if __name__ == "__main__":
    main()
