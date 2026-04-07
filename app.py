"""
採用ファネル モニタリングダッシュボード v3
課題発見→施策出しまでのサイクルを加速するプロ仕様ダッシュボード
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

st.set_page_config(
    page_title="Hiring Funnel",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = ["応募", "書類通過", "1次面接実施", "2次面接実施", "内定", "内定承諾"]
TRANSITIONS = [
    ("応募→書類通過", "応募", "書類通過"),
    ("書類通過→1次面接", "書類通過", "1次面接実施"),
    ("1次面接→2次面接", "1次面接実施", "2次面接実施"),
    ("2次面接→内定", "2次面接実施", "内定"),
    ("内定→内定承諾", "内定", "内定承諾"),
]

C50  = "#EFF6FF"
C100 = "#DBEAFE"
C200 = "#BFDBFE"
C300 = "#93C5FD"
C400 = "#60A5FA"
C500 = "#3B82F6"
C600 = "#2563EB"
C700 = "#1D4ED8"
C800 = "#1E40AF"
C900 = "#1E3A8A"

TXT  = "#E2E8F0"
TXT2 = "#94A3B8"
TXT3 = "#64748B"
BG   = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.06)"

FUNNEL_GRAD = [C100, C200, C300, C400, C500, C600]


CHART_LAYOUT = dict(
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(color=TXT),
)

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {C800} 0%, {C900} 100%);
        padding: 16px 20px; border-radius: 10px; color: white;
    }}
    div[data-testid="stMetric"] label {{ color: rgba(255,255,255,0.7) !important; font-size: 0.85rem !important; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: white !important; font-size: 1.8rem !important; }}
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{ color: {C200} !important; }}
    .section-header {{
        background: rgba(30,58,95,0.35); color: {TXT};
        padding: 8px 16px; border-radius: 6px; font-size: 1rem;
        font-weight: bold; margin: 1.5rem 0 0.8rem 0;
        border-left: 4px solid {C500};
    }}
    .alert-danger {{
        background: rgba(30,64,175,0.15); border-left: 4px solid {C400};
        padding: 12px 16px; border-radius: 6px; margin: 6px 0;
        color: {TXT}; font-size: 0.92rem;
    }}
    .alert-warning {{
        background: rgba(30,64,175,0.10); border-left: 4px solid {C500};
        padding: 12px 16px; border-radius: 6px; margin: 6px 0;
        color: {TXT2}; font-size: 0.92rem;
    }}
    .alert-info {{
        background: rgba(30,64,175,0.08); border-left: 4px solid {C600};
        padding: 12px 16px; border-radius: 6px; margin: 6px 0;
        color: {TXT2}; font-size: 0.92rem;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_process(uploaded_bytes: bytes, encoding: str) -> pd.DataFrame:
    text = uploaded_bytes.decode(encoding, errors="replace")
    df = pd.read_csv(StringIO(text))
    cols = list(df.columns)

    df["_dept"] = df.iloc[:, 1].fillna("").astype(str).str.strip()
    df["_pos"] = df.iloc[:, 2].fillna("").astype(str).str.strip()

    ch_raw = df.iloc[:, 5].fillna("").astype(str).str.strip()
    custom = df.iloc[:, 8].fillna("").astype(str).str.strip()
    mask_custom = ch_raw == "カスタム"
    mask_agent = ch_raw == "エージェントから"
    mask_page = ch_raw == "応募ページから"
    mask_ref = ch_raw == "社員紹介から"
    mask_empty = ch_raw == ""

    channel = custom.where(custom != "", "カスタム")
    channel[mask_agent] = "エージェント"
    channel[mask_page] = "自主応募（応募ページ）"
    channel[mask_ref] = "社員紹介"
    channel[mask_empty] = "不明"
    channel[~(mask_custom | mask_agent | mask_page | mask_ref | mask_empty)] = ch_raw[
        ~(mask_custom | mask_agent | mask_page | mask_ref | mask_empty)
    ]
    df["_channel"] = channel

    df["_agent_name"] = df.iloc[:, 6].fillna("").astype(str).str.strip()

    df["_apply_date"] = df.iloc[:, 14].fillna("").astype(str).str.strip()
    df["_result"] = df.iloc[:, 16].fillna("").astype(str).str.strip()

    doc1 = df.iloc[:, 38].fillna("").astype(str).str.strip() if len(cols) > 38 else pd.Series("")
    doc2 = df.iloc[:, 52].fillna("").astype(str).str.strip() if len(cols) > 52 else pd.Series("")
    df["_doc"] = doc2.where(doc2 != "", doc1)
    df["_int1"] = df.iloc[:, 64].fillna("").astype(str).str.strip() if len(cols) > 64 else pd.Series("")
    df["_int2"] = df.iloc[:, 78].fillna("").astype(str).str.strip() if len(cols) > 78 else pd.Series("")
    off1 = df.iloc[:, 110].fillna("").astype(str).str.strip() if len(cols) > 110 else pd.Series("")
    off2 = df.iloc[:, 120].fillna("").astype(str).str.strip() if len(cols) > 120 else pd.Series("")
    df["_offer"] = off2.where(off2 != "", off1)

    hired = df["_result"] == "採用"
    has_offer = (df["_offer"] != "") | hired
    has_int2 = (df["_int2"] != "") | has_offer
    has_int1 = (df["_int1"] != "") | has_int2
    has_doc = (df["_doc"] == "通過") | has_int1
    has_apply = df["_apply_date"] != ""

    df["_f_応募"] = has_apply.astype(int)
    df["_f_書類通過"] = (has_apply & has_doc).astype(int)
    df["_f_1次面接実施"] = (has_apply & has_int1).astype(int)
    df["_f_2次面接実施"] = (has_apply & has_int2).astype(int)
    df["_f_内定"] = (has_apply & has_offer).astype(int)
    df["_f_内定承諾"] = (has_apply & hired).astype(int)

    flag_cols = [f"_f_{s}" for s in STAGES]
    flags = df[flag_cols].values
    indices = np.arange(len(flag_cols))
    df["_highest"] = np.where(flags.sum(axis=1) > 0, (flags * indices).max(axis=1), -1)

    df["_apply_dt"] = pd.to_datetime(df.iloc[:, 14], errors="coerce")
    df["_result_dt"] = pd.to_datetime(df.iloc[:, 15], errors="coerce") if len(cols) > 15 else pd.NaT
    df["_doc_judge_dt"] = pd.to_datetime(df.iloc[:, 43], errors="coerce") if len(cols) > 43 else pd.NaT
    df["_int1_sched_dt"] = pd.to_datetime(df.iloc[:, 68], errors="coerce") if len(cols) > 68 else pd.NaT
    df["_int1_judge_dt"] = pd.to_datetime(df.iloc[:, 69], errors="coerce") if len(cols) > 69 else pd.NaT
    df["_int2_sched_dt"] = pd.to_datetime(df.iloc[:, 82], errors="coerce") if len(cols) > 82 else pd.NaT
    df["_int2_judge_dt"] = pd.to_datetime(df.iloc[:, 83], errors="coerce") if len(cols) > 83 else pd.NaT
    df["_offer_judge_dt"] = pd.to_datetime(df.iloc[:, 115], errors="coerce") if len(cols) > 115 else pd.NaT

    df["_lt_apply_to_doc"] = (df["_doc_judge_dt"] - df["_apply_dt"]).dt.days
    df["_lt_doc_to_int1"] = (df["_int1_sched_dt"] - df["_doc_judge_dt"]).dt.days
    df["_lt_int1_to_int2"] = (df["_int2_sched_dt"] - df["_int1_judge_dt"]).dt.days
    df["_lt_total"] = (df["_result_dt"] - df["_apply_dt"]).dt.days

    return df[df["_f_応募"] == 1].copy()


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_funnel(df: pd.DataFrame):
    n = len(df)
    if n == 0:
        return None

    counts = {s: int(df[f"_f_{s}"].sum()) for s in STAGES}
    transitions = {}
    for label, from_s, to_s in TRANSITIONS:
        from_idx = STAGES.index(from_s)
        to_idx = STAGES.index(to_s)
        entered = df[df[f"_f_{from_s}"] == 1]
        total_entered = len(entered)
        passed = int(entered[f"_f_{to_s}"].sum())
        not_passed = entered[entered[f"_f_{to_s}"] == 0]
        in_range = not_passed[
            (not_passed["_highest"] >= from_idx) & (not_passed["_highest"] < to_idx)
        ]
        rejected = int((in_range["_result"] == "お見送り").sum())
        withdrew = int((in_range["_result"] == "辞退").sum())
        in_progress = int((in_range["_result"] == "選考中").sum())
        confirmed_total = total_entered - in_progress
        transitions[label] = {
            "total_entered": total_entered,
            "passed": passed,
            "rejected": rejected,
            "withdrew": withdrew,
            "in_progress": in_progress,
            "pass_rate": passed / total_entered if total_entered else 0,
            "reject_rate": rejected / total_entered if total_entered else 0,
            "withdraw_rate": withdrew / total_entered if total_entered else 0,
            "in_progress_rate": in_progress / total_entered if total_entered else 0,
            "confirmed_total": confirmed_total,
            "confirmed_pass_rate": passed / confirmed_total if confirmed_total else 0,
            "confirmed_reject_rate": rejected / confirmed_total if confirmed_total else 0,
            "confirmed_withdraw_rate": withdrew / confirmed_total if confirmed_total else 0,
        }
    return {"counts": counts, "transitions": transitions, "total": n}


def compute_lead_times(df: pd.DataFrame):
    lt_defs = [
        ("応募→書類判定", "_lt_apply_to_doc"),
        ("書類判定→1次面接", "_lt_doc_to_int1"),
        ("1次面接→2次面接", "_lt_int1_to_int2"),
        ("応募→結果確定", "_lt_total"),
    ]
    results = []
    for label, col in lt_defs:
        vals = df[col].dropna()
        vals = vals[vals >= 0]
        if len(vals) >= 3:
            results.append({
                "label": label,
                "n": len(vals),
                "median": vals.median(),
                "mean": vals.mean(),
                "p75": vals.quantile(0.75),
                "p25": vals.quantile(0.25),
            })
    return results


def compute_weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["_apply_dt"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()
    valid["_week_start"] = valid["_apply_dt"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = valid.groupby("_week_start").agg(
        応募=("_f_応募", "sum"),
        書類通過=("_f_書類通過", "sum"),
        内定=("_f_内定", "sum"),
    ).sort_index()
    weekly["書類通過率"] = (weekly["書類通過"] / weekly["応募"] * 100).round(1)
    return weekly


def compute_channel_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, grp in df.groupby("_channel"):
        n = len(grp)
        if n < 2:
            continue
        doc_pass = int(grp["_f_書類通過"].sum())
        int1 = int(grp["_f_1次面接実施"].sum())
        offer = int(grp["_f_内定"].sum())
        hired = int(grp["_f_内定承諾"].sum())
        rows.append({
            "チャネル": ch,
            "応募数": n,
            "書類通過": doc_pass,
            "書類通過率": doc_pass / n if n else 0,
            "1次面接": int1,
            "内定": offer,
            "内定承諾": hired,
            "歩留まり": hired / n if n else 0,
        })
    return pd.DataFrame(rows).sort_values("応募数", ascending=False)


def compute_agent_performance(df: pd.DataFrame) -> pd.DataFrame:
    agents = df[df["_agent_name"] != ""]
    if len(agents) == 0:
        return pd.DataFrame()
    rows = []
    for name, grp in agents.groupby("_agent_name"):
        n = len(grp)
        if n < 2:
            continue
        doc_pass = int(grp["_f_書類通過"].sum())
        int1 = int(grp["_f_1次面接実施"].sum())
        offer = int(grp["_f_内定"].sum())
        rows.append({
            "エージェント": name,
            "紹介数": n,
            "書類通過": doc_pass,
            "書類通過率": doc_pass / n if n else 0,
            "1次面接": int1,
            "内定": offer,
        })
    return pd.DataFrame(rows).sort_values("紹介数", ascending=False)


def detect_alerts(funnel: dict, lead_times: list, channel_eff: pd.DataFrame,
                  agent_perf: pd.DataFrame):
    alerts = []
    trans = funnel["transitions"]

    for label, t in trans.items():
        if t["confirmed_total"] < 5:
            continue
        if t["confirmed_reject_rate"] > 0.40:
            alerts.append({
                "level": "danger",
                "text": f"🚨 {label}：お断り率が {t['confirmed_reject_rate']:.0%}（{t['rejected']}名 / 確定{t['confirmed_total']}名）。"
                        f"応募品質の見直し or 書類選考基準の再検討を推奨。",
            })
        if t["confirmed_withdraw_rate"] > 0.15:
            alerts.append({
                "level": "warning",
                "text": f"⚠️ {label}：辞退率が {t['confirmed_withdraw_rate']:.0%}（{t['withdrew']}名 / 確定{t['confirmed_total']}名）。"
                        f"候補者体験・選考スピードの改善を検討。",
            })
        if t["in_progress_rate"] > 0.20 and t["in_progress"] >= 5:
            alerts.append({
                "level": "info",
                "text": f"📋 {label}：選考中が {t['in_progress']}名（{t['in_progress_rate']:.0%}）滞留。"
                        f"次ステップへの日程調整を促進してください。",
            })

    for lt in lead_times:
        if lt["label"] == "書類判定→1次面接" and lt["median"] > 5:
            alerts.append({
                "level": "warning",
                "text": f"⚠️ 書類判定→1次面接の設定に中央値 {lt['median']:.0f}日（75%ile: {lt['p75']:.0f}日）。"
                        f"目標5日以内。面接官の日程確保がボトルネックの可能性。",
            })
        if lt["label"] == "応募→書類判定" and lt["median"] > 3:
            alerts.append({
                "level": "warning",
                "text": f"⚠️ 書類選考に中央値 {lt['median']:.0f}日。迅速な書類確認体制の構築を。",
            })

    if len(agent_perf) > 0:
        zero_agents = agent_perf[(agent_perf["書類通過率"] == 0) & (agent_perf["紹介数"] >= 5)]
        if len(zero_agents) > 0:
            names = "、".join(zero_agents["エージェント"].tolist()[:3])
            total = int(zero_agents["紹介数"].sum())
            alerts.append({
                "level": "danger",
                "text": f"🚨 書類通過率0%のエージェント: {names}（計{total}名紹介）。"
                        f"紹介要件のすり合わせ or 取引見直しを検討。",
            })

    if len(channel_eff) > 0:
        top = channel_eff.head(5)
        best = top.loc[top["書類通過率"].idxmax()]
        if best["書類通過率"] > 0.30 and best["応募数"] >= 10:
            alerts.append({
                "level": "info",
                "text": f"💡 {best['チャネル']}の書類通過率が {best['書類通過率']:.0%} と高効率"
                        f"（{best['応募数']}名中 {best['書類通過']}名通過）。投資拡大を検討。",
            })

    return alerts


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def make_funnel_chart(funnel, confirmed_mode=True):
    """Single-trace horizontal bar funnel with blue gradient."""
    counts = funnel["counts"]
    trans = funnel["transitions"]
    total = counts["応募"]
    values = [counts[s] for s in STAGES]

    survival = [v / total * 100 if total else 0 for v in values]

    conv_rates = [100.0]
    for i in range(1, len(values)):
        t = trans[TRANSITIONS[i - 1][0]]
        if confirmed_mode:
            conv_rates.append(t["confirmed_pass_rate"] * 100)
        else:
            conv_rates.append(values[i] / values[i - 1] * 100 if values[i - 1] else 0)

    bar_texts = []
    for i, (v, sr, cr) in enumerate(zip(values, survival, conv_rates)):
        if i == 0:
            bar_texts.append(f"  {v:,}名（100%）")
        else:
            ip = trans[TRANSITIONS[i - 1][0]]["in_progress"] if not confirmed_mode else 0
            ip_note = f"  ({ip}名 選考中)" if ip > 0 else ""
            bar_texts.append(f"  {v:,}名  残存{sr:.1f}%  移行率{cr:.0f}%{ip_note}")

    fig = go.Figure(go.Bar(
        y=STAGES, x=values, orientation="h",
        marker_color=FUNNEL_GRAD[:len(STAGES)], opacity=0.85,
        text=bar_texts, textposition="outside",
        textfont=dict(size=12, color=TXT),
        cliponaxis=False,
    ))

    max_val = max(values) if values else 1
    fig.update_layout(
        **CHART_LAYOUT,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=TXT2)),
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, max_val * 2.5]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340, bargap=0.35,
    )
    return fig


def make_drop_bars(funnel, confirmed_mode=True):
    """100% stacked horizontal bars — drop reasons normalized to total drops."""
    trans = funnel["transitions"]
    y_labels = [t[0] for t in TRANSITIONS]
    y_labels.reverse()

    if confirmed_mode:
        seg_keys = [("お断り", "rejected", C700), ("辞退", "withdrew", C400)]
    else:
        seg_keys = [("お断り", "rejected", C700), ("辞退", "withdrew", C400),
                    ("選考中", "in_progress", C200)]

    totals = []
    for t_def in TRANSITIONS:
        t = trans[t_def[0]]
        totals.append(sum(t[k] for _, k, _ in seg_keys))
    totals.reverse()

    fig = go.Figure()
    for name, key, color in seg_keys:
        raw = [trans[t_def[0]][key] for t_def in TRANSITIONS]
        raw.reverse()
        pcts = [v / tot * 100 if tot > 0 else 0 for v, tot in zip(raw, totals)]
        texts = [f"{p:.0f}% ({v}名)" if p >= 12 else (f"{p:.0f}%" if p >= 5 else "")
                 for p, v in zip(pcts, raw)]
        fig.add_trace(go.Bar(
            y=y_labels, x=pcts, orientation="h", name=name,
            marker_color=color, opacity=0.85,
            text=texts, textposition="inside",
            textfont=dict(color="white", size=11),
            customdata=raw,
            hovertemplate="%{y}<br>%{fullData.name}: %{customdata}名 (%{x:.0f}%)<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT, barmode="stack",
        yaxis=dict(tickfont=dict(size=10, color=TXT2)),
        xaxis=dict(ticksuffix="%", range=[0, 105], showgrid=False,
                   tickfont=dict(color=TXT3)),
        legend=dict(orientation="h", y=-0.12, x=0,
                    font=dict(size=10, color=TXT2),
                    traceorder="normal"),
        margin=dict(l=5, r=5, t=10, b=35),
        height=340, bargap=0.3,
    )
    return fig


def make_lead_time_chart(lead_times: list) -> go.Figure:
    labels = [lt["label"] for lt in lead_times]
    medians = [lt["median"] for lt in lead_times]
    p75s = [lt["p75"] for lt in lead_times]
    p25s = [lt["p25"] for lt in lead_times]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=medians, orientation="h", name="中央値",
        marker_color=C500, opacity=0.85,
        text=[f" {v:.0f}日" for v in medians], textposition="outside",
        textfont=dict(size=14, color=C300),
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[p - m for p, m in zip(p75s, medians)], orientation="h",
        name="75%ile", marker_color=C300, opacity=0.4, base=medians,
        text=[f" {v:.0f}日" for v in p75s], textposition="outside",
        textfont=dict(size=12, color=TXT2),
    ))

    fig.update_layout(
        **CHART_LAYOUT, barmode="stack",
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=TXT2)),
        xaxis=dict(title=dict(text="日数", font=dict(color=TXT2)),
                   showgrid=True, gridcolor=GRID, tickfont=dict(color=TXT3)),
        legend=dict(orientation="h", y=-0.2, x=0.2, font=dict(size=12, color=TXT2)),
        margin=dict(l=10, r=60, t=10, b=40), height=240,
    )
    return fig


def make_weekly_chart(weekly: pd.DataFrame) -> go.Figure:
    labels = [d.strftime("W%V (%m/%d〜)") for d in weekly.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=weekly["応募"], name="応募数", marker_color=C500, opacity=0.7,
        text=weekly["応募"], textposition="outside", textfont=dict(color=C300, size=13),
    ))
    fig.add_trace(go.Bar(
        x=labels, y=weekly["書類通過"], name="書類通過", marker_color=C300, opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=weekly["書類通過率"], name="書類通過率(%)",
        yaxis="y2", mode="lines+markers+text",
        line=dict(color=TXT2, width=2), marker=dict(size=7, color=TXT2),
        text=[f"{v:.0f}%" for v in weekly["書類通過率"]], textposition="top center",
        textfont=dict(color=TXT2, size=11),
    ))

    fig.update_layout(
        **CHART_LAYOUT, barmode="group",
        xaxis=dict(tickfont=dict(color=TXT2)),
        yaxis=dict(title=dict(text="人数", font=dict(color=TXT2)),
                   showgrid=True, gridcolor=GRID, tickfont=dict(color=TXT3)),
        yaxis2=dict(title=dict(text="書類通過率(%)", font=dict(color=TXT2)),
                    overlaying="y", side="right", range=[0, 100], showgrid=False,
                    tickfont=dict(color=TXT2)),
        legend=dict(orientation="h", y=-0.2, x=0.15, font=dict(size=12, color=TXT2)),
        margin=dict(l=10, r=10, t=10, b=40), height=320,
    )
    return fig


def make_channel_scatter(ch_eff: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ch_eff["応募数"], y=ch_eff["書類通過率"] * 100,
        mode="markers+text", text=ch_eff["チャネル"],
        textposition="top center", textfont=dict(size=10, color=TXT2),
        marker=dict(
            size=ch_eff["書類通過"].clip(lower=3) * 2.5 + 8,
            color=ch_eff["書類通過率"] * 100,
            colorscale=[[0, C900], [0.5, C500], [1.0, C200]],
            showscale=True, colorbar=dict(
                title=dict(text="通過率%", font=dict(color=TXT3)),
                tickfont=dict(color=TXT3)),
            line=dict(width=1, color="rgba(255,255,255,0.15)"),
        ),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(title=dict(text="応募数", font=dict(color=TXT2)),
                   showgrid=True, gridcolor=GRID, tickfont=dict(color=TXT3)),
        yaxis=dict(title=dict(text="書類通過率 (%)", font=dict(color=TXT2)),
                   showgrid=True, gridcolor=GRID, tickfont=dict(color=TXT3)),
        margin=dict(l=10, r=10, t=10, b=10), height=400,
    )
    return fig


def make_breakdown_chart(df: pd.DataFrame, group_col: str, top_n: int = 12) -> go.Figure:
    grouped = df.groupby(group_col).agg(
        応募=("_f_応募", "sum"), 書類通過=("_f_書類通過", "sum"), 内定承諾=("_f_内定承諾", "sum"),
    ).sort_values("応募", ascending=False).head(top_n)

    fig = go.Figure()
    for col_name, color in [("応募", C500), ("書類通過", C700), ("内定承諾", C300)]:
        fig.add_trace(go.Bar(
            y=grouped.index, x=grouped[col_name], orientation="h",
            name=col_name, marker_color=color, opacity=0.85,
            text=grouped[col_name] if col_name == "応募" else None,
            textposition="auto", textfont=dict(color="white"),
        ))
    fig.update_layout(
        **CHART_LAYOUT, barmode="group",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color=TXT2)),
        xaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(color=TXT3)),
        legend=dict(orientation="h", y=-0.15, x=0.15, font=dict(size=11, color=TXT2)),
        margin=dict(l=10, r=20, t=10, b=40), height=max(280, top_n * 32),
    )
    return fig


def make_active_pipeline(df: pd.DataFrame) -> dict:
    active = df[df["_result"] == "選考中"]
    pipeline = {}
    flag_list = [f"_f_{s}" for s in STAGES]
    for i, stage in enumerate(STAGES[:-1]):
        at_stage = active[(active[flag_list[i]] == 1) & (active[flag_list[i + 1]] == 0)]
        pipeline[stage] = len(at_stage)
    return pipeline


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
st.markdown('<h1 style="font-weight:600;letter-spacing:-0.5px;">Hiring Funnel Dashboard</h1>',
            unsafe_allow_html=True)
st.caption("Talentio エクスポートCSVをアップロードしてください")

with st.sidebar:
    st.header("📁 データアップロード")
    uploaded = st.file_uploader("CSVファイルを選択", type=["csv"],
                                help="Talentioの「候補者一覧」からエクスポートしたCSVファイル")
    encoding = st.selectbox("文字コード", ["cp932", "utf-8", "shift_jis"], index=0)
    st.divider()
    st.header("🔍 フィルタ")
    view_mode = st.radio("表示粒度", ["全社", "部署別", "ポジション別", "応募経路別"], index=0)

if uploaded is None:
    st.info("👈 サイドバーからCSVファイルをアップロードしてください")
    st.markdown("""
    ### 使い方
    1. Talentioから候補者データをCSVエクスポート
    2. 左のサイドバーからCSVをアップロード
    3. フィルタで表示粒度を切り替え
    """)
    st.stop()

df = load_and_process(uploaded.getvalue(), encoding)

filter_value = None
if view_mode == "部署別":
    options = sorted(df["_dept"].unique())
    filter_value = st.sidebar.selectbox("部署を選択", ["すべて"] + options)
elif view_mode == "ポジション別":
    options = sorted(df["_pos"].unique())
    filter_value = st.sidebar.selectbox("ポジションを選択", ["すべて"] + options)
elif view_mode == "応募経路別":
    options = sorted(df["_channel"].unique())
    filter_value = st.sidebar.selectbox("応募経路を選択", ["すべて"] + options)

df_filtered = df.copy()
filter_label = "全社"
if view_mode == "部署別" and filter_value and filter_value != "すべて":
    df_filtered = df[df["_dept"] == filter_value]
    filter_label = filter_value
elif view_mode == "ポジション別" and filter_value and filter_value != "すべて":
    df_filtered = df[df["_pos"] == filter_value]
    filter_label = filter_value
elif view_mode == "応募経路別" and filter_value and filter_value != "すべて":
    df_filtered = df[df["_channel"] == filter_value]
    filter_label = filter_value

funnel = compute_funnel(df_filtered)
if funnel is None:
    st.warning("該当するデータがありません")
    st.stop()

lead_times = compute_lead_times(df_filtered)
weekly = compute_weekly_trend(df_filtered)
ch_eff = compute_channel_efficiency(df_filtered)
agent_perf = compute_agent_performance(df_filtered)

# =====================================================================
# SECTION 1: KPI Cards
# =====================================================================
st.markdown(f'<div class="section-header">■ サマリー KPI — {filter_label}（{funnel["total"]:,}名）</div>',
            unsafe_allow_html=True)

kpi_cols = st.columns(8)
app_count = funnel["counts"]["応募"]
hire_count = funnel["counts"]["内定承諾"]
offer_count = funnel["counts"]["内定"]

for i, stage in enumerate(STAGES):
    c = funnel["counts"][stage]
    pct = c / app_count * 100 if app_count else 0
    kpi_cols[i].metric(label=stage, value=f"{c:,}",
                       delta=f"対応募 {pct:.1f}%" if i > 0 else f"{c:,}名")

yield_val = app_count / hire_count if hire_count > 0 else float("inf")
yield_display = f"{yield_val:.0f}:1" if yield_val != float("inf") else "—"
kpi_cols[6].metric(label="採用効率", value=yield_display,
                   delta="応募/内定承諾" if hire_count > 0 else "内定承諾0")

offer_accept = hire_count / offer_count * 100 if offer_count > 0 else 0
kpi_cols[7].metric(label="内定承諾率", value=f"{offer_accept:.0f}%",
                   delta=f"{hire_count}/{offer_count}" if offer_count > 0 else "—")

pipeline = make_active_pipeline(df_filtered)
if any(v > 0 for v in pipeline.values()):
    st.caption("選考中パイプライン")
    pcols = st.columns(len(pipeline))
    pipe_bg = [C100, C200, C300, C400, C500]
    pipe_fg = [C900, C900, C900, "white", "white"]
    for i, (stage, count) in enumerate(pipeline.items()):
        pcols[i].markdown(
            f"""<div style="background:{pipe_bg[i]};color:{pipe_fg[i]};border-radius:8px;
            padding:8px;text-align:center;">
            <div style="font-size:0.7rem;opacity:0.8;">{stage}</div>
            <div style="font-size:1.4rem;font-weight:bold;">{count}</div>
            </div>""", unsafe_allow_html=True)

# =====================================================================
# SECTION 2: Bottleneck Alerts
# =====================================================================
alerts = detect_alerts(funnel, lead_times, ch_eff, agent_perf)
if alerts:
    st.markdown('<div class="section-header">■ ボトルネック検知・アクション提案</div>',
                unsafe_allow_html=True)
    for a in alerts:
        st.markdown(f'<div class="alert-{a["level"]}">{a["text"]}</div>', unsafe_allow_html=True)

# =====================================================================
# SECTION 3: Funnel Analysis
# =====================================================================
st.markdown('<div class="section-header">■ ファネル分析 — 確定ベース（選考中を母数から除外）</div>',
            unsafe_allow_html=True)
col_f1, col_d1 = st.columns([1.3, 1])
with col_f1:
    st.plotly_chart(make_funnel_chart(funnel, confirmed_mode=True), use_container_width=True)
with col_d1:
    st.caption("離脱内訳（drop理由の内訳）")
    st.plotly_chart(make_drop_bars(funnel, confirmed_mode=True), use_container_width=True)

st.markdown('<div class="section-header">■ ファネル分析 — 全件ベース（選考中を含む）</div>',
            unsafe_allow_html=True)
col_f2, col_d2 = st.columns([1.3, 1])
with col_f2:
    st.plotly_chart(make_funnel_chart(funnel, confirmed_mode=False), use_container_width=True)
with col_d2:
    st.caption("離脱内訳（drop理由の内訳）")
    st.plotly_chart(make_drop_bars(funnel, confirmed_mode=False), use_container_width=True)

with st.expander("📊 転換率テーブル（詳細数値）"):
    trans_data = []
    for label, _, _ in TRANSITIONS:
        t = funnel["transitions"][label]
        trans_data.append({
            "区間": label,
            "母数(確定)": t["confirmed_total"],
            "確定通過率": f"{t['confirmed_pass_rate']:.1%}",
            "確定お断り率": f"{t['confirmed_reject_rate']:.1%}",
            "確定辞退率": f"{t['confirmed_withdraw_rate']:.1%}",
            "母数(全件)": t["total_entered"],
            "全件通過率": f"{t['pass_rate']:.1%}",
            "選考中": t["in_progress"],
        })
    st.dataframe(pd.DataFrame(trans_data).set_index("区間"),
                 use_container_width=True, height=220)

# =====================================================================
# SECTION 4: Lead Time + Weekly Trend (side by side)
# =====================================================================
if lead_times or len(weekly) > 0:
    col_lt, col_wk = st.columns(2)

    if lead_times:
        with col_lt:
            st.markdown('<div class="section-header">■ リードタイム（選考スピード）</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_lead_time_chart(lead_times), use_container_width=True)
            lt_text = " / ".join([f"{lt['label']}: 中央値 **{lt['median']:.0f}日**" for lt in lead_times])
            st.caption(lt_text)

    if len(weekly) > 0:
        with col_wk:
            st.markdown('<div class="section-header">■ 週次トレンド</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_weekly_chart(weekly), use_container_width=True)
            if len(weekly) >= 2:
                diff = int(weekly["応募"].iloc[-1] - weekly["応募"].iloc[-2])
                sign = "+" if diff > 0 else ""
                st.caption(f"直近週: 応募 {int(weekly['応募'].iloc[-1])}名（前週比 {sign}{diff}）")

# =====================================================================
# SECTION 5: Channel Efficiency + Agent Quality
# =====================================================================
if len(ch_eff) > 0 or len(agent_perf) > 0:
    col_ch, col_ag = st.columns(2)

    if len(ch_eff) > 0:
        with col_ch:
            st.markdown('<div class="section-header">■ チャネル効率マップ（量 × 質）</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_channel_scatter(ch_eff), use_container_width=True)

    if len(agent_perf) > 0:
        with col_ag:
            st.markdown('<div class="section-header">■ エージェント品質ランキング</div>',
                        unsafe_allow_html=True)
            disp = agent_perf.copy()
            disp["書類通過率"] = disp["書類通過率"].apply(lambda x: f"{x:.0%}")
            st.dataframe(disp.set_index("エージェント"), use_container_width=True,
                         height=min(500, len(disp) * 38 + 40))

# =====================================================================
# SECTION 6: Breakdown
# =====================================================================
if view_mode == "全社" or filter_value == "すべて":
    group_col = {
        "全社": None, "部署別": "_dept", "ポジション別": "_pos", "応募経路別": "_channel",
    }[view_mode]

    if group_col:
        st.markdown(f'<div class="section-header">■ {view_mode}ブレイクダウン</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.plotly_chart(make_breakdown_chart(df_filtered, group_col), use_container_width=True)
        with c2:
            rows = []
            for name, grp in df_filtered.groupby(group_col):
                f = compute_funnel(grp)
                if f:
                    row = {"名称": name, "応募": f["counts"]["応募"]}
                    for s in STAGES[1:]:
                        row[s] = f["counts"][s]
                    t0 = f["transitions"][TRANSITIONS[0][0]]
                    row["書類通過率"] = f"{t0['pass_rate']:.0%}"
                    row["お断り率"] = f"{t0['reject_rate']:.0%}"
                    rows.append(row)
            bdf = pd.DataFrame(rows).sort_values("応募", ascending=False)
            st.dataframe(bdf.set_index("名称"), use_container_width=True,
                         height=min(600, len(bdf) * 38 + 40))
    else:
        st.markdown('<div class="section-header">■ 部署別・応募経路別 概要</div>',
                    unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🏢 部署別", "🔗 応募経路別"])
        with tab1:
            st.plotly_chart(make_breakdown_chart(df, "_dept"), use_container_width=True)
        with tab2:
            st.plotly_chart(make_breakdown_chart(df, "_channel"), use_container_width=True)

st.divider()
st.caption(f"データ件数: {len(df_filtered):,}名 / 全{len(df):,}名　|　表示粒度: {view_mode}")
