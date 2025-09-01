import os, json, glob
from typing import List, Dict, Tuple

def build_timewindow_table_markdown(
    base_dir: str,
    k: int,
    caliper: str,
    p_threshold: str,
    time_segs: Tuple[str, ...] = ("1_3", "4_6", "7_9"),
    models: Tuple[str, ...] = ("SVM", "LR", "XGB"),
    float_ndigits: int = 3,
) -> str:
    """
    讀取同一組 (k, caliper, p_threshold) 在多個 time_segs、不同 models 的結果 JSON，
    擷取 codeB_metrics，輸出成 Markdown 表格（列=模型；欄=各時間窗的 AUC/Recall/Acc./F1/Spec/PPV）。

    JSON 檔名模式：
        k_{k}_caliper_{caliper}_pthreshold_{p_threshold}_{model}_{time}.json

    參數
    - base_dir     : JSON 檔所在資料夾
    - k            : k
    - caliper      : 以檔名實際字串傳入（避免 0.3 / 0.30 浮點格式差異）
    - p_threshold  : 同上，建議以字串傳入（如 "0.01"）
    - time_segs    : 預設 ("1_3","4_6","7_9")
    - models       : 預設 ("SVM","LR","XGB")
    - float_ndigits: 小數位數（預設 3）

    回傳
    - Markdown 表格（字串）
    """
    # 欄位順序與顯示名稱（對應你圖片：AUC, Recall, Acc., F1, Spec, PPV）
    metric_keys = ["auc", "sensitivity", "accuracy", "f1", "specificity", "precision"]
    metric_labels = ["AUC", "Recall", "Acc.", "F1", "Spec", "PPV"]

    # 時間窗抬頭（可依需求調整）
    time_titles = {
        "1_3": "Last 1–3h",
        "4_6": "Last 4–6h",
        "7_9": "Last 7–9h",
    }

    def _fmt(x):
        try:
            return f"{float(x):.{float_ndigits}f}"
        except Exception:
            return "—"

    # 蒐集資料：[(model, time)] -> {metric_key: value}
    data: Dict[Tuple[str, str], Dict[str, float]] = {}

    for model in models:
        for t in time_segs:
            # 允許用萬用字元 + 子字串嚴格篩（處理 0.3 vs 0.30）
            patt = os.path.join(base_dir, f"k_{k}_caliper_{caliper}_pthreshold_{p_threshold}_{model}_{t}_{model}.json")
            candidates = glob.glob(patt)
            sub_c = f"caliper_{caliper}"
            sub_p = f"pthreshold_{p_threshold}"
            paths = [p for p in candidates if sub_c in os.path.basename(p) and sub_p in os.path.basename(p)]

            if not paths:
                print(f"Warning: No matching file for {patt}")
                exact = os.path.join(base_dir, f"k_{k}_caliper_{caliper}_pthreshold_{p_threshold}_{model}_{t}_{model}.json")
                if os.path.exists(exact):
                    paths = [exact]

            metrics_dict = {}
            if paths:
                try:
                    with open(paths[0], "r", encoding="utf-8") as f:
                        js = json.load(f)
                    cb = js.get("codeB_metrics", {}) or {}
                    for k_ in metric_keys:
                        metrics_dict[k_] = cb.get(k_)
                except Exception:
                    metrics_dict = {}
            data[(model, t)] = metrics_dict

    # === 組 Markdown 表（兩層表頭以模擬群組列） ===
    # 第一列：Model | Last 1–3h | | | | | | Last 4–6h | | | | | | Last 7–9h | | | | | |
    header_top = ["Model"]
    for t in time_segs:
        header_top += [time_titles.get(t, t)] + [""] * (len(metric_labels) - 1)

    # 第二列：|    | AUC | Recall | Acc. | F1 | Spec | PPV | AUC | ... |
    header_mid = [""]
    for _t in time_segs:
        header_mid += metric_labels

    # 分隔線
    sep = "|" + " --- |" * (1 + len(metric_labels) * len(time_segs))

    # 資料列
    rows = []
    for model in models:
        row = [model]
        for t in time_segs:
            md = data.get((model, t), {})
            for key in metric_keys:
                row.append(_fmt(md.get(key)))
        rows.append(row)

    # 串成 Markdown
    def _join_row(cols: List[str]) -> str:
        return "| " + " | ".join(cols) + " |"

    lines = []
    lines.append(_join_row(header_top))
    lines.append(sep)
    lines.append(_join_row(header_mid))
    lines.append(sep)
    for r in rows:
        lines.append(_join_row(r))

    return "\n".join(lines)

if __name__ == "__main__":
    # 測試用：請根據實際路徑和參數調整
    TRAIN = "train_result"
    k = 2
    caliper = "0.1"
    p_threshold = "0.03"
    print(build_timewindow_table_markdown(TRAIN, k, caliper, p_threshold))
