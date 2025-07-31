import os.path as path
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import wfdb
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Tuple,Set
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_CSV = "origin_data_csv"
MATCH = "Z:"
LOGS = "logs"

MAX_WORKERS = os.cpu_count() * 4  # I/O bound；可自行調

if os.path.isdir(LOGS):
    print(f"{LOGS} folder exist.")
else:
    os.makedirs("logs", exist_ok=True)
    print(f"{LOGS} folder doesn't exist, creating new {LOGS}")

# 建立 Logger
try:
    logger = logging.getLogger("data_clean")
    logger.setLevel(logging.WARNING)  # WARNING 以上都會被記錄

    # 建立 FileHandler，寫入 logs/clean.log
    fh = logging.FileHandler(f"{LOGS}/clean.log", mode="w", encoding="utf-8")
    # 只輸出訊息本身：SUBJECT_ID REASON
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)

    # 避免重複加入 handler
    if not logger.handlers:
        logger.addHandler(fh)
    print("Logger module create success.")
except Exception as e:
    raise ValueError(e)

def record_log(subject_id: str, reason: str):
    """
    將不合格的資料記錄到 logs/clean.log。
    例如： logger.warning("12345 invalid_date")
    """
    logger.warning(f"{subject_id} {reason}")



# def main(base: pd.DataFrame, record: pd.DataFrame):
#     # ── 1. 只留在 base & record 都存在的 SUBJECT_ID ──
#     valid_ids = set(base['SUBJECT_ID'])
#     rec_df = record.loc[record['SUBJECT_ID'].isin(valid_ids)].copy()

#     # ── 2. 拆 PATH → prefix / folder ──
#     rec_df['prefix'] = rec_df['PATH'].map(lambda x:x.split("/")[0])
#     rec_df['folder'] = rec_df['PATH'].map(lambda x:x.split("/")[1])
#     # rec_df= rec_df.drop(['PATH'],axis=1)
    
#     # rec_df[['prefix', 'folder']] = rec_df['PATH'].str.split('/', n=2, expand=True)
#     rec_df.drop(columns='PATH', inplace=True)

#     results = []

#     for row in tqdm(rec_df.itertuples(index=False), desc="Subject", unit=" id"):
#         rec_dir = os.path.join(MATCH, row.prefix, row.folder)   # Z:/pXX/pXXXXXX
        
#         if not os.path.isdir(rec_dir):
#             # print(f"{row.SUBJECT_ID}: 資料夾不存在")
#             record_log(row.SUBJECT_ID,f"{rec_dir}: 資料夾不存在")
#             continue

#         # 只取非 n 結尾的 .hea
        
#         headers = [x.removesuffix(".hea") for x in os.listdir(rec_dir) if row.folder in x and not x.split(".")[0].endswith("n")]

#         if not headers:
            
#             record_log(row.SUBJECT_ID," 無符合 header")
#             continue
        
#         for hdr_name in headers:
#             hdr_path = os.path.join(rec_dir, hdr_name)
#             hdr      = wfdb.rdheader(hdr_path, rd_segments=True)
#             base_dt  = parse_base_datetime(hdr)
#             if base_dt is None:
                
#                 record_log(row.SUBJECT_ID,f"{hdr_path}: base_datetime 解析錯誤")
#                 continue

#             seg_info = segment_info(hdr, rec_dir)
        

#             t1, t1_lead2, total_l2 = seg_info
#             results.append({
#                 "SUBJECT_ID": row.SUBJECT_ID,
#                 "PREFIX"    : row.prefix,
#                 "FOLDER"    : row.folder,
#                 "HEADER"    : hdr_name,
#                 "T0"        : base_dt,
#                 "T1"        : t1,
#                 "T1_lead2"  : t1_lead2,
#                 "Total_lead2_sec": total_l2
#             })

#     pd.DataFrame(results).to_csv(os.path.join(LOGS, "ecg_match_info.csv"), index=False)

"""Performance Update"""
def main(base: pd.DataFrame, record: pd.DataFrame):
    # ── 1. 篩 ID + 拆 prefix / folder (一次向量化) ──
    rec_df = (record
              .loc[record['SUBJECT_ID'].isin(base['SUBJECT_ID'])]
              .copy())
    
    rec_df['prefix'] = rec_df['PATH'].map(lambda x:x.split("/")[0])
    rec_df['folder'] = rec_df['PATH'].map(lambda x:x.split("/")[1])
    rec_df= rec_df.drop(['PATH'],axis=1)

    # ── 2. 依 rec_dir 分組併行處理 ──
    tasks, results = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        for row in rec_df.itertuples(index=False):
            rec_dir = Path(MATCH) / row.prefix / row.folder
            tasks.append(exe.submit(process_subject, row, rec_dir))
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Subject"):
            res = fut.result()
            if res:                       # 可能回傳空 list
                results.extend(res)

    pd.DataFrame(results).to_csv(os.path.join(LOGS, "ecg_match_info.csv"), index=False)

# ──────────────────────────────────────────────────────────────
def process_subject(row, rec_dir: Path) -> list[dict]:
    """處理單一 SUBJECT_ID，回傳多筆 header 資訊（可能為空）"""
    if not rec_dir.is_dir():
        record_log(row.SUBJECT_ID, f"{rec_dir}: 資料夾不存在")
        return []

    # 列出 .hea 且不以 n 結尾
    headers = [f.stem for f in rec_dir.glob("*.hea") if not f.stem.endswith("n")]
    if not headers:
        record_log(row.SUBJECT_ID, "無符合 header")
        return []

    out = []
    for hname in headers:
        hdr_path = rec_dir / hname
        hdr = wfdb.rdheader(str(hdr_path), rd_segments=True)
        base_dt = parse_base_datetime(hdr)
        if base_dt is None:
            record_log(row.SUBJECT_ID, f"{hdr_path}: base_datetime 解析錯誤")
            continue

        seg_res = segment_info(hdr, rec_dir)   # tuple 或 None
        if seg_res is None:
            continue
        t1, t1_l2, total_l2 = seg_res

        out.append({
            "SUBJECT_ID": row.SUBJECT_ID,
            "PREFIX"    : row.prefix,
            "FOLDER"    : row.folder,
            "HEADER"    : hname,
            "T0"        : base_dt,
            "T1"        : t1,
            "T1_lead2"  : t1_l2,
            "Total_lead2_sec": total_l2
        })
    return out

            

# ===================== Helper Function ===================== 
def list_headers(rec_dir: Path) -> list[str]:
    """回傳該資料夾所有合法 .hea（不以 n 結尾，去掉副檔名）"""
    return [
        f.stem                    # 去 .hea
        for f in rec_dir.glob("*.hea")
        if not f.stem.endswith("n")
    ]

def parse_base_datetime(hdr):
    """
    解析 WFDB header 的起始日期時間
    - 若無法解析，回傳 None（由上層決定是否捨棄）
    """
    # v4.x 直接有 datetime 物件
    if getattr(hdr, "base_datetime", None):
        return hdr.base_datetime

    # ↓ 以下為舊版欄位，先確認存在
    if hdr.base_date is None or hdr.base_time is None:
        return None  # <-- 關鍵：缺欄位直接放棄

    # 解析 base_date
    if isinstance(hdr.base_date, date):
        bdate = hdr.base_date
    else:
        try:
            bdate = datetime.strptime(hdr.base_date, "%d/%m/%Y").date()
        except Exception:
            return None

    # 解析 base_time
    if isinstance(hdr.base_time, time):
        btime = hdr.base_time
    else:
        try:
            btime = datetime.strptime(hdr.base_time.split(".")[0], "%H:%M:%S").time()
        except Exception:
            return None

    return datetime.combine(bdate, btime)

# def segment_info(hdr, rec_dir: str) -> Optional[Tuple[datetime,datetime, float]]:
#     """
#     計算:
#     - T1: 最後一個 segment 結束的時間作為整段ECG訊號的結束
#     - T1_lead2: 最後一個含 Lead II 的 segment 的結束作為整段ECG訊號的結束
#     - Total_lead2_sec: 每一個 含有 Lead II segment 的 ECG 時間長度(in sec)

#     Args:
#     - hdr: wfdb.rdheader() 回傳的 MultiRecordHeader 物件
#     - rec_dir: 該 header 檔案所在資料夾 (不含 .hea)

#     Return
#     - (T1, T1_lead2 ,total_lead2_sec): 
#     """
#     base_dt = parse_base_datetime(hdr)


#     seg_names: List[str] = hdr.seg_name
#     seg_lens:  List[int] = hdr.seg_len
#     n = len(seg_names)

#     # --- 取樣率列表 ---------------------------------------------------------
#     if hasattr(hdr, "seg_fs") and hdr.seg_fs:          # WFDB ≥4.1
#         seg_fs = np.array(hdr.seg_fs, dtype=float)
#     else:                                              # 舊版需逐段讀
#         seg_fs = np.empty(n, dtype=float)
#         for i, name in enumerate(seg_names):
#             if name == "~":           # gap 段隨便給 1, 不會被用到
#                 seg_fs[i] = 1.0
#             else:
#                 sub_hdr = wfdb.rdheader(os.path.join(rec_dir, name))
#                 seg_fs[i] = float(sub_hdr.fs)

#     # --- 各段起迄時間 -------------------------------------------------------
#     start_idx = np.insert(np.cumsum(seg_lens[:-1]), 0, 0)
#     start_sec = start_idx / seg_fs
#     seg_sec   = np.array(seg_lens) / seg_fs

#     last_end:        datetime | None = None
#     last_lead2_end:  datetime | None = None
#     total_lead2_sec: float           = 0.0

#     for i in range(n - 1, -1, -1):                    # 倒序掃描
#         name = seg_names[i]
#         if name == "~":
#             continue

#         seg_path = os.path.join(rec_dir, name)
#         sub_hdr  = wfdb.rdheader(seg_path)

#         seg_start_abs = base_dt + timedelta(seconds=float(start_sec[i]))
#         seg_end_abs   = seg_start_abs + timedelta(seconds=float(seg_sec[i]))

#         if last_end is None:
#             last_end = seg_end_abs                     # 整段最後結束

#         # 是否含 Lead II
#         if any(ch.upper().replace(" ", "") == "II" for ch in sub_hdr.sig_name):
#             total_lead2_sec += float(seg_sec[i])
#             if last_lead2_end is None:
#                 last_lead2_end = seg_end_abs           # 最後含 II 的結束

#         # 若兩者都找到且無需再累積秒數，可早停
#         if last_end is not None and last_lead2_end is not None and total_lead2_sec:
#             pass  # 仍要往前加總 Lead II 秒數，所以不提前 break

#     if last_lead2_end is None:        # 若整段都沒 Lead II
#         last_lead2_end = None

#     return last_end, last_lead2_end, total_lead2_sec

"""Performance Update"""
def segment_info(hdr, rec_dir: Path):
    base_dt = parse_base_datetime(hdr)
    seg_names, seg_lens = hdr.seg_name, hdr.seg_len
    n = len(seg_names)

    # 先一次把子段 header 讀好 → 避免之後重複 I/O
    sub_hdrs = [None] * n
    seg_fs   = np.empty(n, dtype=float)

    for i, name in enumerate(seg_names):
        if name == "~":
            seg_fs[i] = 1.0
            continue
        h = wfdb.rdheader(str(rec_dir / name))
        sub_hdrs[i] = h
        seg_fs[i]   = float(getattr(h, "fs", hdr.fs))   # 保底 hdr.fs

    start_idx = np.insert(np.cumsum(seg_lens[:-1]), 0, 0)
    start_sec = start_idx / seg_fs
    seg_sec   = np.array(seg_lens) / seg_fs

    last_end = last_l2_end = None
    total_l2_sec = 0.0

    for i in range(n - 1, -1, -1):
        if seg_names[i] == "~":
            continue
        seg_start_abs = base_dt + timedelta(seconds=float(start_sec[i]))
        seg_end_abs   = seg_start_abs + timedelta(seconds=float(seg_sec[i]))

        if last_end is None:
            last_end = seg_end_abs

        hdr_i = sub_hdrs[i]
        if hdr_i and any(ch.upper().replace(" ", "") == "II" for ch in hdr_i.sig_name):
            total_l2_sec += float(seg_sec[i])
            if last_l2_end is None:
                last_l2_end = seg_end_abs

    return last_end, last_l2_end or None, total_l2_sec

if __name__=="__main__":
    record = pd.read_csv(path.join(DATA_CSV,"RECORDS.csv"))
    base = pd.read_csv(path.join(LOGS,"base.csv"))

    main(base,record)
    # hdr = wfdb.rdheader("Z:\p09\p095396\p095396-2142-11-06-12-12",rd_segments=True)
    # print(segment_info(hdr,"Z:\p09\p095396"))