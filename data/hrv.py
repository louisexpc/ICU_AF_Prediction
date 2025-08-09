import os
import glob
import wfdb
import numpy as np
import neurokit2 as nk
import pandas as pd
import logging
from collections import Counter
from scipy.signal import iirnotch, filtfilt
from typing import Tuple, List, Optional
from tqdm import tqdm
from functools import lru_cache
# ======================================== Multiprocessing ======================================== 
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from functools import lru_cache

# === 變數定義 ===
LOGS = "logs"
MATCH = 'Z:'
HRV = os.path.join(LOGS,"hrv")
time_label = ['1_3','4_6','7_9']

# === 既有旗標（保留），但新增能力快取與 notch 係數快取 ===
_powerline_warning_printed = False
_NK_HAS_POWERLINE = hasattr(nk, 'powerline_filter')  # 新增：只檢查一次

@lru_cache(maxsize=64)
def _cached_notch_ba(fs: float, freq: float, Q: float):
    # 新增：快取 notch 濾波器係數（完全相同輸入 → 係數唯一）
    return iirnotch(freq, Q, fs)



# === Logger 初始化（改）===
# 原本在 import 階段就建立 FileHandler → 會造成多程序同檔寫入競爭
# 改為：主程序在 main() 裡建立 QueueListener；子程序只綁 QueueHandler
logger = logging.getLogger("data_clean")
logger.setLevel(logging.WARNING)  # 保留原等級；實際 handler 於 main()/worker 設定

def setup_main_logging(logfile: str):
    """
    在主程序啟動一個 QueueListener，統一從 queue 收到的 logs 寫入到 logfile。
    """
    log_queue = mp.Queue(-1)
    file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    return log_queue, listener

def setup_worker_logging(log_queue: mp.Queue):
    """
    在子程序中，把 logger 綁到 QueueHandler，所有 log 丟回主程序寫檔。
    """
    global logger
    # 清掉繼承的 handlers，避免重複
    logger.handlers = []
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)
    logger.setLevel(logging.WARNING)

def record_log(subject_id: str, reason: str):
    """
    將不合格的資料記錄到 logs/clean.log。
    例如： logger.warning("12345 invalid_date")
    """
    logger.warning(f"{subject_id} {reason}")

    
def process_patient(patient: pd.Series):

    # 1. 路徑處理
    subject_id = patient['SUBJECT_ID']
    header_path = os.path.join(MATCH,patient['PREFIX'],patient['FOLDER'],patient['HEADER'])
    rec_dir = os.path.join(MATCH,patient['PREFIX'],patient['FOLDER'])

    # 2. 讀取實際 header records
    try:
        hdr = wfdb.rdheader(header_path, rd_segments = True)
    except Exception as e:
        raise ValueError(f"{header_path}: file not found : {e}")
    
    # 3. 計算該 header 內所有 segments 的 info
    fs, segments, durations = concat_signals(hdr, rec_dir, channel= "II")

    # 4. 該段落 signals 的品質檢測
    if not is_signal_quality_good(segments):
        record_log(subject_id, "siganl is not qulified")
        # print(f"{subject_id} 's siganl is not qulified")
        return
    
    # 5.計算實際擷取訊號的時間段
    sample_regions = get_sampling_regions(patient['TIME_DIFFERENCE_SEC'],durations)

    # 6. 針對每個時間端做長度檢測，並計算 hrv
    
    total_valid_hrv_metric_df = {}
    for idx, (start_sec, end_sec) in enumerate(sample_regions):
        # 原本每 epoch 做一次 concat，改成先收集在 list，最後一次性 concat（結果完全相同）
        hrv_rows = []  # 新增：暫存各 epoch 的 DataFrame
        
        # 6.1 檢查該時間段是否存在
        if start_sec is None or end_sec is None:
            record_log(subject_id, f"在時間段 {time_label[idx]} 沒有足夠 signal (少於 5400s)")
            # print(f"{subject_id}: 在時間段 {time_label[idx]} 沒有足夠 signal (少於 5400s)")
            total_valid_hrv_metric_df[idx] = pd.DataFrame()
            continue

        signals = extract_time_region(segments,durations,fs,start_sec,end_sec) #1D np.ndarray

        # 6.2 檢查實際時間擷取的 lead 2 signals 是否真的有充足時間
        if signals is None or len(signals)/fs < 5400:
            record_log(subject_id, f"在時間段 {time_label[idx]} 沒有足夠多真實 lead 2 signal (少於 5400s)")
            # print(f"{subject_id}: 在時間段 {time_label[idx]} 沒有足夠多真實 lead 2 signal (少於 5400s)") 
            total_valid_hrv_metric_df[idx] = pd.DataFrame()
            continue

        # 6.3 逐段(5 min, 300s) 計算 HRV
        cum = start_sec
        epoch_len  = 300 #5min 300s
        segments_count = -1
        len_signals = len(signals)

        while(cum<= end_sec):
            rel = (cum - start_sec)
            cum += epoch_len
            segments_count +=1

            # 6.3.1 計算 epocch 訊號起點與終點 idx
            start_idx = int(round(rel*fs))
            if start_idx >= len_signals:
                break

            end_idx = min(int(round((rel+epoch_len) * fs)),len(signals)) # 避免結束超出時間段
            epoch_signals = signals[start_idx:end_idx]

            


            # 6.3.2 過濾 signal 與計算 RR peaks
            try:
                filtered_signals = apply_filters(epoch_signals,fs)
                r_peaks_seires, rpeaks_info = nk.ecg_peaks(filtered_signals, sampling_rate=fs)
                rpeaks = np.asarray(rpeaks_info.get("ECG_R_Peaks", []), dtype=int)
                # print(f"info : {type(rpeaks_info)}: {rpeaks_info}")
                # print(f"series: {type(r_peaks_seires)}: {r_peaks_seires}")
            except Exception as e:
                record_log(subject_id,f"in time segment {time_label[idx]} 第 {segments_count} 段 ( {cum - epoch_len} - {cum} ) 計算 R peak 失敗: {e}")
                # print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段計算 R peak 失敗: {e}")
                continue
            
            # 6.3.3 計算 RR intervals
            if rpeaks.size < 2:
                record_log(subject_id, f"有效 R peaks 過少無法計算 RR interval (at least 2, only {rpeaks.size})")
                # print(f"有效 R peaks 過少無法計算 RR interval (at least 2)")
                continue

            rr_intervals = np.diff(rpeaks) / fs  # seconds
            if len(rr_intervals) <250:
                record_log(subject_id, f"in time segment {time_label[idx]} 第 {segments_count} 段 ( {cum-epoch_len} - {cum} ) 有效 RR intervals 小於 250 (only {len(rr_intervals)})")
                # print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段有效 RR intervals 小於 250")
                continue

            # 6.3.4 計算 HRV indicators
            try:
                hrv_metrics = nk.hrv(rpeaks, sampling_rate=fs, show=False)
            except Exception as e:
                record_log(subject_id,f"in time segment {time_label[idx]} 第 {segments_count} 段 ( {cum-epoch_len} - {cum} ) 計算 HRV 失效")
                # print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段計算 HRV 失效")
                continue

            if not isinstance(hrv_metrics,pd.DataFrame):
                raise ValueError(f'hrv metrics not pd.DataFrame, type {type(hrv_metrics)}')
            
            hrv_rows.append(hrv_metrics)

        valid_hrv_metric_df = pd.concat(hrv_rows, axis=0, ignore_index=True) if hrv_rows else pd.DataFrame()
        _to_csv_atomic(valid_hrv_metric_df, os.path.join(HRV,f"{subject_id}_{time_label[idx]}_hrv.csv"))
            # 解讀 hrv_metrics 並轉換為 dict 格式 (根據neurokit2 版本回傳可能是 dict or pd.DataFrame)
            # print(f'HRV metric type {type(hrv_metrics)}')
            # if isinstance(hrv_metrics, pd.DataFrame):
            #     valid_hrv_metric_df = pd.concat([valid_hrv_metric_df,hrv_metrics], axis=0)
            #     hrv_metrics_dict = hrv_metrics.iloc[0].to_dict()
            # elif isinstance(hrv_metrics, dict):
            #     hrv_metrics_dict = hrv_metrics
            # else:
            #     hrv_metrics_dict = {}
            
            # if not hrv_metrics_dict:
            #     print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段解析 HRV metrics dict 為空 )")
            #     continue
            # else:
            #     valid_hrv_metric_dicts.append(hrv_metrics_dict)
            

        total_valid_hrv_metric_df[idx] = valid_hrv_metric_df
    
    return total_valid_hrv_metric_df

    
    
    


def main(target_path: str):
    """
    Args:
    - df: pd.Dataframe
    """
    os.makedirs(LOGS, exist_ok = True)
    os.makedirs(HRV, exist_ok = True)

    df = pd.read_csv(target_path, dtype={
                                    'SUBJECT_ID':int,'HADM_ID':int,'ICUSTAY_ID':int,
                                    'PREFIX':str,'FOLDER':str,'HEADER':str
                                })

    time_cols = ['INTIME','OUTTIME','ADMITTIME','DISCHTIME','DEATHTIME','T0','T1_lead2']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col],errors='coerce')
    
    print(df.info())

    """
    TIME_DIFFERENCE_MIN：就是ecg結束時間跟死亡/離開時間的差距
    - Surv: T1_lead2 - DISCHTIME
    - Mort: T1_lead2 - DEATHTIME
    """
    df['TIME_DIFFERENCE_SEC'] = (
        (df['DEATHTIME'] - df['T1_lead2']).dt.total_seconds()
            .where(df['DEATHTIME'].notna(),
                (df['OUTTIME'] - df['T1_lead2']).dt.total_seconds())
            .abs()
    )
    print(df.info())
    print(df.head())
        

    masks = [[],[],[]] # following the order : 1-3, 4-6, 7-9

    # === 新增：主程序日誌 listener（集中寫檔） ===
    log_queue, log_listener = setup_main_logging(f"{LOGS}/hrv.log")

    try:
        # 轉成 dict 列（序列化成本較小）
        rows = df.to_dict('records')
        workers = max(1, os.cpu_count() or 1)

        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=setup_worker_logging,
                initargs=(log_queue,)
        ) as ex:
            it = ex.map(_worker_process_patient, rows, chunksize=1)
            for have_1_3, have_4_6, have_7_9 in tqdm(
                    it, total=len(rows), desc="HRV per patient", unit="pt"):
                masks[0].append(have_1_3)
                masks[1].append(have_4_6)
                masks[2].append(have_7_9)
    finally:
        # 關閉 listener，確保所有 log 刷寫完成
        log_listener.stop()

    for idx, time in enumerate(time_label):
        df[f"have_{time}_hrv"] = masks[idx]
    
    return df
# ============================= HRV Function =============================
def notch_filter(signal: np.ndarray,
                 fs: float,
                 freq: float = 50.0,
                 Q: float = 30.0) -> np.ndarray:
    """
    對 ECG 訊號施作 notch（帶阻）濾波以移除電源頻率干擾。

    Args:
    - signal (np.ndarray): 一維原始訊號向量。
    - fs (float): 訊號取樣頻率 (Hz)。
    - freq (float, optional): 要移除之電源雜訊中心頻率，預設 50 Hz。
    - Q (float, optional): 品質因數；數值越高帶寬越窄，預設 30。

    Returns
    - np.ndarray: 已濾波之訊號，資料型別與輸入相同。
    """
    b, a = _cached_notch_ba(fs, freq, Q)  # 改：使用快取係數
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_filters(signal: np.ndarray,
                  fs: float,
                  notch_freq: float = 50.0,
                  notch_Q: float = 30.0) -> np.ndarray:
    """
    先以 Butterworth 高通 (0.5 Hz) 去除基線漂移，再以 notch 移除電源雜訊。

    Args:
    - signal (np.ndarray): 一維原始訊號向量。
    - fs (float): 取樣頻率 (Hz)。
    - notch_freq (float, optional): notch 中心頻率，預設 50 Hz。
    - notch_Q (float, optional): notch 品質因數，預設 30。

    Returns
    - np.ndarray: 經兩階段濾波後的訊號。
    """
    filtered_signal = nk.signal_filter(signal, sampling_rate=fs, lowcut=0.5,
                                       method="butterworth", order=5)
    if _NK_HAS_POWERLINE:
        filtered_signal = nk.powerline_filter(filtered_signal, sampling_rate=fs,
                                              method='notch', line_frequency=notch_freq)
    else:
        global _powerline_warning_printed
        if not _powerline_warning_printed:
            # print("[WARNING] nk.powerline_filter not found, using custom notch filter.")
            _powerline_warning_printed = True
        filtered_signal = notch_filter(filtered_signal, fs, freq=notch_freq, Q=notch_Q)
    return filtered_signal

# ============================= Helper Function =============================
def get_sampling_regions(
        time_diff_sec:float,
        durations:List[float]
)->List[Tuple[float,float]]:
    """
    根據離院／死亡時間為基準，回推 1–3、4–6、7–9 小時的區段，並轉換為
    ECG 記錄中的實際時間範圍（秒），供後續切割與 HRV 分析。

    計算邏輯：
    - 以 ECG 結束時間與臨床事件時間的差值（time_diff_min）作為參考點。
    - ECG 結束時間 T_ecg_end 通常早於事件時間 T_event，故 time_diff_min 多為負值，已在前處理轉換為絕對值
    - 欲擷取事件發生前 N 小時的區段，應從 ECG 結束時間往前推 (N*60 - |time_diff_min|) 分鐘。
    - 區段將轉換為 ECG 記錄的相對秒數位置：從 total_length_sec 倒退而得。

    Args:
        - time_diff_sec (float): ECG 結束時間 - 離院／死亡時間，單位為秒(s)。
        - durations (List[float]): 各段的秒數長度（含空段），用於計算 total_length_sec。

    Return:
        Tuple[Tuple[float, float]]: 一組最多三個時間區段（start_sec, end_sec），
                                    單位為秒，對應事件前 1–3、4–6、7–9 小時。
                                    若某段長度不足 1.5 小時（5400 秒）則略過。
    """
    total_length_sec = sum(durations)

    if not np.isfinite(time_diff_sec):
        return [(None,None),(None,None),(None,None)]


    windows_hr = [(1, 3), (4, 6), (7, 9)]
    regions = []

    for start_h, end_h in windows_hr:
        end_sec   = total_length_sec - (start_h * 3600 - time_diff_sec)
        start_sec = total_length_sec - (end_h   * 3600 - time_diff_sec)
        # 修正負索引
        start_sec = max(start_sec, 0.0)
        end_sec   = min(total_length_sec, end_sec)
        # 至少 1.5 小時 (= 5400 s) 才保留
        if end_sec - start_sec >= 5400:
            regions.append((start_sec, end_sec))
        else:
            regions.append((None,None))
    return regions

def is_signal_quality_good(
    segments: List[Optional[np.ndarray]],
    min_points: int = 10,
    std_threshold: float = 0.01
) -> bool:
    """
    以資料點數與標準差粗略評估多段訊號品質。
    
    Args:
      segments: 各段一維訊號陣列，若該段無資料則為 None。
      fs: 取樣頻率 (保留介面，但此函式未使用)。
      min_points: 最少有效點數。
      std_threshold: 標準差門檻。

    Returns:
      若任一段串接後的總訊號通過檢查，回傳 True；否則 False。
    """
    # 1) 將所有有資料的 segments 取出並串起來
    valid_sigs = [seg for seg in segments if seg is not None and seg.size > 0]
    if not valid_sigs:
        # 完全沒有序列可評估
        return False

    full_signal = np.concatenate(valid_sigs)
    
    # 2) 計算有效點數（非 NaN）
    valid_points = np.count_nonzero(~np.isnan(full_signal))
    if valid_points < min_points:
        return False

    # 3) 計算標準差，若太小代表過於平坦
    if np.nanstd(full_signal) < std_threshold:
        return False

    return True

def extract_time_region(
    segments: List[Optional[np.ndarray]],
    durations: List[float],
    fs: float,
    t_start: float,
    t_end: float
) -> Optional[np.ndarray]:
    """
    回傳真實時序內 [t_start, t_end) 的 samples 合併結果。

    Args:
      segments: 各段波形陣列，若該段沒有訊號則為 None。
      durations: 對應每段的秒數長度，若為 0 或負值則視為空段。
      fs: 取樣率 (Hz)。
      t_start: 擷取區段起始時間（秒）。
      t_end:   擷取區段結束時間（秒），開區間。

    Returns:
      - 若有任何 overlap，回傳 concatenated 之 1D np.ndarray；  
      - 若無任何重疊或輸入無效，回傳 None。
    """
    if t_end <= t_start or fs <= 0 or not segments or not durations:
        return None

    out_slices: List[np.ndarray] = []
    cum_time: float = 0.0  # 已過累積秒數

    for seg, dur in zip(segments, durations):
        # Edge Case: 無訊號或長度非正，直接跳過但仍要累積時間（以 dur 計）
        if dur is None or dur <= 0:
            cum_time += max(0.0, dur or 0.0)
            continue

        seg_start = cum_time
        seg_end   = cum_time + dur

      
        # 1) 如果這一整段在目標區段之外，就跳過
        if seg_end <= t_start or seg_start >= t_end:
            cum_time += dur
            continue
        
        # 2) 否則一定有 overlap，計算在此段要抽取的本地區間 [local_start, local_end)
        # local_start: 在本段內對應 t_start 的 offset (不小於 0)
        local_start = max(0.0, t_start - seg_start)
        # local_end:   在本段內對應 t_end 的 offset (不超過 dur)
        local_end   = min(dur,   t_end   - seg_start)

        # 轉成 sample index
        i0 = int(round(local_start * fs))
        i1 = int(round(local_end   * fs))

        # 4) 從該段取出 samples，然後加入輸出清單
        # 如果有真實訊號再擷取
        if seg is not None and i1 > i0:
            out_slices.append(seg[i0:i1])

        cum_time += dur

    if not out_slices:
        return None

    # 串接所有切片
    return np.concatenate(out_slices)



def concat_signals(hdr, rec_dir: str, channel: str="II")->Tuple[float, List, List]:
    """
    依原始 seg_name 與 seg_len，在真實時序中回傳：
     - fs: 最新遇到的取樣率
     - segments: list of 1D numpy arrays (只有指定 channel 的訊號片段或 None)
     - durations: list of each segment 的長度（秒）
    """
    seg_names, seg_lens = hdr.seg_name, hdr.seg_len
    fs_list   = []
    segments  = []
    durations = []

    target = channel.upper().replace("LEAD ", "").strip()

    for name, seg_len in zip(seg_names, seg_lens):
        if name == "~":
            durations.append(seg_len / hdr.fs)
            segments.append(None)
            continue

        if name.split("_")[-1] =="layout":
            continue
        try:
            h = wfdb.rdrecord(os.path.join(rec_dir, name)) # including metadata and ecg signal
        except Exception as e:
            record_log(os.path.split()[-1],f"[ERROR] 無法讀取 {os.path.join(rec_dir, name)}: {e}")
            # print(f"[ERROR] 無法讀取 {os.path.join(rec_dir, name)}: {e}")
            durations.append(seg_len / hdr.fs)
            segments.append(None)
            continue
        
        fs = getattr(h, "fs", hdr.fs)
        fs_list.append(fs)

        # 找出導程索引
        names_norm = [ch.upper().replace("LEAD ", "").strip() for ch in h.sig_name]

        if target in names_norm:
            """
            h.p_signal : numpy.ndarray with (n_samples, n_channels)，每一欄是一個導程的資料
            [:, idx]: 取出所有時間點的第 idx 個導程資料」(單一導程的完整波形)。
            """
            idx = names_norm.index(target)
            sig = h.p_signal[:, idx]
            segments.append(sig)
            durations.append(len(sig) / fs)
        else:
            # 這段沒有目標導程
            durations.append(seg_len / fs)
            segments.append(None)

    # 決定最終 fs（如果多段 fs 不同，採「最後一個非空值」）
    final_fs = hdr.fs
    for fs in fs_list:
        if fs != final_fs:
            print(f"[WARNING] fs 變動: 原 {final_fs}Hz → {fs}Hz，採用新值")
            final_fs = fs

    return final_fs, segments, durations

# === Multiprocessing Function :  原子寫入 CSV（新增）===
def _to_csv_atomic(df: pd.DataFrame, out_path: str):
    """
    安全寫檔：先寫暫存檔，再原子替換，避免平行寫入產生半檔案。
    """
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=out_dir, suffix='.tmp', encoding='utf-8', newline='') as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_name = tmp.name
    os.replace(tmp_name, out_path)

# === Worker 包裝函式（新增）：只回傳 3 個布林，避免把整個 dict/DataFrame pickling 回主程序 ===
def _worker_process_patient(patient_dict: dict) -> Tuple[bool, bool, bool]:
    """
    子程序呼叫：執行 process_patient()，只把各時窗是否有 HRV 結果（非空）回傳。
    減少跨程序傳輸成本，CSV 已在子程序內完成寫入。
    """
    patient = pd.Series(patient_dict)
    result = process_patient(patient)   # {0: df, 1: df, 2: df}
    flags = []
    for i in range(3):
        df_i = result.get(i, pd.DataFrame())
        flags.append(not df_i.empty)
    return tuple(flags)  # (have_1_3, have_4_6, have_7_9)

if __name__=="__main__":
    target_path = os.path.join(LOGS,"mort_stage2_filtered.csv")
    result_df = main(target_path)
    result_df.to_csv(os.path.join(LOGS,"mort_stage2_filtered_hrv.csv"),index=False)