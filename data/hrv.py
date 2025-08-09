import os
import glob
import wfdb
import numpy as np
import neurokit2 as nk
import pandas as pd
from collections import Counter
from scipy.signal import iirnotch, filtfilt
from typing import Tuple, List, Optional

LOGS = "logs"
MATCH = 'Z:'
from typing import List, Optional
import numpy as np

_powerline_warning_printed = False

    
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
        print(f"{subject_id} 's siganl is not qulified")
        return
    
    # 5.計算實際擷取訊號的時間段
    sample_regions = get_sampling_regions(patient['TIME_DIFFERENCE_SEC'],durations)

    # 6. 針對每個時間端做長度檢測，並計算 hrv
    time_label = ['1_3','4_6','7_9']
    total_valid_hrv_metric_dicts = {}
    for idx, (start_sec, end_sec) in enumerate(sample_regions):
        valid_hrv_metric_dicts = []
        # 6.1 檢查該時間段是否存在
        if start_sec is None or end_sec is None:
            print(f"{subject_id}: 在時間段 {time_label[idx]} 沒有足夠 signal (少於 5400s)")
            total_valid_hrv_metric_dicts[idx] = valid_hrv_metric_dicts
            continue

        signals = extract_time_region(segments,durations,fs,start_sec,end_sec) #1D np.ndarray

        # 6.2 檢查實際時間擷取的 lead 2 signals 是否真的有充足時間
        if signals is None or len(signals)/fs < 5400:
            print(f"{subject_id}: 在時間段 {time_label[idx]} 沒有足夠多真實 lead 2 signal (少於 5400s)") 
            total_valid_hrv_metric_dicts[idx] = valid_hrv_metric_dicts
            continue

        # 6.3 逐段(5 min, 300s) 計算 HRV
        cum = start_sec
        epoch_len  = 300 #5min 300s
        segments_count = -1

        while(cum<= end_sec):
            rel = (cum - start_sec)
            cum += epoch_len
            segments_count +=1

            # 6.3.1 計算 epocch 訊號起點與終點 idx
            start_idx = int(round(rel*fs))
            end_idx = min(int(round((rel+epoch_len) * fs)),len(signals)) # 避免結束超出時間段
            epoch_signals = signals[start_idx:end_idx]

            if start_idx >= len(signals):
                break


            # 6.3.2 過濾 signal 與計算 RR peaks
            try:
                filtered_signals = apply_filters(epoch_signals,fs)
                r_peaks_seires, rpeaks_info = nk.ecg_peaks(filtered_signals, sampling_rate=fs)
                rpeaks = np.asarray(rpeaks_info.get("ECG_R_Peaks", []), dtype=int)
                # print(f"info : {type(rpeaks_info)}: {rpeaks_info}")
                # print(f"series: {type(r_peaks_seires)}: {r_peaks_seires}")
            except Exception as e:
                print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段計算 R peak 失敗: {e}")
                continue
            
            # 6.3.3 計算 RR intervals
            if rpeaks.size < 2:
                print(f"有效 R peaks 過少無法計算 RR interval (at least 2)")
                continue

            rr_intervals = np.diff(rpeaks) / fs  # seconds
            if len(rr_intervals) <250:
                print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段有效 RR intervals 小於 250")
                continue

            # 6.3.4 計算 HRV indicators
            try:
                hrv_metrics = nk.hrv(rpeaks, sampling_rate=fs, show=False)
            except Exception as e:
                print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段計算 HRV 失效")
                continue

            # 解讀 hrv_metrics 並轉換為 dict 格式 (根據neurokit2 版本回傳可能是 dict or pd.DataFrame)
            if isinstance(hrv_metrics, pd.DataFrame):
                hrv_metrics_dict = hrv_metrics.iloc[0].to_dict()
            elif isinstance(hrv_metrics, dict):
                hrv_metrics_dict = hrv_metrics
            else:
                hrv_metrics_dict = {}
            
            if not hrv_metrics_dict:
                print(f"[Error] {subject_id} in time segment {time_label[idx]} 第 {segments_count} 段解析 HRV metrics dict 為空 )")
                continue
            else:
                valid_hrv_metric_dicts.append(hrv_metrics_dict)

        total_valid_hrv_metric_dicts[idx] = valid_hrv_metric_dicts
    
    return total_valid_hrv_metric_dicts

    
    
    


def main(target_path: str):
    """
    Args:
    - df: pd.Dataframe
    """
    os.makedirs(LOGS, exist_ok = True)

    df = pd.read_csv(target_path, usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID',
                                           'INTIME','OUTTIME','ADMITTIME','DISCHTIME',
                                           'DEATHTIME','T0','T1_lead2','Total_lead2_sec',
                                           'PREFIX','FOLDER','HEADER'],
                                           dtype={
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



        
    count = 0
    for _ , patient in df.iterrows():
        result=process_patient(patient)
        print(f"{patient['SUBJECT_ID']} : {result}")
        if count < 3:
            count+=1
        else:
            break
        pass
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
    b, a = iirnotch(freq, Q, fs)
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
    # 先通過頻率濾波
    filtered_signal = nk.signal_filter(signal, sampling_rate=fs, lowcut=0.5,
                                       method="butterworth", order=5)
    # 再做電源雜訊（notch 或 powerline_filter）
    if hasattr(nk, 'powerline_filter'):
        # 如果有 neurokit2.powerline_filter 就用它
        filtered_signal = nk.powerline_filter(filtered_signal, sampling_rate=fs,
                                              method='notch', line_frequency=notch_freq)
    else:
        # 否則就用自定義的 notch
        global _powerline_warning_printed
        if not _powerline_warning_printed:
            print("[WARNING] nk.powerline_filter not found, using custom notch filter.")
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

        try:
            h = wfdb.rdrecord(os.path.join(rec_dir, name)) # including metadata and ecg signal
        except Exception as e:
            print(f"[ERROR] 無法讀取 {os.path.join(rec_dir, name)}: {e}")
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

if __name__=="__main__":
    target_path = os.path.join(LOGS,"surv_stage2_filtered_with_wave_id.csv")
    main(target_path)