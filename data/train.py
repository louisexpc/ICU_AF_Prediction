# MyFrameWork.py
# ============================================================
# SVM (RBF) – CodeA/CodeB 合併版
# - Stage-1: RandomizedSearchCV（廣搜，保留原參數）
# - Stage-2: GridSearchCV（細搜，保留原參數）
# - Final  : 以眾數超參數全資料重訓並存檔（與 CodeA 一致）
# - EvalB  : 依 CodeB 的極窄格網 + 固定閾值 0.110，做20×Nested CV
#            並以「累加混淆矩陣」產生整體營運指標與 ROC（與 CodeB 一致）
# ============================================================
# 置於檔案最上方、在 import pyplot 之前
import matplotlib
matplotlib.use('Agg')  # 非互動後端，避免 Tk 與平行化衝突

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Tuple, Dict, List, Iterable

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from scipy.stats import loguniform
from xgboost import XGBClassifier

TRAIN = "train_result"

def _prepare_dataset(surv_df: pd.DataFrame, mort_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """
    整併 surv/mort，僅保留 HRV_* 特徵與標籤欄位（HOSPITAL_EXPIRE_FLAG）。
    - 回傳 X(np.ndarray), y(np.ndarray), dataset(pd.DataFrame for檢視)
    """
    combined_df = pd.concat([surv_df, mort_df], axis=0, ignore_index=True)

    # 容錯：標籤欄位名稱兼容
    label_col = "HOSPITAL_EXPIRE_FLAG"
    if label_col not in combined_df.columns:
        raise ValueError("找不到標籤欄位：需包含 'HOSPITAL_EXPIRE_FLAG' 。")

    selected_cols = [c for c in combined_df.columns if c.startswith("HRV_")] + [label_col]
    dataset = combined_df[selected_cols].copy()

    # 與 CodeA/CodeB 一致：處理 inf / NaN
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    if dataset.isna().any().any():
        na_cols = dataset.columns[dataset.isna().any()]
        raise ValueError(f"有缺失值存在，請檢察:\n{dataset[na_cols].isna().sum()}")

    # before = len(dataset)
    # dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    # na_rows = dataset.isna().any(axis=1).sum()
    # dataset.dropna(inplace=True)
    # after = len(dataset)

    print("=== Dataset 檢查 ===")
    # print(f"- 原始筆數：{before}  | 移除含 NaN/Inf 筆數：{na_rows}  | 最終可用：{after}")
    total_feature = len([c for c in dataset.columns if c.startswith('HRV_')])
    print(f"- 特徵數：{len([c for c in dataset.columns if c.startswith('HRV_')])}  | 標籤欄：{label_col}")

    X = dataset.drop(columns=[label_col]).values  # 轉 ndarray（與 CodeA 索引方式一致）
    y = dataset[label_col].values.astype(int)

    return X, y, dataset, total_feature


def _build_pipeline(model: str = "SVM") -> Pipeline:
    """
    建立 SVM(RBF) 縮放 + 分類 Pipeline。
    - balanced=True  : 與 CodeA 一致（class_weight='balanced'）
    - balanced=False : 與 CodeB 一致（不設定 class_weight）
    """
    if model=="SVM":
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm),
        ])
    elif model =="XGB":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(
                eval_metric='logloss'
            ))
        ])


# ------------------------------
# 公用：Nested CV（供 Stage-1/2 使用）
# ------------------------------
def nested_cv(
    X: np.ndarray, y: np.ndarray, pipe: Pipeline, search_space,
    seeds: Iterable[int] = range(10), n_outer: int = 10, n_inner: int = 5,
    use_random: bool = False, n_iter: int = 40
) -> Tuple[float, List[Dict], Tuple[np.ndarray, List[np.ndarray]]]:
    """
    與 CodeA 一致的 Nested CV。
    回傳：
      - mean_auc: 外層 AUC 平均
      - best_params_history: 每個外層 fold 的最佳參數（內層搜尋產出）
      - roc_info: (fpr_mean, tprs_list) 供平均 ROC 繪圖
    """
    best_params_hist, auc_scores = [], []
    fpr_mean = np.linspace(0, 1, 100)
    tprs_list = []

    for seed in seeds:
        outer_kf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

        for tr_idx, te_idx in outer_kf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            inner_kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

            if use_random:
                searcher = RandomizedSearchCV(
                    pipe,
                    param_distributions=search_space,
                    n_iter=n_iter,
                    scoring="roc_auc",
                    cv=inner_kf,
                    n_jobs=-1,
                    random_state=seed,
                )
            else:
                searcher = GridSearchCV(
                    pipe,
                    param_grid=search_space,
                    scoring="roc_auc",
                    cv=inner_kf,
                    n_jobs=-1,
                )

            searcher.fit(X_tr, y_tr)

            # 外層結果
            best_params_hist.append(searcher.best_params_)
            y_prob = searcher.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            auc_scores.append(auc)

            fpr, tpr, _ = roc_curve(y_te, y_prob)
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_list.append(interp_tpr)

    return float(np.mean(auc_scores)), best_params_hist, (fpr_mean, tprs_list)


# ------------------------------
# 兩階段搜尋 + 最終存模
# ------------------------------
def _stage1_random_search(X: np.ndarray, y: np.ndarray,model:str = "SVM") -> Tuple[float, List[Dict], pd.DataFrame]:
    """
    Stage-1（與 CodeA 一致）：
    - seeds=range(5), n_iter=60
    - C ~ loguniform(1e-3, 1e3), gamma ~ loguniform(1e-4, 1e1)
    """
    np.random.seed(42)
    base_pipe_a = _build_pipeline(model=model)

    param_dist_coarse = {
        "svm__C":     loguniform(1e-3, 1e3),
        "svm__gamma": loguniform(1e-4, 1e1),
    }

    print("\n=== Stage-1：RandomizedSearchCV（粗搜尋） ===")
    auc1, best_hist1, _ = nested_cv(
        X, y, base_pipe_a, param_dist_coarse,
        seeds=range(5), use_random=True, n_iter=60
    )
    print(f"Stage-1  mean AUC = {auc1:.4f}")

    stage1_df = pd.DataFrame(best_hist1)
    stage1_summary = stage1_df.value_counts().rename("count")
    print("Stage-1  Top-5 Best (C, gamma)：")
    print(stage1_summary.head())

    return auc1, best_hist1, stage1_df


def _build_fine_grid_from_stage1(stage1_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    與 CodeA 一致：
    - 以 Stage-1 眾數（此處取 median of log）為中心，±1 decade，9點對數等距
    """
    logC_center = np.log10(stage1_df["svm__C"]).median()
    logG_center = np.log10(stage1_df["svm__gamma"]).median()

    C_fine = np.logspace(logC_center - 1, logC_center + 1, 9)
    G_fine = np.logspace(logG_center - 1, logG_center + 1, 9)

    param_grid_fine = {"svm__C": C_fine, "svm__gamma": G_fine}
    print("\nStage-2 搜尋網格（由 Stage-1 推導）：")
    print({k: (float(v.min()), float(v.max()), len(v)) for k, v in param_grid_fine.items()})
    return param_grid_fine


def _stage2_grid_search_and_finalize(
    X: np.ndarray, y: np.ndarray, param_grid_fine: Dict[str, np.ndarray], save_dir: str,time_range: str,model_path: str = "svm_final_model.pkl",model:str="SVM"
) -> Tuple[float, Dict, Tuple[np.ndarray, List[np.ndarray]], str]:
    """
    Stage-2（與 CodeA 一致）：
    - seeds=range(10) 的 Nested CV + GridSearchCV
    - 以眾數參數在全資料重訓並存檔
    - 回傳：Stage-2 mean AUC、final_params、roc_info（供平均 ROC）
    """
    base_pipe_a = _build_pipeline(model = model)

    print("\n=== Stage-2：GridSearchCV（細搜尋） ===")
    auc2, best_hist2, roc_info2 = nested_cv(
        X, y, base_pipe_a, param_grid_fine,
        seeds=range(10), use_random=False
    )
    print(f"Stage-2  mean AUC = {auc2:.4f}")

    stage2_df = pd.DataFrame(best_hist2)
    final_params = stage2_df.mode().iloc[0].to_dict()  # 眾數
    print("最終超參數（眾數） =", final_params)

    final_model = base_pipe_a.set_params(**final_params)
    final_model.fit(X, y)


    print("Stage-2  Best params distribution（Top-5）：")
    print(stage2_df.value_counts().rename("count").head())

    # 儲存 stage2 history
    stage2_csv = os.path.join(save_dir, f"stage2_best_hist_{time_range}.csv")
    stage2_df.to_csv(stage2_csv, index=False)
    print(f"- Stage-2 history saved → {stage2_csv}")

    # 儲存 final_params
    final_params_serializable = {k: (int(v) if np.issubdtype(type(v), np.integer) else float(v)) for k, v in final_params.items()}
    params_path = os.path.join(save_dir, f"final_params_{time_range}.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(final_params_serializable, f, indent=2)
    print(f"- Final params saved → {params_path}")

    # 儲存 model（joblib.dump 已在此，但確保使用 model_path 已含 time_range）
    joblib.dump(final_model, model_path)
    print(f"- Model saved → {os.path.abspath(model_path)}")


    return auc2, final_params, roc_info2, model_path


def _plot_avg_roc_from_tprs(roc_info: Tuple[np.ndarray, List[np.ndarray]], title: str, auc_value: float, save_path :str = None) -> None:
    """
    與 CodeA 一致的平均 ROC 視覺化（以 tprs 的平均曲線）。
    """
    fpr_mean, tprs = roc_info
    mean_tpr = np.mean(tprs, axis=0)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_mean, mean_tpr, label=f"Mean ROC (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"- Stage-2 average ROC saved → {save_path}")
    
    plt.close()


# ------------------------------
# CodeB：20×Nested CV + 累加混淆矩陣 + ROC
# ------------------------------
def evaluate_with_cum_confusion_like_CodeB(
    X: np.ndarray,
    y: np.ndarray,
    param_grid_b: Dict[str, np.ndarray] = None,
    prob_threshold: float = 0.110,
    seeds: Iterable[int] = range(20),
    n_outer: int = 10,
    n_inner: int = 5,
    save_dir: str = TRAIN,
    time_range: str = None,
    model: str = "SVM"
) -> Dict[str, float]:
    """
    與 CodeB 一致的評估：
    - Pipeline 無 class_weight
    - 固定閾值 0.110 轉標籤
    - 極窄格網（若未提供，沿用 CodeB 預設：C∈[2.02,2.03]×15；gamma∈[0.018,0.019]×15）
    - 20 個 seed × 外層10-fold，內層5-fold
    - 逐 fold 列印指標；最終以「累加混淆矩陣」計算整體指標 + 以全部 y_proba 計算 AUC 與 ROC
    """
    if model == "SVM":
        if param_grid_b is None:
            param_grid_b = {
                "svm__C":     np.linspace(2.02, 2.03, 15),
                "svm__gamma": np.linspace(0.018, 0.019, 15),
            }
        else :
            param_grid_b = {
                "svm__C" : np.linspace(param_grid_b['svm__C']-0.05,param_grid_b['svm__C']+0.05,15),
                "svm__gamma" : np.linspace(param_grid_b['svm__gamma']-0.0005,param_grid_b['svm__gamma']+0.0005,15)
            }
    elif model == "XGB":
        param_grid_b  = {
            'xgb__n_estimators':   [100, 250],
            'xgb__max_depth':      [3, 6],
            'xgb__learning_rate':  [0.01, 0.065],
            'xgb__gamma':          [0, 3]
        }

    pipeline_b = _build_pipeline(model=model)  
    print(f"\n=== CodeB 評估：使用參數 ===\nparam_grid : {param_grid_b}\npipeline_b: {pipeline_b}")
    print("\n=== CodeB 評估：20×Nested CV + 累加混淆矩陣 ===")

    # 收集器
    accuracy_scores, f1_scores, auc_scores = [], [], []
    sensitivities, specificities, precision_scores = [], [], []
    mean_fpr = np.linspace(0, 1, 35)
    tprs_list, auc_list = [], []
    all_y_test, all_y_proba = [], []

    total_tp = total_fp = total_tn = total_fn = 0

    for seed in seeds:
        outer_kf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            gs = GridSearchCV(
                pipeline_b,
                param_grid=param_grid_b,
                scoring="roc_auc",
                cv=inner_kf,
                n_jobs=-1
            )
            gs.fit(X_train, y_train)

            y_proba = gs.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= prob_threshold).astype(int)

            # 收集合併
            all_y_test.append(y_test)
            all_y_proba.append(y_proba)

            # 混淆矩陣累加
            
            cm = confusion_matrix(y_test, y_pred, labels=[0,1]) # Origin : tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            tn, fp, fn, tp = cm.ravel()

            total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

            # 當前 fold 指標
            acc  = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0) # Original: f1   = f1_score(y_test, y_pred) if (tp + fp + fn) > 0 else 0.0 
            
            aucv = roc_auc_score(y_test, y_proba)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # print(f"[Seed {seed:02d} | Fold {fold_idx:02d}] "
            #       f"accuracy={acc:.3f}, F1={f1:.3f}, AUC={aucv:.3f}, "
            #       f"Sensitivity={sens:.3f}, Specificity={spec:.3f}, Precision={prec:.3f}")

            # 保存 per-fold
            accuracy_scores.append(acc); f1_scores.append(f1); auc_scores.append(aucv)
            sensitivities.append(sens); specificities.append(spec); precision_scores.append(prec)

            # ROC 插值
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tprs_list.append(np.interp(mean_fpr, fpr, tpr))
            auc_list.append(aucv)

    # 整體（累加混淆矩陣）
    total_samples = total_tp + total_tn + total_fp + total_fn
    acc_total  = (total_tp + total_tn) / total_samples if total_samples > 0 else 0.0
    sens_total = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    spec_total = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    prec_total = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1_total   = (2 * prec_total * sens_total) / (prec_total + sens_total) if (prec_total + sens_total) > 0 else 0.0

    # 以全部 y_test/proba 計算 AUC 與 ROC（與 CodeB 一致）
    y_test_all  = np.concatenate(all_y_test)
    y_proba_all = np.concatenate(all_y_proba)
    auc_total   = roc_auc_score(y_test_all, y_proba_all)

    print("\n=== 20×(10-Fold Outer) 累加混淆矩陣結果 ===")
    print(f"Accuracy:    {acc_total:.3f}")
    print(f"F1-score:    {f1_total:.3f}")
    print(f"AUC:         {auc_total:.3f}")
    print(f"Sensitivity: {sens_total:.3f}")
    print(f"Specificity: {spec_total:.3f}")
    print(f"Precision:   {prec_total:.3f}")

    # ROC（整體）
    fpr_total, tpr_total, _ = roc_curve(y_test_all, y_proba_all)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_total, tpr_total, label=f"average ROC (AUC={auc_total:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance level")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic curve (20×Nested CV)")
    plt.legend()
    roc_fig_path = os.path.join(save_dir, f"roc_codeB_{time_range}.png")
    plt.savefig(roc_fig_path)
    plt.close()
    print(f"- CodeB ROC saved → {roc_fig_path}")


    metrics = {
        "accuracy": acc_total,
        "f1": f1_total,
        "auc": auc_total,
        "sensitivity": sens_total,
        "specificity": spec_total,
        "precision": prec_total,
    }
    metrics_path = os.path.join(save_dir, f"codeB_metrics_{time_range}.json")
    metrics_serializable = {k: float(v) for k,v in metrics.items()}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"- CodeB metrics saved → {metrics_path}")

    return metrics


def train(surv_df: pd.DataFrame, mort_df: pd.DataFrame, time_range :str,summary_fileName:str,model_path: str = "svm_final_model.pkl",model: str = "SVM") -> None:
    """
      1) 準備資料（僅 HRV_* + 標籤，dropna/inf）
      2) Stage-1 RandomizedSearchCV（seeds=range(5), n_iter=60）
      3) Stage-2 GridSearchCV（seeds=range(10)），以眾數參數全資料重訓並存檔
      4) 顯示 Stage-2 平均 ROC
      5) CodeB 評估：20×Nested CV + 累加混淆矩陣 + ROC（閾值 0.110）
    """
    os.makedirs(TRAIN, exist_ok = True)
    save_dir = TRAIN

    if model_path is None:
        model_path = os.path.join(save_dir, f"svm_final_model_{time_range}.pkl")
    else:
        # 若給相對/絕對路徑，也在檔名結尾加 time_range
        base, ext = os.path.splitext(os.path.basename(model_path))
        model_path = os.path.join(save_dir, f"{base}_{time_range}{ext}")

    # === 1) 準備資料 ===
    X, y, dataset, total_features = _prepare_dataset(surv_df, mort_df)
    print(f"\n=== Dataset Info ===\n- Total Samples: {len(dataset)} | Features: {X.shape[1]} "
          f"| Surv: {len(surv_df)} | Mort: {len(mort_df)}")
    dataset_path = os.path.join(save_dir, f"dataset_{time_range}.csv")
    dataset.to_csv(dataset_path, index=False)
    print(f"- dataset saved → {dataset_path}")

    print(f"\n=== Modl Info  -- {model} ===")
    if model == "SVM":

        # === 2) Stage-1（粗搜） ===
        auc1, best_hist1, stage1_df = _stage1_random_search(X, y, model=model)

        stage1_csv = os.path.join(save_dir, f"stage1_best_hist_{time_range}.csv")
        stage1_df.to_csv(stage1_csv, index=False)
        print(f"- Stage-1 history saved → {stage1_csv}")

        # === 3) Stage-2（細搜 + 存模） ===
        param_grid_fine = _build_fine_grid_from_stage1(stage1_df)
        auc2, final_params, roc_info2 ,model_path_stage2 = _stage2_grid_search_and_finalize(X, y, param_grid_fine,save_dir,time_range, model_path=model_path, model=model)

        # === 4) 平均 ROC（Stage-2） ===
        fig_path = os.path.join(save_dir, f"avg_roc_stage2_{time_range}.png")
        _plot_avg_roc_from_tprs(roc_info2, title="Average ROC – Stage-2 Nested CV", auc_value=auc2,save_path= fig_path)
        


        # === 5) CodeB 評估（與原版一致） ===
        codeb_metrics_dict = evaluate_with_cum_confusion_like_CodeB(X, y,param_grid_b=final_params,save_dir=save_dir,time_range=time_range,model=model)

        # === Summary ===
        print("\n=== Summary ===")
        print(f"Stage-1  mean AUC : {auc1:.4f}")
        print(f"Stage-2  mean AUC : {auc2:.4f}")
        print(f"Final model saved : {os.path.abspath(model_path)}")

        summary = {
            "Match_surv":len(surv_df),
            "Match_mort":len(mort_df),
            "features":int(total_features),
            "stage1_mean_auc": float(auc1),
            "stage2_mean_auc": float(auc2),
            "Evaluation_mean_auc":float(codeb_metrics_dict['auc']),
            "final_model_path": os.path.abspath(model_path),
            "time_range": time_range,
            # 如果 evaluate 返回 metrics dict，包含它
            "codeB_metrics": codeb_metrics_dict
        }
        

    elif model == "XGB":
        codeb_metrics_dict = evaluate_with_cum_confusion_like_CodeB(X, y,save_dir=save_dir,time_range=time_range,model=model)
        summary = {
            "Match_surv":len(surv_df),
            "Match_mort":len(mort_df),
            "features":int(total_features),
            "Evaluation_mean_auc":float(codeb_metrics_dict['auc']),
            "time_range": time_range,
            # 如果 evaluate 返回 metrics dict，包含它
            "codeB_metrics": codeb_metrics_dict
        }
    
    summary_path = os.path.join(save_dir, f"{summary_fileName}_{time_range}_{model}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"- Summary saved → {summary_path}")
