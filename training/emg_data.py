"""
녹화 데이터 읽기 · 필터 · 실시간 지표 · 그림 (train_v2.py와 host_check.py가 함께 씁니다)

    data/Fixed/fixed01.csv ~ fixed15.csv      고정 순서 세션 (학습 / 교차검증)
    data/Random/random01.csv ~ random05.csv   무작위 순서 세션 (시험 전용)

이 파일의 함수들은 v1(STFT + CNN, v1-stft-cnn 브랜치의 train_emg.py)에서 쓰던 것을 그대로 가져온 것입니다.
그래서 지연 · 오출력 같은 지표가 v1과 같은 기준으로 계산됩니다.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 설정 (데이터와 신호에 관한 것만. 모델 · 판단 규칙 설정은 train_v2.py의 CFG2)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    # ---- 경로 ----
    "fist_dir": os.path.join(BASE_DIR, "data", "Fixed"),
    "random_dir": os.path.join(BASE_DIR, "data", "Random"),
    "exclude": [],              # 직접 뺄 파일 이름 (예: ["fixed03"])
    "qc_min_ratio": 1.3,        # 힘줄 때 신호가 쉴 때의 1.3배도 안 되면 불량 파일로 자동 제외

    # ---- 라벨 ----
    # 실제로 쓰는 동작 목록은 데이터의 순서 파일을 보고 자동으로 정해짐 (아래 순서대로 번호가 매겨짐)
    "all_classes": ["Rest", "Thumb", "Index", "Middle", "Ring", "Little", "Fist", "Pinch"],
    "class_names": ["Rest", "Thumb", "Index", "Middle", "Ring", "Little"],
    "fist_order": ["Rest", "Thumb", "Index", "Middle", "Ring", "Little"],
    "random_orders": {
        1: ["Rest", "Ring", "Middle", "Little", "Thumb", "Index"],
        2: ["Rest", "Ring", "Thumb", "Middle", "Little", "Index"],
        3: ["Rest", "Index", "Ring", "Thumb", "Little", "Middle"],
        4: ["Rest", "Little", "Thumb", "Index", "Ring", "Middle"],
        5: ["Rest", "Middle", "Index", "Little", "Thumb", "Ring"],
    },
    "phase_s": 5.0,             # GUI의 힘주기/쉬기 한 구간 길이 (초)

    # ---- 신호 ----
    "fs": 1000,
    "channels": [0, 1, 2, 3],   # 0=Ch 00 ... 3=Ch 03
    "hp_cutoff": 20.0,          # 고역통과 (Hz)
    "notch_freq": 60.0,         # 전원 노이즈 제거 (한국 60Hz)
    "notch_q": 30.0,

    # ---- 채점할 때 지시 구간 앞뒤에서 빼는 시간 (사람 반응 지연) ----
    "contract_trim_start_s": 0.2,
    "first_rest_trim_start_s": 0.5,
    "relax_trim_start_s": 0.6,
    "relax_trim_end_s": 0.2,
}


# =====================================================================
# 데이터 읽기 / 라벨링
# =====================================================================
def _dirs(path):
    """data/FIST_Re 대신 GUI가 만드는 data/Fist 폴더도 같이 찾음"""
    base, name = os.path.split(path)
    out = [path]
    if name in ("FIST_Re", "Fist", "Fixed"):
        out += [os.path.join(base, n) for n in ("FIST_Re", "Fist", "Fixed") if n != name]
    return [d for d in out if os.path.isdir(d)]


def _read_order_file(csv_path):
    """GUI가 Random 녹화 때 저장한 순서 파일 (random01_order.txt) 읽기"""
    order_path = os.path.splitext(csv_path)[0] + "_order.txt"
    if not os.path.exists(order_path):
        return None
    with open(order_path, encoding="utf-8") as f:
        order = [w.strip() for w in f.read().strip().split(",") if w.strip()]
    return order


def list_sessions(cfg):
    sessions = []
    exclude = {e.lower() for e in cfg["exclude"]}
    seen = set()

    for d in _dirs(cfg["fist_dir"]):
        for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
            name = os.path.splitext(os.path.basename(path))[0]
            # 고정 순서 세션: 예전 이름 fist01, 새 이름 fixed01 둘 다 인식
            if not re.fullmatch(r"(fist|fixed)\d+", name, re.IGNORECASE) or name.lower() in seen:
                continue
            seen.add(name.lower())
            if name.lower() in exclude:
                print(f"  제외: {name}")
                continue
            order = _read_order_file(path) or cfg["fist_order"]
            sessions.append({"name": name, "path": path, "group": "fist", "order": order})

    for path in sorted(glob.glob(os.path.join(cfg["random_dir"], "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.fullmatch(r"random(\d+)", name, re.IGNORECASE)
        if not m or name.lower() in exclude:
            continue
        order = _read_order_file(path)            # 1순위: GUI가 저장한 순서 파일
        if order is None:
            order = cfg["random_orders"].get(int(m.group(1)))   # 2순위: CFG에 적은 순서
        if order is None:
            print(f"  [경고] {name}: 순서 정보가 없어서 제외 ({name}_order.txt 또는 CFG random_orders 필요)")
            continue
        bad = [g for g in order if g not in cfg["all_classes"]]
        if bad:
            print(f"  [경고] {name}: 순서 파일에 모르는 동작 {bad} -> 제외")
            continue
        sessions.append({"name": name, "path": path, "group": "random", "order": order})
    return sessions


def runs_of(mask):
    """True가 연속된 구간 [(start, end), ...], end는 포함 안 함"""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def design_sos(cfg):
    hp = butter(4, cfg["hp_cutoff"], btype="highpass", fs=cfg["fs"], output="sos")
    b, a = iirnotch(cfg["notch_freq"], cfg["notch_q"], fs=cfg["fs"])
    notch = np.hstack([b / a[0], a / a[0]])[None, :]
    return np.vstack([hp, notch]).astype(np.float32)


def load_session(sess, cfg, sos):
    df = pd.read_csv(sess["path"], skiprows=3)
    df.columns = [str(c).strip() for c in df.columns]
    ch_cols = [f"Ch {c:02d}" for c in cfg["channels"]]
    # 전송 중 깨진 값(예: "73-6.0")은 NaN으로 바꾼 뒤 앞뒤 값으로 채움
    raw = df[ch_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(np.float32)
    phase = df["Phase"].astype(str).str.strip().str.lower().to_numpy()

    contract = runs_of(phase == "contract")
    relax = runs_of(phase == "relax")
    if len(contract) != len(sess["order"]):
        print(f"  [경고] {sess['name']}: Contract 구간 {len(contract)}개 "
              f"(예상 {len(sess['order'])}개) -> 제외")
        return None

    # 기기처럼 파일 처음부터 끝까지 필터를 연속으로 적용 (과거 샘플만 사용)
    filt = sosfilt(sos, raw, axis=0).astype(np.float32)          # (N, ch)

    n = len(raw)
    cls = {name: i for i, name in enumerate(cfg["class_names"])}
    instr = np.full(n, cls["Rest"], dtype=np.int64)              # 샘플별 지시 라벨
    zones = []                                                   # (start, end, label, kind)
    for k, (s, e) in enumerate(contract):
        lab = cls[sess["order"][k]]
        instr[s:e] = lab
        zones.append((s, e, lab, "contract", k))
    for j, (s, e) in enumerate(relax):
        zones.append((s, e, cls["Rest"], "relax", j))

    # 길이가 비정상인 구간(녹화 시작 지연 등으로 라벨이 밀린 구간)은 사용하지 않음
    nominal = cfg["phase_s"] * cfg["fs"]
    valid = np.ones(n, dtype=bool)
    bad = []
    for (s, e, lab, kind, k) in zones:
        if abs((e - s) - nominal) > 0.05 * nominal:
            valid[s:e] = False
            bad.append(f"{'수축' if kind == 'contract' else '휴식'}{k + 1}({cfg['class_names'][lab] if kind == 'contract' else 'Rest'})")

    sess.update({"raw": raw, "filt": filt, "instr": instr, "zones": zones, "n": n,
                 "valid": valid, "bad_zones": bad})

    # ---- 품질 검사: 손가락 구간마다 가장 크게 반응한 채널이 휴식 때보다 몇 배인지 ----
    rest_rms = np.median([np.sqrt((filt[s + 500:e] ** 2).mean(axis=0))
                          for (s, e, lab, kind, k) in zones if kind == "relax"], axis=0)
    rest_rms = np.maximum(rest_rms, 1e-6)
    ratios = []
    for (s, e, lab, kind, k) in zones:
        if kind == "contract" and lab != cls["Rest"]:
            r = np.sqrt((filt[s + 500:e] ** 2).mean(axis=0)) / rest_rms
            ratios.append((cfg["class_names"][lab], float(r.max())))
    sess["qc_ratio"] = float(np.median([r for _, r in ratios]))
    sess["qc_weak"] = [name for name, r in ratios if r < 1.2]

    # 채널별 점검: 어떤 동작에도 반응하지 않는 채널 / 쉴 때부터 노이즈가 큰 채널
    per_ch_max = np.zeros(filt.shape[1])
    for (s, e, lab, kind, k) in zones:
        if kind == "contract" and lab != cls["Rest"] and valid[s:e].all():
            r = np.sqrt((filt[s + 500:e] ** 2).mean(axis=0)) / rest_rms
            per_ch_max = np.maximum(per_ch_max, r)
    sess["dead_channels"] = [f"Ch {cfg['channels'][c]:02d}" for c in range(len(per_ch_max))
                             if per_ch_max[c] < 1.5]
    sess["noisy_channels"] = [f"Ch {cfg['channels'][c]:02d}" for c in range(len(rest_rms))
                              if rest_rms[c] > 5 * np.median(rest_rms)]
    return sess


# =====================================================================
# 평가
# =====================================================================
def confusion(y, p, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def realtime_metrics(results, cfg):
    fs = cfg["fs"]
    n_cls = len(cfg["class_names"])
    rest = cfg["class_names"].index("Rest")
    span = int((cfg["epoch_s"] + (cfg["vote_n"] - 1) * cfg["step_eval_s"]) * fs)
    tol = int(2.5 * fs)
    stable_true, stable_pred = [], []
    false_moves, relax_total = 0, 0
    on_lat, off_lat, misses = [], [], 0
    false_events, total_minutes = 0, 0.0

    for r in results:
        sess, ends, voted = r["sess"], r["ends"], r["voted"]
        total_minutes += sess["n"] / fs / 60
        finger_zones = [(s, e, lab) for (s, e, lab, kind, k) in sess["zones"]
                        if kind == "contract" and lab != rest and sess["valid"][s:e].all()]
        for (s, e, lab, kind, k) in sess["zones"]:
            if not sess["valid"][s:e].all():
                continue
            trim = cfg["contract_trim_start_s"] if kind == "contract" else cfg["relax_trim_start_s"]
            # 투표에 쓰인 창 전체가 한 지시 구간 안에 들어간 시점만 채점
            sel = (ends - span >= s + int(trim * fs)) & (ends <= e)
            stable_true += [lab] * int(sel.sum())
            stable_pred += voted[sel].tolist()
            if kind == "relax":
                relax_total += int(sel.sum())
                false_moves += int((voted[sel] != rest).sum())
        for (s, e, lab) in finger_zones:
            on = np.flatnonzero((ends > s) & (ends <= e) & (voted == lab))
            if len(on):
                on_lat.append((ends[on[0]] - s) / fs)
                off = np.flatnonzero((ends > e) & (voted != lab))
                if len(off):
                    off_lat.append((ends[off[0]] - e) / fs)
            else:
                misses += 1
        # 엉뚱한 손가락이 켜진 횟수: 출력이 손가락인 구간마다, 같은 손가락 지시와 겹치는지 확인
        i = 0
        while i < len(voted):
            if voted[i] == rest:
                i += 1
                continue
            j = i
            while j + 1 < len(voted) and voted[j + 1] == voted[i]:
                j += 1
            a, b = ends[i], ends[j]
            ok = any(lab == voted[i] and a <= e + tol and b >= s for (s, e, lab) in finger_zones)
            false_events += 0 if ok else 1
            i = j + 1

    # 근전도가 실제로 올라온 시점 / 내려간 시점 기준 지연 (논문의 "제어 지연"과 같은 기준)
    emg_on, emg_off = [], []
    for r in results:
        sess, ends, voted = r["sess"], r["ends"], r["voted"]
        env = np.sqrt(np.convolve((sess["filt"] ** 2).max(axis=1), np.ones(100) / 100, mode="same"))
        relax = [(s, e) for (s, e, lab, kind, k) in sess["zones"] if kind == "relax" and sess["valid"][s:e].all()]
        if not relax:
            continue
        base = np.median(np.concatenate([env[s + 800:e] for (s, e) in relax]))
        for (s, e, lab, kind, k) in sess["zones"]:
            if kind != "contract" or lab == rest or not sess["valid"][s:e].all():
                continue
            peak = np.median(env[s + 1000:e])
            thr = base + 0.3 * (peak - base)
            a0 = max(0, s - 500)
            up = np.flatnonzero(env[a0:e] > thr)
            if not len(up):
                continue
            onset = a0 + up[0]
            hit = np.flatnonzero((ends > onset) & (ends <= e + 1000) & (voted == lab))
            if len(hit):
                emg_on.append((ends[hit[0]] - onset) / fs)
            down = np.flatnonzero(env[e - 500:min(len(env), e + 3000)] < thr)
            down = down[down > 400]
            if len(down):
                offset = e - 500 + down[0]
                rel = np.flatnonzero((ends > offset) & (voted != lab))
                if len(rel):
                    emg_off.append((ends[rel[0]] - offset) / fs)

    cm = confusion(stable_true, stable_pred, n_cls)
    return {
        "emg_on_median": float(np.median(emg_on)) if emg_on else float("nan"),
        "emg_on_p90": float(np.percentile(emg_on, 90)) if emg_on else float("nan"),
        "emg_off_median": float(np.median(emg_off)) if emg_off else float("nan"),
        "cm": cm,
        "acc": np.trace(cm) / max(cm.sum(), 1),
        "false_move_rate": false_moves / max(relax_total, 1),
        "latency_median": float(np.median(on_lat)) if on_lat else float("nan"),
        "release_median": float(np.median(off_lat)) if off_lat else float("nan"),
        "detected": len(on_lat),
        "missed": misses,
        "false_events": false_events,
        "minutes": total_minutes,
    }


# =====================================================================
# 그림
# =====================================================================
def plot_confusion(cm, names, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    rownorm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(rownorm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if rownorm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_realtime(results, cfg, path):
    names = cfg["class_names"]
    fig, axes = plt.subplots(len(results), 1, figsize=(13, 2.2 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        t = np.arange(r["sess"]["n"]) / cfg["fs"]
        ax.plot(t, r["sess"]["instr"], color="0.75", lw=6, label="Instruction")
        ax.step(r["ends"] / cfg["fs"], r["voted"], where="post", color="tab:blue",
                lw=1.8, label="ESP32 output (vote)")
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_ylim(-0.5, len(names) - 0.5)
        ax.set_title(r["sess"]["name"], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# C 헤더 내보내기
# =====================================================================
def c_array(name, arr, ctype="float"):
    flat = np.asarray(arr).ravel()
    if ctype == "float":
        items = []
        for v in flat:
            t = f"{float(v):.9g}"
            if not any(ch in t for ch in ".eE"):
                t += ".0"          # C에서 1f 는 오류라서 1.0f 로
            items.append(t + "f")
    else:
        items = [str(int(v)) for v in flat]
    lines = [", ".join(items[i:i + 8]) for i in range(0, len(items), 8)]
    body = ",\n    ".join(lines)
    return f"static const {ctype} {name}[{len(flat)}] = {{\n    {body}\n}};\n"
