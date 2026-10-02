"""
EMG 손가락 동작 분류: PyTorch 학습 -> ESP32용 C 헤더 내보내기

실행 방법 (VS Code 터미널):
    python train_emg.py

필요한 폴더 구조:
    data/Fixed/fixed01.csv ~ fixed15.csv
    data/Random/random01.csv ~ random05.csv

결과 (output 폴더):
    report.txt              정확도 요약
    confusion_window.png    1초 단위 혼동행렬
    realtime_random.png     실시간 흉내 결과 (Random 세션 5개)
    training_curve.png      학습 곡선
    emg_model.pt            PyTorch 모델
    emg_model.h             ESP32용 모델 가중치 + 전처리 설정
    emg_test_vectors.h      ESP32 정답 맞추기 테스트 데이터
  -> emg_model.h, emg_test_vectors.h 는 ../firmware/ 의 세 스케치 폴더에도 자동 복사됨
"""

import os
import re
import glob
import copy
import random
import shutil

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    # ---- 경로 ----
    "fist_dir": os.path.join(BASE_DIR, "data", "Fixed"),
    "random_dir": os.path.join(BASE_DIR, "data", "Random"),
    "out_dir": os.path.join(BASE_DIR, "output"),
    "sketch_dirs": [os.path.join(BASE_DIR, "..", "firmware", "esp32_emg_test"),
                    os.path.join(BASE_DIR, "..", "firmware", "esp32_emg_realtime"),
                    os.path.join(BASE_DIR, "..", "firmware", "esp32_emg_bench")],
    "exclude": [],              # 직접 뺄 파일 이름 (예: ["fist03"])
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
    # 0=Ch 00 ... 3=Ch 03 (9/24 데이터부터 4채널 모두 정상)
    "channels": [0, 1, 2, 3],
    "hp_cutoff": 20.0,          # 고역통과 (Hz)
    "notch_freq": 60.0,         # 전원 노이즈 제거 (한국 60Hz)
    "notch_q": 30.0,
    "warmup_s": 0.5,            # 파일 시작 직후 필터 안정화 구간은 사용 안 함

    # ---- 구간 자르기 (반응 지연 고려) ----
    "epoch_s": 1.0,
    "step_eval_s": 0.5,         # 검증/테스트/ESP32: 0.5초마다 판단
    "step_train_s": 0.25,       # 학습 데이터는 더 촘촘히 잘라서 양을 늘림
    "contract_trim_start_s": 0.25,
    "first_rest_trim_start_s": 0.5,
    "relax_trim_start_s": 0.75,
    "relax_trim_end_s": 0.25,
    # "end": 1초 창의 '끝부분' 상태로 라벨 (힘을 빼면 더 빨리 Rest로 바뀌게 학습)
    # "zone": 지시 구간 안에 완전히 들어간 창만 사용
    "label_mode": "end",
    "label_delay_s": 0.15,      # 사람 반응 지연 보정
    "label_margin_s": 0.25,     # 동작이 바뀌는 순간 앞뒤 0.25초는 애매해서 제외

    # ---- STFT (ESP32에서 똑같이 계산) ----
    "win": 256,
    "hop": 32,
    "nfft": 256,
    "band_start_bin": 6,        # 6 * 3.906Hz = 23Hz 부터
    "band_width": 5,            # 주파수 칸 5개씩 묶음
    "num_bands": 22,            # 22묶음 -> 23 ~ 449Hz
    "zscore_epoch": True,       # 1초마다 채널별 정규화 (신호 크기 변화에 덜 민감)

    # ---- 모델 / 학습 ----
    "widths": [8, 16, 32],
    "dropout": 0.3,
    "val_sessions": 2,          # FIST_Re 중 검증용 세션 수
    "batch_size": 64,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "max_epochs": 80,
    "patience": 12,
    "seed": 0,

    # ---- 실시간 흉내 ----
    "vote_n": 3,                # 최근 vote_n번 판단 중
    "vote_k": 2,                # vote_k번 이상 같은 손가락이면 켬 (vote_n = vote_k 이면 "연속 k번")
    "vote_switch_k": None,      # 켜진 손가락에서 다른 손가락으로 바꿀 때 필요한 연속 횟수 (None이면 vote_k와 같음)
    "vote_release_n": 1,        # 휴식 판단이 이만큼 연속으로 나와야 끔
    "vote_hold_s": 0.0,         # 출력이 바뀐 뒤 최소 이 시간 동안은 다시 안 바꿈 (깜빡임 방지)
    "rest_gate": 0.0,           # 모든 채널 신호가 휴식 기준의 이 배수보다 작으면 무조건 휴식 (0이면 끔)

    # ---- ESP32 테스트 데이터 ----
    "test_chunk_s": 2.0,        # 1초 필터 안정화 + 1초 판단 구간
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


def make_windows(sess, cfg, step_s):
    """학습/평가용 1초 창 목록: (시작 샘플, 라벨, 구간 id)"""
    fs = cfg["fs"]
    L = int(cfg["epoch_s"] * fs)
    step = int(step_s * fs)
    warm = int(cfg["warmup_s"] * fs)
    out = []
    if cfg["label_mode"] == "end":
        delay = int(cfg["label_delay_s"] * fs)
        margin = int(cfg["label_margin_s"] * fs)
        bounds = np.array(sorted({z[0] for z in sess["zones"]} | {z[1] for z in sess["zones"]}))
        bounds = bounds[(bounds > 0) & (bounds < sess["n"])]
        zone_of = np.zeros(sess["n"], dtype=np.int64)
        for (s, e, lab, kind, k) in sess["zones"]:
            zone_of[s:e] = k if kind == "contract" else 100 + k
        for end in range(warm + L, sess["n"] + 1, step):
            tau = end - delay
            if len(bounds) and np.min(np.abs(bounds - tau)) < margin:
                continue
            if not sess["valid"][end - L:end].all():
                continue
            out.append((end - L, int(sess["instr"][tau]), int(zone_of[tau])))
        return out
    for (s, e, lab, kind, k) in sess["zones"]:
        if kind == "contract":
            trim = cfg["first_rest_trim_start_s"] if k == 0 else cfg["contract_trim_start_s"]
            a, b = s + int(trim * fs), e
            seg_id = k                      # 0~5: 수축 지시 구간
        else:
            a = s + int(cfg["relax_trim_start_s"] * fs)
            b = e - int(cfg["relax_trim_end_s"] * fs)
            seg_id = 100 + k                # 100~: 휴식 구간
        a = max(a, warm)
        for st in range(a, b - L + 1, step):
            if not sess["valid"][st:st + L].all():
                continue
            out.append((st, lab, seg_id))
    return out


# =====================================================================
# 특징(STFT) 계산: ESP32의 emg_infer.cpp와 같은 계산
# =====================================================================
def feature_shape(cfg):
    L = int(cfg["epoch_s"] * cfg["fs"])
    frames = (L - cfg["win"]) // cfg["hop"] + 1
    return len(cfg["channels"]), cfg["num_bands"], frames


def compute_features(epochs, cfg):
    """epochs: (N, ch, L) float32 필터 통과한 신호 -> (N, ch, bands, frames) float32 (정규화 전)"""
    n_ch, nb, nf = feature_shape(cfg)
    win, hop, nfft = cfg["win"], cfg["hop"], cfg["nfft"]
    b0, bw = cfg["band_start_bin"], cfg["band_width"]
    hamming = np.hamming(win).astype(np.float32)
    idx = np.arange(nf)[:, None] * hop + np.arange(win)[None, :]   # (frames, win)

    out = np.empty((len(epochs), n_ch, nb, nf), dtype=np.float32)
    for i0 in range(0, len(epochs), 256):
        x = epochs[i0:i0 + 256].astype(np.float32)
        if cfg["zscore_epoch"]:
            m = x.mean(axis=2, keepdims=True, dtype=np.float64)
            s = x.std(axis=2, keepdims=True, dtype=np.float64)
            x = ((x - m) / np.maximum(s, 1e-6)).astype(np.float32)
        frames = x[:, :, idx] * hamming                             # (B, ch, frames, win)
        spec = np.fft.rfft(frames, n=nfft, axis=3)
        power = (spec.real ** 2 + spec.imag ** 2)[..., b0:b0 + nb * bw]
        power = power.reshape(*power.shape[:3], nb, bw).mean(axis=4)  # (B, ch, frames, bands)
        feat = 10.0 * np.log10(power + 1e-10)
        out[i0:i0 + 256] = np.transpose(feat, (0, 1, 3, 2)).astype(np.float32)
    return out


def windows_to_arrays(sessions, win_lists, cfg):
    L = int(cfg["epoch_s"] * cfg["fs"])
    X, y, meta = [], [], []
    for si, (sess, wl) in enumerate(zip(sessions, win_lists)):
        for (st, lab, seg_id) in wl:
            X.append(sess["filt"][st:st + L].T)      # (ch, L)
            y.append(lab)
            meta.append((si, seg_id, st))
    if not X:
        return np.zeros((0,)), np.zeros((0,), np.int64), []
    return compute_features(np.stack(X), cfg), np.array(y, np.int64), meta


# =====================================================================
# 모델
# =====================================================================
class EmgNet(nn.Module):
    def __init__(self, n_ch, n_classes, widths, dropout):
        super().__init__()
        c1, c2, c3 = widths
        self.features = nn.Sequential(
            nn.Conv2d(n_ch, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, n_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))   # 채널별 전체 평균
        return self.fc(self.drop(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_logits(model, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i:i + batch])).numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 0), np.float32)


def train_model(Xtr, ytr, Xva, yva, cfg, verbose=True):
    set_seed(cfg["seed"])
    n_cls = len(cfg["class_names"])
    model = EmgNet(Xtr.shape[1], n_cls, cfg["widths"], cfg["dropout"])

    counts = np.bincount(ytr, minlength=n_cls).astype(np.float32)
    weights = counts.sum() / (n_cls * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    g = torch.Generator().manual_seed(cfg["seed"])

    best = {"loss": np.inf, "state": None, "epoch": 0}
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    wait = 0
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            lv = model(Xva_t)
            vloss = loss_fn(lv, yva_t).item()
            vacc = (lv.argmax(1) == yva_t).float().mean().item()
        hist["train_loss"].append(tot / len(perm))
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        if verbose:
            print(f"  epoch {ep:3d}  train_loss {tot / len(perm):.4f}  "
                  f"val_loss {vloss:.4f}  val_acc {vacc * 100:5.1f}%")

        if vloss < best["loss"] - 1e-4:
            best = {"loss": vloss, "state": copy.deepcopy(model.state_dict()), "epoch": ep}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                if verbose:
                    print(f"  -> 검증 손실이 {cfg['patience']}번 연속 개선되지 않아 종료 "
                          f"(최고 시점: epoch {best['epoch']})")
                break

    model.load_state_dict(best["state"])
    model.eval()
    return model, hist, best["epoch"]


# =====================================================================
# 평가
# =====================================================================
def confusion(y, p, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def vote_sequence(preds, vote_n, rest_idx, vote_k=None, switch_k=None, release_n=1, hold_n=0):
    """켜기는 신중하게, 끄기는 빠르게 (ESP32의 emg_vote_update와 같은 규칙)
    - 이번 판단이 Rest면 바로 Rest
    - 휴식 상태에서: 최근 vote_n번 중 vote_k번 이상 같은 손가락이면 켬
    - 손가락이 켜진 상태에서 다른 손가락으로 바꿀 때: 최근 switch_k번 중 switch_k번(연속) 같아야 바꿈
    """
    if vote_k is None:
        vote_k = vote_n // 2 + 1
    if switch_k is None:
        switch_k = vote_k
    out = []
    prev = rest_idx
    since = 1 << 30          # 마지막으로 출력이 바뀐 뒤 지난 판단 수
    rest_run = 0             # 연속된 휴식 판단 수
    for i in range(len(preds)):
        since += 1
        cand = prev
        if preds[i] == rest_idx:
            rest_run += 1
            if rest_run >= release_n:          # 휴식이 release_n번 연속이면 끔
                cand = rest_idx
        else:
            rest_run = 0
            if prev == rest_idx:
                hist, need = preds[max(0, i - vote_n + 1):i + 1], vote_k
            else:
                hist, need = preds[max(0, i - switch_k + 1):i + 1], switch_k
            vals, cnt = np.unique(hist, return_counts=True)
            best = int(vals[np.argmax(cnt)])
            if cnt.max() >= need and best != rest_idx and best != prev:
                cand = best
        if cand != prev and since >= hold_n:   # 바뀐 지 얼마 안 됐으면 유지
            prev = cand
            since = 0
        out.append(prev)
    return np.array(out)


def hold_decisions(cfg):
    """최소 유지 시간을 판단 횟수로 바꿈"""
    return int(round(cfg["vote_hold_s"] / cfg["step_eval_s"]))


def rest_gate_mask(windows_rms, rest_rms, gate):
    """모든 채널의 신호 크기가 휴식 기준 x gate 보다 작으면 True (= 무조건 휴식)"""
    if not gate:
        return np.zeros(len(windows_rms), dtype=bool)
    return (windows_rms < gate * rest_rms).all(axis=1)


def compute_rest_rms(sessions, cfg):
    """학습 세션의 휴식 구간에서 채널별 기준 신호 크기 (창 길이 기준)"""
    L = int(cfg["epoch_s"] * cfg["fs"])
    vals = []
    for s in sessions:
        for (a, e, lab, kind, k) in s["zones"]:
            if kind == "relax" and s["valid"][a:e].all():
                for st in range(a + 800, e - L, 250):
                    vals.append(np.sqrt((s["filt"][st:st + L] ** 2).mean(axis=0)))
    return np.median(np.array(vals), axis=0).astype(np.float32)


def realtime_simulation(model, sessions, cfg, norm, rest_rms=None):
    """세션 전체를 0.5초마다 판단하는 실시간 흉내"""
    fs = cfg["fs"]
    L = int(cfg["epoch_s"] * fs)
    step = int(cfg["step_eval_s"] * fs)
    warm = int(cfg["warmup_s"] * fs)
    rest = cfg["class_names"].index("Rest")
    results = []
    for sess in sessions:
        ends = np.arange(warm + L, sess["n"] + 1, step)
        X = compute_features(np.stack([sess["filt"][t - L:t].T for t in ends]), cfg)
        X = ((X - norm[0]) / norm[1]).astype(np.float32)
        raw_pred = predict_logits(model, X).argmax(1)
        if cfg["rest_gate"] and rest_rms is not None:
            rms = np.stack([np.sqrt((sess["filt"][t - L:t] ** 2).mean(axis=0)) for t in ends])
            raw_pred = np.where(rest_gate_mask(rms, rest_rms, cfg["rest_gate"]), rest, raw_pred)
        voted = vote_sequence(raw_pred, cfg["vote_n"], rest, cfg["vote_k"], cfg["vote_switch_k"],
                              cfg["vote_release_n"], hold_decisions(cfg))
        results.append({"sess": sess, "ends": ends, "raw": raw_pred, "voted": voted})
    return results


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


def plot_history(hist, best_epoch, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].axvline(best_epoch - 1, color="k", ls="--", lw=1)
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(np.array(hist["val_acc"]) * 100)
    ax[1].axvline(best_epoch - 1, color="k", ls="--", lw=1)
    ax[1].set_title("Validation accuracy (%)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# ESP32용 내보내기
# =====================================================================
def fold_bn(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    scale = (bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)).astype(np.float64)
    wf = w * scale[:, None, None, None]
    bf = (bn.bias.detach().numpy() - bn.running_mean.numpy() * scale).astype(np.float64)
    return wf.astype(np.float32), bf.astype(np.float32)


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


def export_model_header(model, cfg, sos, norm, path, rest_rms=None):
    n_ch, nb, nf = feature_shape(cfg)
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]
    folded = [fold_bn(c, b) for c, b in zip(convs, bns)]
    c1, c2, c3 = cfg["widths"]
    buf = max(c1 * nb * nf, c2 * (nb // 2) * (nf // 2), c3 * (nb // 4) * (nf // 4), n_ch * nb * nf)
    names = ", ".join(f'"{n}"' for n in cfg["class_names"])

    h = []
    h.append("// 자동 생성 파일 (train_emg.py) - 직접 수정하지 마세요\n")
    h.append("#ifndef EMG_MODEL_H\n#define EMG_MODEL_H\n\n")
    h.append(f"#define EMG_FS            {cfg['fs']}\n")
    h.append(f"#define EMG_NUM_CH        {n_ch}\n")
    h.append(f"#define EMG_EPOCH_LEN     {int(cfg['epoch_s'] * cfg['fs'])}\n")
    h.append(f"#define EMG_STEP_LEN      {int(cfg['step_eval_s'] * cfg['fs'])}\n")
    h.append(f"#define EMG_WIN           {cfg['win']}\n")
    h.append(f"#define EMG_HOP           {cfg['hop']}\n")
    h.append(f"#define EMG_NFFT          {cfg['nfft']}\n")
    h.append(f"#define EMG_BAND_START    {cfg['band_start_bin']}\n")
    h.append(f"#define EMG_BAND_WIDTH    {cfg['band_width']}\n")
    h.append(f"#define EMG_NUM_BANDS     {nb}\n")
    h.append(f"#define EMG_NUM_FRAMES    {nf}\n")
    h.append(f"#define EMG_ZSCORE_EPOCH  {1 if cfg['zscore_epoch'] else 0}\n")
    h.append(f"#define EMG_NUM_SOS       {len(sos)}\n")
    h.append(f"#define EMG_C1            {c1}\n#define EMG_C2            {c2}\n#define EMG_C3            {c3}\n")
    h.append(f"#define EMG_NUM_CLASSES   {len(cfg['class_names'])}\n")
    h.append(f"#define EMG_BUF_SIZE      {buf}\n")
    h.append(f"#define EMG_VOTE_N        {cfg['vote_n']}\n")
    h.append(f"#define EMG_VOTE_K        {cfg['vote_k']}\n")
    h.append(f"#define EMG_VOTE_SWITCH_K {cfg['vote_switch_k'] or cfg['vote_k']}\n")
    h.append(f"#define EMG_REST_GATE     {float(cfg['rest_gate'] or 0.0):.3f}f\n")
    h.append(f"#define EMG_VOTE_RELEASE_N {cfg['vote_release_n']}\n")
    h.append(f"#define EMG_VOTE_HOLD_N   {hold_decisions(cfg)}\n")
    h.append(f"#define EMG_REST_CLASS    {cfg['class_names'].index('Rest')}\n\n")
    h.append("// 모델이 쓰는 채널 (CSV의 Ch 번호 = 보드의 EMG_PINS 순서)\n")
    h.append("static const int EMG_CHANNEL_IDX[EMG_NUM_CH] = { "
             + ", ".join(str(c) for c in cfg["channels"]) + " };\n")
    h.append(f"static const char *const EMG_CLASS_NAMES[EMG_NUM_CLASSES] = {{ {names} }};\n\n")
    h.append("#ifdef EMG_MODEL_DATA\n\n")
    h.append("// 필터: 행마다 b0 b1 b2 a0 a1 a2 (고역통과 2단 + 60Hz 노치)\n")
    h.append(c_array("EMG_SOS", sos))
    h.append(c_array("EMG_FEAT_MEAN", norm[0].ravel()))
    h.append(c_array("EMG_FEAT_STD", norm[1].ravel()))
    h.append("// 휴식 게이트 기준: 학습 데이터 휴식 구간의 채널별 신호 크기(RMS)\n")
    h.append(c_array("EMG_REST_RMS", rest_rms if rest_rms is not None else np.zeros(n_ch)))
    for i, (w, b) in enumerate(folded, start=1):
        h.append(c_array(f"EMG_CONV{i}_W", w))
        h.append(c_array(f"EMG_CONV{i}_B", b))
    h.append(c_array("EMG_FC_W", model.fc.weight.detach().numpy()))
    h.append(c_array("EMG_FC_B", model.fc.bias.detach().numpy()))
    h.append("\n#endif  // EMG_MODEL_DATA\n#endif  // EMG_MODEL_H\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))


def reference_probs_from_chunk(model, chunk_raw, cfg, sos, norm):
    """ESP32 테스트와 똑같이: 필터를 0부터 시작해 chunk 전체에 걸고, 마지막 1초로 판단"""
    L = int(cfg["epoch_s"] * cfg["fs"])
    filt = sosfilt(sos, chunk_raw.astype(np.float32), axis=0).astype(np.float32)
    X = compute_features(filt[-L:].T[None], cfg)
    X = ((X - norm[0]) / norm[1]).astype(np.float32)
    logits = predict_logits(model, X)[0]
    p = np.exp(logits - logits.max())
    return p / p.sum()


def export_test_vectors(model, test_sessions, cfg, sos, norm, path):
    fs = cfg["fs"]
    L = int(cfg["epoch_s"] * fs)
    chunk = int(cfg["test_chunk_s"] * fs)
    n_cls = len(cfg["class_names"])
    picked = []
    rng = np.random.RandomState(cfg["seed"])
    for lab in range(n_cls):
        cands = []
        for sess in test_sessions:
            for (s, e, zl, kind, k) in sess["zones"]:
                if zl != lab or (lab == cfg["class_names"].index("Rest") and kind != "relax"):
                    continue
                mid_end = (s + e) // 2 + L // 2
                if mid_end - chunk >= 0 and mid_end <= sess["n"]:
                    cands.append((sess, mid_end))
        if cands:
            sess, end = cands[rng.randint(len(cands))]
            picked.append((lab, sess, end))

    raws, labels, probs, preds, origins = [], [], [], [], []
    for lab, sess, end in picked:
        chunk_raw = sess["raw"][end - chunk:end]                 # (chunk, ch)
        p = reference_probs_from_chunk(model, chunk_raw, cfg, sos, norm)
        raws.append(np.round(chunk_raw.T).astype(np.int16))      # (ch, chunk)
        labels.append(lab)
        probs.append(p)
        preds.append(int(np.argmax(p)))
        origins.append(f"{sess['name']} {(end - chunk) / fs:.1f}~{end / fs:.1f}s")

    n_ch = raws[0].shape[0]
    h = ["// 자동 생성 파일 (train_emg.py): ESP32 정답 맞추기 테스트용\n",
         "#ifndef EMG_TEST_VECTORS_H\n#define EMG_TEST_VECTORS_H\n#include <stdint.h>\n\n",
         f"#define EMG_TEST_COUNT {len(raws)}\n#define EMG_TEST_LEN   {chunk}\n\n"]
    for i, o in enumerate(origins):
        h.append(f"// [{i}] {cfg['class_names'][labels[i]]}: {o}\n")
    h.append(f"\nstatic const int16_t EMG_TEST_RAW[{len(raws)}][{n_ch}][{chunk}] = {{\n")
    for r in raws:
        h.append("  {\n")
        for c in range(n_ch):
            vals = ", ".join(str(int(v)) for v in r[c])
            h.append(f"    {{ {vals} }},\n")
        h.append("  },\n")
    h.append("};\n\n")
    h.append(f"static const int EMG_TEST_LABEL[{len(raws)}] = {{ {', '.join(map(str, labels))} }};\n")
    h.append(f"static const int EMG_TEST_EXPECTED_PRED[{len(raws)}] = {{ {', '.join(map(str, preds))} }};\n")
    h.append(f"static const float EMG_TEST_EXPECTED_PROB[{len(raws)}][{n_cls}] = {{\n")
    for p in probs:
        h.append("  { " + ", ".join(f"{v:.6f}f" for v in p) + " },\n")
    h.append("};\n\n#endif\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))
    return picked


def count_ops(cfg):
    n_ch, nb, nf = feature_shape(cfg)
    c1, c2, c3 = cfg["widths"]
    ops = 9 * n_ch * c1 * nb * nf
    ops += 9 * c1 * c2 * (nb // 2) * (nf // 2)
    ops += 9 * c2 * c3 * (nb // 4) * (nf // 4)
    params = 9 * (n_ch * c1 + c1 * c2 + c2 * c3) + c1 + c2 + c3 + c3 * len(cfg["class_names"]) + len(cfg["class_names"])
    return ops, params


# =====================================================================
# 메인
# =====================================================================
def run(cfg, verbose=True, export=True):
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sos = design_sos(cfg)
    names = cfg["class_names"]
    n_cls = len(names)

    print("[1/6] 데이터 읽는 중...")
    listed = list_sessions(cfg)
    used = {g for s in listed for g in s["order"]}
    cfg["class_names"] = [c for c in cfg["all_classes"] if c in used]
    names = cfg["class_names"]
    n_cls = len(names)
    print(f"  동작 {n_cls}가지: {', '.join(names)}")
    sessions = []
    for s in listed:
        loaded = load_session(s, cfg, sos)
        if loaded is None:
            continue
        expected = int(len(loaded["order"]) * 2 * cfg["phase_s"] * cfg["fs"])   # 6동작 60초, 8동작 80초
        if loaded["n"] < 0.95 * expected:
            print(f"  [경고] {loaded['name']}: 샘플 {loaded['n']}개 (예상 약 {expected}개) -> 녹화 중 샘플 손실")
        if loaded["n"] > 1.03 * expected:
            print(f"  [경고] {loaded['name']}: 샘플 {loaded['n']}개 (예상 약 {expected}개) -> 녹화 앞부분에 밀린 데이터가 섞였을 수 있음")
        if loaded["bad_zones"]:
            print(f"  [제외 구간] {loaded['name']}: {', '.join(loaded['bad_zones'])} (길이가 비정상이라 사용 안 함)")
        if loaded["qc_ratio"] < cfg["qc_min_ratio"]:
            print(f"  [제외] {loaded['name']}: 힘줄 때 신호가 휴식과 거의 같음 "
                  f"(x{loaded['qc_ratio']:.2f}) -> 센서 접촉 불량 의심")
            continue
        if loaded["dead_channels"]:
            print(f"  [경고] {loaded['name']}: {', '.join(loaded['dead_channels'])} 가 어떤 동작에도 반응 안 함 -> 센서/케이블 확인")
        if loaded["noisy_channels"]:
            print(f"  [경고] {loaded['name']}: {', '.join(loaded['noisy_channels'])} 가 쉴 때부터 노이즈가 큼 -> 전극 접촉 확인")
        if loaded["qc_weak"]:
            print(f"  [경고] {loaded['name']}: 반응이 약한 구간 {', '.join(loaded['qc_weak'])}")
        sessions.append(loaded)
    fist = [s for s in sessions if s["group"] == "fist"]
    rand = [s for s in sessions if s["group"] == "random"]
    print(f"  고정 순서 {len(fist)}개, Random {len(rand)}개")
    if len(fist) <= cfg["val_sessions"] or not rand:
        raise SystemExit(
            "데이터가 부족합니다. 아래 폴더에 CSV가 있는지 확인하세요.\n"
            f"  {cfg['fist_dir']}  (fist01.csv ...)\n"
            f"  {cfg['random_dir']}  (Random01.csv ...)")

    rng = random.Random(cfg["seed"])
    order = list(range(len(fist)))
    rng.shuffle(order)
    val = [fist[i] for i in sorted(order[:cfg["val_sessions"]])]
    train = [fist[i] for i in sorted(order[cfg["val_sessions"]:])]
    print("  학습:", ", ".join(s["name"] for s in train))
    print("  검증:", ", ".join(s["name"] for s in val))
    print("  시험:", ", ".join(s["name"] for s in rand))

    print("[2/6] 1초 구간 자르기 + STFT 계산 중...")
    Xtr, ytr, _ = windows_to_arrays(train, [make_windows(s, cfg, cfg["step_train_s"]) for s in train], cfg)
    Xva, yva, _ = windows_to_arrays(val, [make_windows(s, cfg, cfg["step_eval_s"]) for s in val], cfg)
    # 시험 채점은 MATLAB과 같은 기준(지시 구간 안에 완전히 들어간 창)으로
    zone_cfg = dict(cfg, label_mode="zone")
    Xte, yte, mte = windows_to_arrays(rand, [make_windows(s, zone_cfg, cfg["step_eval_s"]) for s in rand], cfg)

    mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)    # (ch,1,1)
    sd = Xtr.std(axis=(0, 2, 3), keepdims=True)[0].astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    norm = (mu, sd)
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xva = ((Xva - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    print(f"  입력 크기: {Xtr.shape[1:]} (채널, 주파수, 시간)")
    print("  학습 데이터 수:", {names[i]: int(c) for i, c in enumerate(np.bincount(ytr, minlength=n_cls))})

    print("[3/6] 학습 중...")
    model, hist, best_epoch = train_model(Xtr, ytr, Xva, yva, cfg, verbose=verbose)

    print("[4/6] 평가 중...")
    pte = predict_logits(model, Xte).argmax(1)
    cm = confusion(yte, pte, n_cls)
    win_acc = np.trace(cm) / cm.sum()
    finger = yte != names.index("Rest")
    finger_acc = (pte[finger] == yte[finger]).mean()
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    seg_true, seg_pred = [], []
    mte = np.array(mte)
    for key in sorted({(a, b) for a, b, _ in mte if b < 100}):
        sel = (mte[:, 0] == key[0]) & (mte[:, 1] == key[1])
        vals, cnt = np.unique(pte[sel], return_counts=True)
        seg_true.append(int(yte[sel][0]))
        seg_pred.append(int(vals[np.argmax(cnt)]))
    seg_acc = np.mean(np.array(seg_true) == np.array(seg_pred))

    rest_rms = compute_rest_rms(train, cfg)
    rt = realtime_simulation(model, rand, cfg, norm, rest_rms)
    rtm = realtime_metrics(rt, cfg)
    ops, params = count_ops(cfg)

    lines = [
        "=== Random 세션(순서 섞은 데이터) 시험 결과 ===",
        f"1초 단위 정확도 (휴식 포함 {n_cls}종): {win_acc * 100:.1f}%",
        f"1초 단위 정확도 (휴식 제외 동작 {n_cls - 1}종): {finger_acc * 100:.1f}%",
        f"한 번 힘주기(수축 구간) 다수결: {seg_acc * 100:.1f}% ({int(np.sum(np.array(seg_true) == np.array(seg_pred)))}/{len(seg_true)})",
        "클래스별 1초 단위 정답률: " + ", ".join(f"{n} {r * 100:.0f}%" for n, r in zip(names, recall)),
        "",
        f"=== 실시간 흉내 ({cfg['step_eval_s'] * 1000:.0f}ms마다 판단 / 켜기: 최근 {cfg['vote_n']}번 중 {cfg['vote_k']}번, "
        f"바꾸기: 연속 {cfg['vote_switch_k'] or cfg['vote_k']}번, 끄기: Rest {cfg['vote_release_n']}번 연속, "
        f"최소 유지 {cfg['vote_hold_s']}초, 휴식 게이트 {cfg['rest_gate'] or '끔'}) ===",
        f"안정 구간 정확도: {rtm['acc'] * 100:.1f}%",
        f"휴식 중 잘못 움직임 판단 비율: {rtm['false_move_rate'] * 100:.1f}%",
        f"힘주기 시작 -> 올바른 판단까지 걸린 시간(중앙값): {rtm['latency_median']:.2f}초",
        f"힘 빼기 -> 손가락 판단이 풀릴 때까지 걸린 시간(중앙값): {rtm['release_median']:.2f}초",
        f"근전도 시작 -> 올바른 판단 (제어 지연): 중앙값 {rtm['emg_on_median'] * 1000:.0f}ms, 90% {rtm['emg_on_p90'] * 1000:.0f}ms",
        f"근전도 끝 -> 풀림: 중앙값 {rtm['emg_off_median'] * 1000:.0f}ms",
        f"손가락 동작 인식: {rtm['detected']}번 성공 / {rtm['missed']}번 놓침",
        f"엉뚱한 손가락이 켜진 횟수: {rtm['false_events']}번 (총 {rtm['minutes']:.1f}분 동안)",
        "",
        "=== ESP32 부담 ===",
        f"가중치 수: {params}개 (float 기준 약 {params * 4 / 1024:.0f}KB)",
        f"한 번 판단에 곱셈 약 {ops / 1e6:.2f}백만 번 (MATLAB 모델은 약 231백만 번)",
        f"최고 시점: epoch {best_epoch}",
    ]
    report = "\n".join(lines)
    print("\n" + report + "\n")

    if export:
        print("[5/6] 그림 저장 중...")
        with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        plot_confusion(cm, names, f"Random test, 1s windows (acc {win_acc * 100:.1f}%)",
                       os.path.join(out_dir, "confusion_window.png"))
        plot_realtime(rt, cfg, os.path.join(out_dir, "realtime_random.png"))
        plot_history(hist, best_epoch, os.path.join(out_dir, "training_curve.png"))

        print("[6/6] ESP32용 파일 내보내는 중...")
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "norm": norm, "sos": sos},
                   os.path.join(out_dir, "emg_model.pt"))
        model_h = os.path.join(out_dir, "emg_model.h")
        vec_h = os.path.join(out_dir, "emg_test_vectors.h")
        export_model_header(model, cfg, sos, norm, model_h, rest_rms)
        export_test_vectors(model, rand, cfg, sos, norm, vec_h)
        for d in cfg["sketch_dirs"]:
            if os.path.isdir(d):
                shutil.copy(model_h, d)
                shutil.copy(vec_h, d)
                print(f"  {os.path.basename(d)} 폴더에 .h 파일 복사 완료")
        print(f"\n완료! 결과는 {out_dir} 폴더에 있어요.")

    return {"model": model, "norm": norm, "sos": sos, "rest_rms": rest_rms, "win_acc": win_acc, "finger_acc": finger_acc,
            "seg_acc": seg_acc, "rt": rtm, "val_loss": min(hist["val_loss"]),
            "val_acc": hist["val_acc"][best_epoch - 1], "cm": cm, "rand": rand}


# 빠른 모드: 0.5초 창을 50ms마다 판단 (9/24 데이터 실험 기준 제어 지연 중앙값 약 0.3초)
FAST_PRESET = {
    "epoch_s": 0.5, "step_eval_s": 0.05, "step_train_s": 0.1,
    "win": 64, "hop": 32, "nfft": 64, "band_start_bin": 2, "band_width": 2, "num_bands": 13,
    "label_delay_s": 0.05, "label_margin_s": 0.15, "contract_trim_start_s": 0.15,
    "relax_trim_start_s": 0.5, "relax_trim_end_s": 0.15,
    "vote_n": 3, "vote_k": 3, "vote_switch_k": 8, "rest_gate": 2.0,
    "vote_release_n": 3, "vote_hold_s": 0.5,
    "out_dir": os.path.join(BASE_DIR, "output_fast"),
}


if __name__ == "__main__":
    import sys
    cfg = copy.deepcopy(CFG)
    if "--fast" in sys.argv:
        cfg.update(FAST_PRESET)
        print("===== 빠른 모드로 학습 (0.5초 창, 50ms마다 판단) =====")
    run(cfg)
