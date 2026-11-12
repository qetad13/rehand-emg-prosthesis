"""
EMG 손가락 동작 분류 v2: 스트리밍 특징 + 작은 1D 시간 합성곱(TCN) -> ESP32용 C 헤더 내보내기

실행 방법 (VS Code 터미널):
    python train_v2.py            학습 + Random 시험 + 내보내기
    python train_v2.py --cv       위에 더해 고정 순서 세션 15개로 5겹 교차검증도 함 (약 5분 더 걸림)

데이터: data/Fixed (고정 순서, 학습/교차검증), data/Random (무작위 순서, 시험 전용). 읽기는 emg_data.py

v1(STFT + 2D CNN, v1-stft-cnn 브랜치)과 달라진 점
  - 입력: 0.75초 창의 STFT 그림 대신, 25ms마다 계산하는 특징 22개
      채널별 로그 진폭(쉴 때 대비) / 주파수 비 2개 / 영교차율 + 채널 쌍 상관 6개
      (v1은 창마다 채널별로 크기를 정규화해서 "어느 채널이 얼마나 큰지"를 버렸는데,
       손가락마다 채널별 크기 비율이 뚜렷하게 달라서 이 정보가 가장 유용함)
  - 모델: 2D CNN 대신 과거만 보는 1D 시간 합성곱 3층 (약 6천 가중치, 1회 판단 약 6천 곱셈)
      최근 0.375초를 보고 25ms마다 판단
  - 라벨: 화면 지시 시각 + 고정 지연 대신, 근전도가 실제로 올라온/내려간 시점 기준
      (사람마다, 매번 반응 시간이 달라서 고정 지연은 경계 라벨이 부정확함)
  - 출력 규칙: 횟수 투표 대신, 평활한 확률이 문턱을 넘으면 켬 (애매한 초반 판단으로 엉뚱한 손가락이 켜지는 것을 줄임)
  - 학습 데이터 늘리기: 전극 이득이 달라진 상황(로그 진폭에 상수를 더함)과
      약하게/세게 힘준 상황(수축 구간의 신호만 0.5~2배로 바꾼 복사본)을 흉내

결과 (output_v2 폴더):
    report.txt, confusion_window.png, realtime_random.png, emg2_model.pt
    emg2_model.h, emg2_test_vectors.h  -> ../firmware/emg_board_v2/ 에 자동 복사
"""

import os
import sys
import copy
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn

import emg_data as base       # 데이터 읽기, 필터, 구간 정보, 실시간 지표


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG2 = {
    "out_dir": os.path.join(BASE_DIR, "output_v2"),
    "sketch_dirs": [os.path.join(BASE_DIR, "..", "firmware", "emg_board_v2")],

    # ---- 특징 ----
    "hop": 25,                  # 25ms마다 특징 계산 + 판단
    "k_hops": 2,                # 특징은 최근 2 hop(50ms)의 누적값으로 계산
    "feat_f1": True,            # 주파수 비 1: 1차 차분 크기 / 신호 크기
    "feat_f2": True,            # 주파수 비 2: 2차 차분 크기 / 1차 차분 크기
    "feat_zc": True,            # 영교차율
    "feat_cr": True,            # 채널 쌍 상관

    # ---- 라벨 (근전도 실제 시작/끝 기준) ----
    "label_on_skip_ms": 50,     # 시작 직후 이만큼은 손실에서 제외 (아직 어떤 동작인지 알 수 없는 구간)
    "label_on_pre_ms": 75,      # 시작 직전 이만큼도 제외
    "label_off_pre_ms": 150,    # 끝 직전
    "label_off_post_ms": 50,    # 끝 직후

    # ---- 모델 / 학습 ----
    "widths": [24, 24, 32],
    "kernels": [3, 3, 3],
    "dilations": [1, 2, 4],     # 보는 범위 = 1 + 2*(1+2+4) = 15 프레임 = 0.375초
    "dropout": 0.1,
    "steps": 1500,
    "seq_len": 240,             # 학습할 때 한 번에 넣는 길이 (프레임, 6초)
    "batch_size": 32,
    "lr": 3e-3,
    "weight_decay": 1e-3,
    "label_smoothing": 0.05,
    "aug_gain_common": 0.15,    # 힘 세기 변화 흉내: 모든 채널 로그 진폭에 같은 값을 더함 (표준편차)
    "aug_gain_channel": 0.10,   # 전극 이득 변화 흉내: 채널마다 다른 값을 더함 (표준편차)
    "aug_force_copies": 1,      # 약하게/세게 힘준 상황 흉내: 세션마다 이 개수만큼 복사본을 만들어 학습에 추가 (0이면 끔)
    "aug_force_range": [0.5, 2.0],   # 수축 구간마다 신호 크기를 이 범위의 임의 배율로 바꿈 (잡음 크기는 그대로 유지)
                                # (복사본을 2~3개로 늘리면 약한 힘은 더 잘 잡지만 평소 힘에서 오출력이 늘어남)
    "seed": 0,

    # ---- 출력 규칙 (ESP32의 emg2_decide와 같음) ----
    "a_on": 0.5,                # 확률 평활 계수 (클수록 느리고 안정적)
    "th_on": 0.8,               # 평활한 확률이 이 값 이상이면 켬
    "rel_n": 3,                 # 휴식 판단이 이만큼 연속이면 끔 (3번 = 75ms)
    "hold_n": 4,                # 출력이 바뀐 뒤 이만큼은 다시 안 바꿈 (4번 = 100ms)
    "switch_free_classes": ["Fist"],   # 휴식 없이 바로 바꿀 수 있는 동작
    "th_sw": 0.9,               # 바로 바꿀 때 필요한 평활 확률
    "sw_n": 6,                  # 위 조건이 이만큼 연속이어야 바꿈
    "rest_gate": 1.5,           # 신호 크기가 쉴 때의 이 배수보다 작으면 무조건 휴식 (0이면 끔)
    "warmup_frames": 20,        # 전원을 켠 직후 필터가 안정될 때까지 (0.5초) 무조건 휴식
    "strength_hops": 8,         # 힘 세기를 재는 구간 (8 hop = 0.2초)

    # ---- ESP32 테스트 데이터 ----
    "test_chunk_s": 2.0,
}


# =====================================================================
# 특징: ESP32의 emg2.cpp와 같은 계산
# =====================================================================
def channel_pairs(n_ch):
    return [(a, b) for a in range(n_ch) for b in range(a + 1, n_ch)]


def hop_accumulate(filt, hop):
    """hop마다의 누적값 (emg2_front_push와 같음): s0, s1, s2, zc (n, ch) / cr (n, 쌍)"""
    n_ch = filt.shape[1]
    n = len(filt) // hop
    x = filt[:n * hop].astype(np.float64)
    xp = np.vstack([np.zeros((1, n_ch)), x[:-1]])
    d1 = x - xp
    d1p = np.vstack([np.zeros((1, n_ch)), d1[:-1]])
    d2 = d1 - d1p
    r = lambda a: a.reshape(n, hop, -1).sum(axis=1)
    cr = np.stack([x[:, a] * x[:, b] for a, b in channel_pairs(n_ch)], axis=1)
    return {"s0": r(x * x), "s1": r(d1 * d1), "s2": r(d2 * d2),
            "zc": r(((x * xp) < 0).astype(np.float64)), "cr": r(cr)}


def sliding_sum(a, k):
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[k:] = c[k:] - c[:-k]
    return out


def compute_features(acc, cfg2, rest_pow):
    """(프레임 수, 특징 수). 앞의 채널 수만큼이 로그 진폭 (emg2_back_frame과 같은 순서)"""
    hop, k = cfg2["hop"], cfg2["k_hops"]
    s0, s1, s2, zc, cr = [sliding_sum(acc[n], k) for n in ("s0", "s1", "s2", "zc", "cr")]
    n = float(k * hop)
    eps = 1e-3 * n
    n_ch = s0.shape[1]
    out = [0.5 * np.log((s0 / n + 1e-3) / rest_pow)]
    if cfg2["feat_f1"]:
        out.append(0.5 * np.log((s1 + eps) / (s0 + eps)))
    if cfg2["feat_f2"]:
        out.append(0.5 * np.log((s2 + eps) / (s1 + eps)))
    if cfg2["feat_zc"]:
        out.append(zc / n)
    if cfg2["feat_cr"]:
        out.append(np.stack([cr[:, i] / np.sqrt(s0[:, a] * s0[:, b] + eps * eps)
                             for i, (a, b) in enumerate(channel_pairs(n_ch))], axis=1))
    return np.hstack(out).astype(np.float32)


def compute_strength(acc, cfg2, rest_pow, gain_off=None):
    """힘 세기: 최근 strength_hops개 hop의 신호 크기 / 쉴 때 크기, 채널 중 최댓값 (emg2_back_frame과 같음)"""
    h = cfg2["strength_hops"]
    p = sliding_sum(acc["s0"], h) / (h * cfg2["hop"])
    r = np.sqrt(p / rest_pow)
    if gain_off is not None:
        r = r * np.exp(-np.asarray(gain_off))
    return r.max(axis=1).astype(np.float32)


def rest_power(sessions):
    """학습 세션 휴식 구간의 채널별 신호 제곱 평균 (로그 진폭의 0 기준)"""
    vals = []
    for s in sessions:
        for (a, e, lab, kind, k) in s["zones"]:
            if kind == "relax" and s["valid"][a:e].all():
                vals.append((s["filt"][a + 800:e].astype(np.float64) ** 2).mean(axis=0))
    return np.median(np.array(vals), axis=0)


# =====================================================================
# 라벨: 근전도가 실제로 올라온/내려간 시점 기준
# =====================================================================
def emg_events(sess, cfg):
    """수축 구간마다 (지시 시작, 지시 끝, 라벨, 근전도 시작, 근전도 끝). 못 찾으면 None
    (emg_data.realtime_metrics가 제어 지연을 잴 때 쓰는 정의와 같음)"""
    rest = cfg["class_names"].index("Rest")
    env = np.sqrt(np.convolve((sess["filt"] ** 2).max(axis=1), np.ones(100) / 100, mode="same"))
    relax = [(s, e) for (s, e, lab, kind, k) in sess["zones"] if kind == "relax" and sess["valid"][s:e].all()]
    base_level = np.median(np.concatenate([env[s + 800:e] for (s, e) in relax]))
    events = []
    for (s, e, lab, kind, k) in sess["zones"]:
        if kind != "contract" or lab == rest:
            continue
        peak = np.median(env[s + 1000:e])
        thr = base_level + 0.3 * (peak - base_level)
        a0 = max(0, s - 500)
        up = np.flatnonzero(env[a0:e] > thr)
        on = a0 + up[0] if len(up) else None
        down = np.flatnonzero(env[e - 500:min(len(env), e + 3000)] < thr)
        down = down[down > 400]
        off = e - 500 + down[0] if len(down) else None
        events.append((s, e, lab, on, off))
    return events


def make_labels(sess, cfg, cfg2):
    """샘플별 라벨과 '손실에 쓸지' 표시"""
    n = sess["n"]
    rest = cfg["class_names"].index("Rest")
    lab = np.full(n, rest, dtype=np.int64)
    use = sess["valid"].copy()
    for (s, e, g, on, off) in emg_events(sess, cfg):
        if on is None or off is None or off <= on:
            # 시작/끝을 못 찾은 구간: 지시 시각 + 0.1초로 라벨하고 경계 주변은 넓게 제외
            lab[s + 100:e + 100] = g
            use[max(0, s - 200):s + 400] = False
            use[e - 200:e + 500] = False
            continue
        lab[on:off] = g
        use[max(0, on - cfg2["label_on_pre_ms"]):on + cfg2["label_on_skip_ms"]] = False
        use[max(0, off - cfg2["label_off_pre_ms"]):off + cfg2["label_off_post_ms"]] = False
    use[:500] = False                        # 파일 시작 직후(필터 안정화)
    return lab, use


def force_scaled_filt(sess, cfg, rng, lo, hi):
    """약하게/세게 힘준 상황 흉내: 수축 구간마다 신호를 lo~hi배(구간마다 다른 값)로 바꾼 신호.
    약하게 만들 때는 줄어든 만큼 같은 세션의 휴식 신호를 더해서 잡음 크기는 그대로 유지
    (실제로 약하게 힘주면 근전도만 작아지고 센서 잡음은 그대로이기 때문)"""
    filt = sess["filt"].copy()
    noise = np.concatenate([sess["filt"][a + 800:e] for (a, e, lab, kind, k) in sess["zones"]
                            if kind == "relax" and sess["valid"][a:e].all()])
    for (s, e, g, on, off) in emg_events(sess, cfg):
        a = max(0, (on if on is not None else s) - 150)
        b = min(len(filt), (off if off is not None else e) + 150)
        alpha = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        seg = filt[a:b] * alpha
        if alpha < 1.0 and len(noise) > (b - a):
            i0 = rng.randint(0, len(noise) - (b - a))
            seg = seg + np.sqrt(1.0 - alpha * alpha) * noise[i0:i0 + (b - a)]
        filt[a:b] = seg
    return filt.astype(np.float32)


def prepare(sessions, cfg, cfg2, rest_pow, force_copies=0, seed=0):
    """세션마다 프레임 단위 특징/라벨/힘 세기. force_copies > 0이면 힘 세기를 바꾼 복사본도 함께 (학습용)"""
    out = []
    hop = cfg2["hop"]
    rng = np.random.RandomState(seed)
    for s in sessions:
        key = f"acc{hop}"
        if key not in s:
            s[key] = hop_accumulate(s["filt"], hop)
        x = compute_features(s[key], cfg2, rest_pow)
        lab, use = make_labels(s, cfg, cfg2)
        ends = (np.arange(len(x)) + 1) * hop          # 프레임이 끝나는 샘플 번호
        out.append({"X": x, "y": lab[ends - 1], "m": use[ends - 1], "ends": ends, "sess": s, "orig": True,
                    "strength": compute_strength(s[key], cfg2, rest_pow)})
        for _ in range(force_copies):                 # 라벨(시작/끝 시점)은 원본 것을 그대로 사용
            acc = hop_accumulate(force_scaled_filt(s, cfg, rng, *cfg2["aug_force_range"]), hop)
            out.append({"X": compute_features(acc, cfg2, rest_pow), "y": lab[ends - 1], "m": use[ends - 1],
                        "ends": ends, "sess": s, "orig": False,
                        "strength": compute_strength(acc, cfg2, rest_pow)})
    return out


# =====================================================================
# 모델: 과거만 보는 1D 시간 합성곱
# =====================================================================
class EmgTCN(nn.Module):
    def __init__(self, n_in, n_classes, widths, kernels, dilations, dropout):
        super().__init__()
        self.spec = list(zip(widths, kernels, dilations))
        convs = []
        c = n_in
        for w, k, d in self.spec:
            convs.append(nn.Conv1d(c, w, k, dilation=d))
            c = w
        self.convs = nn.ModuleList(convs)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, n_classes)
        self.rf = 1 + sum((k - 1) * d for _, k, d in self.spec)    # 보는 범위 (프레임)

    def forward(self, x):                    # x: (배치, 시간, 특징) -> (배치, 시간, 클래스)
        h = x.transpose(1, 2)
        for conv, (w, k, d) in zip(self.convs, self.spec):
            h = torch.relu(conv(nn.functional.pad(h, ((k - 1) * d, 0))))   # 왼쪽에만 0을 채움 = 과거만 봄
        return self.fc(self.drop(h.transpose(1, 2)))


def build_model(n_in, n_classes, cfg2):
    return EmgTCN(n_in, n_classes, cfg2["widths"], cfg2["kernels"], cfg2["dilations"], cfg2["dropout"])


def train_model(data, n_classes, n_ch, cfg2, seed, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    x_all = np.concatenate([d["X"][d["m"]] for d in data])
    mu = x_all.mean(axis=0)
    sd = np.maximum(x_all.std(axis=0), 1e-3)
    y_all = np.concatenate([d["y"][d["m"]] for d in data])
    counts = np.bincount(y_all, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** 0.5     # 휴식이 많은 것을 절반쯤 보정

    model = build_model(data[0]["X"].shape[1], n_classes, cfg2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg2["lr"], weight_decay=cfg2["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg2["lr"], total_steps=cfg2["steps"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), reduction="none",
                                  label_smoothing=cfg2["label_smoothing"])
    t_len, bsz = cfg2["seq_len"], cfg2["batch_size"]
    model.train()
    for step in range(cfg2["steps"]):
        xb, yb, mb = [], [], []
        for _ in range(bsz):
            d = data[rng.randint(len(data))]
            i = rng.randint(0, len(d["X"]) - t_len)
            x = d["X"][i:i + t_len].copy()
            x[:, :n_ch] += rng.randn() * cfg2["aug_gain_common"] + rng.randn(n_ch) * cfg2["aug_gain_channel"]
            xb.append((x - mu) / sd)
            yb.append(d["y"][i:i + t_len])
            mb.append(d["m"][i:i + t_len])
        xb = torch.from_numpy(np.stack(xb).astype(np.float32))
        yb = torch.from_numpy(np.stack(yb))
        mb = torch.from_numpy(np.stack(mb).astype(np.float32))
        mb[:, :model.rf] = 0                 # 잘라낸 조각의 앞부분은 과거가 없으므로 손실에서 제외
        loss = (loss_fn(model(xb).reshape(-1, n_classes), yb.reshape(-1)) * mb.reshape(-1)).sum() / mb.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 250 == 0 or step == cfg2["steps"] - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    model.eval()
    return model, (mu.astype(np.float32), sd.astype(np.float32))


def predict_probs(model, norm, x):
    with torch.no_grad():
        logits = model(torch.from_numpy(((x - norm[0]) / norm[1]).astype(np.float32))[None])[0]
        return torch.softmax(logits, dim=1).numpy()


# =====================================================================
# 출력 규칙 (ESP32의 emg2_decide와 같음)
# =====================================================================
def decide_sequence(probs, strength, cfg, cfg2):
    names = cfg["class_names"]
    rest = names.index("Rest")
    free = {names.index(n) for n in cfg2["switch_free_classes"] if n in names}
    a_on, th_on, th_sw = cfg2["a_on"], cfg2["th_on"], cfg2["th_sw"]
    onehot = np.zeros(len(names), dtype=np.float32)
    onehot[rest] = 1.0
    ps = onehot.copy()
    out = []
    output, since, rest_run, sw_run, sw_c = rest, 1 << 30, 0, 0, -1
    for i in range(len(probs)):
        p = probs[i]
        if i < cfg2["warmup_frames"] or (cfg2["rest_gate"] and strength[i] < cfg2["rest_gate"]):
            p = onehot
        k = int(np.argmax(p))
        ps = a_on * ps + (1.0 - a_on) * p
        q = ps.copy()
        q[rest] = -1.0
        c = int(np.argmax(q))
        since += 1
        rest_run = rest_run + 1 if k == rest else 0
        cand = output
        if output != rest and rest_run >= cfg2["rel_n"]:
            cand = rest
        elif output == rest:
            if ps[c] >= th_on and k != rest:
                cand = c
        elif c != output and ps[c] >= th_sw and c in free:
            sw_run = sw_run + 1 if sw_c == c else 1
            sw_c = c
            if sw_run >= cfg2["sw_n"]:
                cand = c
        else:
            sw_run = 0
        if cand != output and (since >= cfg2["hold_n"] or (cand != rest and output != rest)):
            if cand == rest:
                ps = onehot.copy()
            output, since, sw_run = cand, 0, 0
        out.append(output)
    return np.array(out)


# =====================================================================
# 평가
# =====================================================================
def evaluate(preds, cfg, cfg2, rf):
    """preds: [(세션 데이터, 확률)] -> 창 단위 정확도 + 실시간 지표 (v1과 같은 정의)"""
    fs, hop = cfg["fs"], cfg2["hop"]
    names = cfg["class_names"]
    rest = names.index("Rest")
    span = rf * hop                                   # 한 번 판단이 보는 길이 (샘플)
    results, y_true, y_pred, seg_true, seg_pred = [], [], [], [], []
    for d, probs in preds:
        raw = probs.argmax(axis=1)
        voted = decide_sequence(probs, d["strength"], cfg, cfg2)
        sess, ends = d["sess"], d["ends"]
        results.append({"sess": sess, "ends": ends, "raw": raw, "voted": voted})
        for (s, e, lab, kind, k) in sess["zones"]:
            if not sess["valid"][s:e].all():
                continue
            # v1 채점과 같은 기준: 지시 구간 안에 완전히 들어간 판단만 (구간 앞부분은 반응 시간만큼 제외)
            if kind == "contract":
                trim = cfg["first_rest_trim_start_s"] if k == 0 else cfg["contract_trim_start_s"]
                last = e
            else:
                trim = cfg["relax_trim_start_s"]
                last = e - int(cfg["relax_trim_end_s"] * fs)
            sel = (ends - span >= s + int(trim * fs)) & (ends <= last)
            y_true += [lab] * int(sel.sum())
            y_pred += raw[sel].tolist()
            if kind == "contract" and sel.any():
                vals, cnt = np.unique(raw[sel], return_counts=True)
                seg_true.append(lab)
                seg_pred.append(int(vals[np.argmax(cnt)]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rt_cfg = dict(cfg, epoch_s=span / fs, step_eval_s=hop / fs, vote_n=1)
    rtm = base.realtime_metrics(results, rt_cfg)
    cm = base.confusion(y_true, y_pred, len(names))
    return {
        "cm": cm, "win_acc": float((y_true == y_pred).mean()),
        "finger_acc": float((y_true == y_pred)[y_true != rest].mean()),
        "recall": np.diag(cm) / np.maximum(cm.sum(axis=1), 1),
        "seg_ok": int(np.sum(np.array(seg_true) == np.array(seg_pred))), "seg_n": len(seg_true),
        "rt": rtm, "results": results,
    }


def report_lines(title, ev, cfg, cfg2):
    names = cfg["class_names"]
    rt = ev["rt"]
    hop_ms = cfg2["hop"] * 1000 / cfg["fs"]
    return [
        f"=== {title} ===",
        f"창 단위 정확도 (휴식 포함 {len(names)}종): {ev['win_acc'] * 100:.1f}%",
        f"창 단위 정확도 (휴식 제외 동작 {len(names) - 1}종): {ev['finger_acc'] * 100:.1f}%",
        f"한 번 힘주기(수축 구간) 다수결: {ev['seg_ok'] / max(ev['seg_n'], 1) * 100:.1f}% ({ev['seg_ok']}/{ev['seg_n']})",
        "클래스별 창 단위 정답률: " + ", ".join(f"{n} {r * 100:.0f}%" for n, r in zip(names, ev["recall"])),
        f"[실시간 흉내: {hop_ms:.0f}ms마다 판단, 켜기 확률 {cfg2['th_on']}, 끄기 Rest {cfg2['rel_n']}번 연속, "
        f"최소 유지 {cfg2['hold_n'] * hop_ms / 1000:.2f}초, 휴식 게이트 {cfg2['rest_gate'] or '끔'}]",
        f"안정 구간 정확도: {rt['acc'] * 100:.1f}%",
        f"휴식 중 잘못 움직임 판단 비율: {rt['false_move_rate'] * 100:.2f}%",
        f"근전도 시작 -> 올바른 판단 (제어 지연): 중앙값 {rt['emg_on_median'] * 1000:.0f}ms, 90% {rt['emg_on_p90'] * 1000:.0f}ms",
        f"근전도 끝 -> 풀림: 중앙값 {rt['emg_off_median'] * 1000:.0f}ms",
        f"손가락 동작 인식: {rt['detected']}번 성공 / {rt['missed']}번 놓침",
        f"엉뚱한 손가락이 켜진 횟수: {rt['false_events']}번 (총 {rt['minutes']:.1f}분 동안)",
        "",
    ]


def class_tables(data, cfg, n_ch):
    """사용자 보정 기준(동작별 채널별 로그 진폭 중앙값)과 동작별 힘 세기 100% 기준"""
    n_cls = len(cfg["class_names"])
    logamp = np.zeros((n_cls, n_ch), dtype=np.float32)
    full = np.ones(n_cls, dtype=np.float32)
    data = [d for d in data if d.get("orig", True)]          # 힘 세기를 바꾼 복사본은 제외
    for g in range(n_cls):
        x = np.concatenate([d["X"][(d["y"] == g) & d["m"], :n_ch] for d in data])
        s = np.concatenate([d["strength"][(d["y"] == g) & d["m"]] for d in data])
        if len(x):
            logamp[g] = np.median(x, axis=0)
            full[g] = np.median(s)
    return logamp, full


def calibration_simulation(model, norm, data, cfg, cfg2, cal_logamp, n_ch, lo, hi, trials=20, seed=0):
    """전극을 다시 붙여 채널별 이득이 달라진 상황 흉내 (채널마다 lo~hi배).
    첫 세션의 수축 구간으로 '사용자 보정'을 하고 나머지 세션에서 프레임 단위 정확도를 비교.
    돌려주는 값: (이득 변화 없음, 변화 + 보정 없음, 변화 + 보정)"""
    rest = cfg["class_names"].index("Rest")
    rng = np.random.RandomState(seed)
    cal, test = data[0], data[1:]

    def accuracy(shift):
        accs = []
        for d in test:
            m = d["m"].copy()
            m[:cfg2["warmup_frames"]] = False
            x = d["X"].copy()
            x[:, :n_ch] += shift
            accs.append((predict_probs(model, norm, x).argmax(axis=1)[m] == d["y"][m]).mean())
        return float(np.mean(accs))

    acc_none, acc_cal = [], []
    for _ in range(trials):
        g = np.log(rng.uniform(lo, hi, n_ch)).astype(np.float32)
        # 보정: 동작별로 측정한 로그 진폭과 기준의 차이를 채널마다 중앙값으로 (펌웨어의 보정과 같은 계산)
        diffs = []
        for cls in range(len(cfg["class_names"])):
            sel = (cal["y"] == cls) & cal["m"]
            if cls == rest or not sel.any():
                continue
            diffs.append(np.median(cal["X"][sel, :n_ch] + g, axis=0) - cal_logamp[cls])
        off = np.clip(np.median(np.array(diffs), axis=0), -0.7, 0.7)
        acc_none.append(accuracy(g))
        acc_cal.append(accuracy(g - off))
    return accuracy(np.zeros(n_ch, np.float32)), float(np.mean(acc_none)), float(np.mean(acc_cal))


# =====================================================================
# ESP32용 내보내기
# =====================================================================
def count_ops(model, n_classes):
    ops = sum(c.weight.numel() for c in model.convs) + model.fc.weight.numel()
    params = sum(p.numel() for p in model.parameters())
    return ops, params


def export_model_header(model, cfg, cfg2, sos, norm, rest_pow, cal_logamp, strength_full, path):
    names = cfg["class_names"]
    n_ch = len(cfg["channels"])
    n_feat = len(norm[0])
    hist = [(k - 1) * d + 1 for _, k, d in model.spec]
    ins = [n_feat] + [w for w, _, _ in model.spec[:-1]]
    free_mask = sum(1 << names.index(n) for n in cfg2["switch_free_classes"] if n in names)
    free_names = ", ".join(n for n in cfg2["switch_free_classes"] if n in names) or "없음"
    quoted = ", ".join(f'"{n}"' for n in names)
    ca = base.c_array

    h = ["// 자동 생성 파일 (train_v2.py) - 직접 수정하지 마세요\n",
         "#ifndef EMG2_MODEL_H\n#define EMG2_MODEL_H\n\n",
         f"#define EMG2_FS            {cfg['fs']}\n",
         f"#define EMG2_NUM_CH        {n_ch}\n",
         f"#define EMG2_NUM_PAIRS     {n_ch * (n_ch - 1) // 2}\n",
         f"#define EMG2_HOP           {cfg2['hop']}      // 이만큼의 샘플마다 판단 ({cfg2['hop'] * 1000 // cfg['fs']}ms)\n",
         f"#define EMG2_K             {cfg2['k_hops']}\n",
         f"#define EMG2_FEAT_F1       {int(cfg2['feat_f1'])}\n",
         f"#define EMG2_FEAT_F2       {int(cfg2['feat_f2'])}\n",
         f"#define EMG2_FEAT_ZC       {int(cfg2['feat_zc'])}\n",
         f"#define EMG2_FEAT_CR       {int(cfg2['feat_cr'])}\n",
         f"#define EMG2_NUM_FEAT      {n_feat}\n",
         f"#define EMG2_NUM_SOS       {len(sos)}\n",
         f"#define EMG2_NUM_LAYERS    {len(model.spec)}\n",
         f"#define EMG2_MAX_WIDTH     {max(w for w, _, _ in model.spec)}\n",
         f"#define EMG2_EMBED         {model.spec[-1][0]}\n",
         f"#define EMG2_HIST_FLOATS   {sum(hl * i for hl, i in zip(hist, ins))}\n",
         f"#define EMG2_RF_FRAMES     {model.rf}     // 한 번 판단이 보는 범위 (프레임)\n",
         f"#define EMG2_NUM_CLASSES   {len(names)}\n",
         f"#define EMG2_REST_CLASS    {names.index('Rest')}\n",
         f"#define EMG2_STRENGTH_HOPS {cfg2['strength_hops']}\n",
         f"#define EMG2_WARMUP_FRAMES {cfg2['warmup_frames']}\n",
         f"#define EMG2_A_ON          {cfg2['a_on']:.3f}f\n",
         f"#define EMG2_TH_ON         {cfg2['th_on']:.3f}f\n",
         f"#define EMG2_REL_N         {cfg2['rel_n']}\n",
         f"#define EMG2_HOLD_N        {cfg2['hold_n']}\n",
         f"#define EMG2_TH_SW         {cfg2['th_sw']:.3f}f\n",
         f"#define EMG2_SW_N          {cfg2['sw_n']}\n",
         f"#define EMG2_SWITCH_FREE_MASK {free_mask}   // 휴식 없이 바로 바꿀 수 있는 동작: {free_names}\n",
         f"#define EMG2_REST_GATE     {float(cfg2['rest_gate'] or 0.0):.3f}f\n\n",
         "// 모델이 쓰는 채널 (CSV의 Ch 번호 = 보드의 EMG_PINS 순서)\n",
         f"static const int EMG2_CHANNEL_IDX[EMG2_NUM_CH] = {{ {', '.join(str(c) for c in cfg['channels'])} }};\n",
         f"static const char *const EMG2_CLASS_NAMES[EMG2_NUM_CLASSES] = {{ {quoted} }};\n\n",
         "// 사용자 보정 기준: 학습 데이터에서 동작별[행] 채널별[열] 로그 진폭 중앙값 (쉴 때 = 0)\n",
         ca("EMG2_CAL_LOGAMP", cal_logamp),
         "// 동작별 힘 세기 100% 기준: 학습 데이터에서 평소 힘으로 줬을 때의 세기(쉴 때의 몇 배) 중앙값\n",
         ca("EMG2_STRENGTH_FULL", strength_full),
         "\ntypedef struct {\n  int in, out, k, dil, hist;   // 입력 수, 출력 수, 탭 수, 탭 간격, 기억할 과거 길이\n"
         "  const float *w;              // 가중치 [출력][탭][입력]\n  const float *b;\n} Emg2LayerDef;\n\n",
         "#ifdef EMG2_MODEL_DATA\n\n",
         "// 필터: 행마다 b0 b1 b2 a0 a1 a2 (고역통과 2단 + 60Hz 노치), v1과 같은 계수\n",
         ca("EMG2_SOS", sos),
         "// 로그 진폭의 0 기준: 학습 데이터 휴식 구간의 채널별 신호 제곱 평균\n",
         ca("EMG2_REST_POW", rest_pow),
         ca("EMG2_FEAT_MEAN", norm[0]),
         ca("EMG2_FEAT_STD", norm[1])]
    for i, conv in enumerate(model.convs):
        w = conv.weight.detach().numpy().transpose(0, 2, 1)      # (출력, 입력, 탭) -> (출력, 탭, 입력)
        h.append(ca(f"EMG2_L{i}_W", w))
        h.append(ca(f"EMG2_L{i}_B", conv.bias.detach().numpy()))
    h.append("static const Emg2LayerDef EMG2_LAYERS[EMG2_NUM_LAYERS] = {\n")
    for i, ((w, k, d), n_in, hl) in enumerate(zip(model.spec, ins, hist)):
        h.append(f"  {{ {n_in}, {w}, {k}, {d}, {hl}, EMG2_L{i}_W, EMG2_L{i}_B }},\n")
    h.append("};\n")
    h.append(ca("EMG2_FC_W", model.fc.weight.detach().numpy()))
    h.append(ca("EMG2_FC_B", model.fc.bias.detach().numpy()))
    h.append("\n#endif  // EMG2_MODEL_DATA\n#endif  // EMG2_MODEL_H\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))


def reference_from_chunk(model, norm, chunk_raw, cfg, cfg2, sos, rest_pow):
    """ESP32 테스트와 똑같이: 모든 상태를 0에서 시작해 chunk 전체를 흘려보낸 뒤 마지막 프레임의 확률과 힘 세기"""
    from scipy.signal import sosfilt
    filt = sosfilt(sos, chunk_raw.astype(np.float32), axis=0).astype(np.float32)
    acc = hop_accumulate(filt, cfg2["hop"])
    p = predict_probs(model, norm, compute_features(acc, cfg2, rest_pow))
    return p[-1], float(compute_strength(acc, cfg2, rest_pow)[-1])


def export_test_vectors(model, norm, sessions, cfg, cfg2, sos, rest_pow, path):
    fs = cfg["fs"]
    chunk = int(cfg2["test_chunk_s"] * fs)
    names = cfg["class_names"]
    rest = names.index("Rest")
    rng = np.random.RandomState(cfg2["seed"])
    picked = []
    for lab in range(len(names)):
        cands = []
        for sess in sessions:
            for (s, e, zl, kind, k) in sess["zones"]:
                if zl != lab or (lab == rest and kind != "relax"):
                    continue
                end = (s + e) // 2 + 500
                if end - chunk >= 0 and end <= sess["n"]:
                    cands.append((sess, end))
        if cands:
            sess, end = cands[rng.randint(len(cands))]
            picked.append((lab, sess, end))

    h = ["// 자동 생성 파일 (train_v2.py): ESP32 정답 맞추기 테스트용\n",
         "#ifndef EMG2_TEST_VECTORS_H\n#define EMG2_TEST_VECTORS_H\n#include <stdint.h>\n\n",
         f"#define EMG2_TEST_COUNT {len(picked)}\n#define EMG2_TEST_LEN   {chunk}\n\n"]
    raws, probs, strengths = [], [], []
    for i, (lab, sess, end) in enumerate(picked):
        raw = sess["raw"][end - chunk:end]
        p, st = reference_from_chunk(model, norm, raw, cfg, cfg2, sos, rest_pow)
        raws.append(np.round(raw.T).astype(np.int16))
        probs.append(p)
        strengths.append(st)
        h.append(f"// [{i}] {names[lab]}: {sess['name']} {(end - chunk) / fs:.1f}~{end / fs:.1f}s\n")
    n_ch = raws[0].shape[0]
    h.append(f"\nstatic const int16_t EMG2_TEST_RAW[{len(raws)}][{n_ch}][{chunk}] = {{\n")
    for r in raws:
        h.append("  {\n")
        for c in range(n_ch):
            h.append("    { " + ", ".join(str(int(v)) for v in r[c]) + " },\n")
        h.append("  },\n")
    h.append("};\n\n")
    h.append(f"static const int EMG2_TEST_LABEL[{len(raws)}] = {{ {', '.join(str(l) for l, _, _ in picked)} }};\n")
    h.append(f"static const int EMG2_TEST_EXPECTED_PRED[{len(raws)}] = {{ {', '.join(str(int(np.argmax(p))) for p in probs)} }};\n")
    h.append(f"static const float EMG2_TEST_EXPECTED_STRENGTH[{len(raws)}] = {{ {', '.join(f'{s:.4f}f' for s in strengths)} }};\n")
    h.append(f"static const float EMG2_TEST_EXPECTED_PROB[{len(raws)}][{len(names)}] = {{\n")
    for p in probs:
        h.append("  { " + ", ".join(f"{v:.6f}f" for v in p) + " },\n")
    h.append("};\n\n#endif\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(h))


# =====================================================================
# 메인
# =====================================================================
def load_sessions(cfg):
    sos = base.design_sos(cfg)
    listed = base.list_sessions(cfg)
    used = {g for s in listed for g in s["order"]}
    cfg["class_names"] = [c for c in cfg["all_classes"] if c in used]
    sessions = []
    for s in listed:
        loaded = base.load_session(s, cfg, sos)
        if loaded is None:
            continue
        if loaded["bad_zones"]:
            print(f"  [제외 구간] {loaded['name']}: {', '.join(loaded['bad_zones'])} (길이가 비정상이라 사용 안 함)")
        if loaded["qc_ratio"] < cfg["qc_min_ratio"]:
            print(f"  [제외] {loaded['name']}: 힘줄 때 신호가 휴식과 거의 같음 (x{loaded['qc_ratio']:.2f}) -> 센서 접촉 불량 의심")
            continue
        if loaded["dead_channels"]:
            print(f"  [경고] {loaded['name']}: {', '.join(loaded['dead_channels'])} 가 어떤 동작에도 반응 안 함")
        if loaded["noisy_channels"]:
            print(f"  [경고] {loaded['name']}: {', '.join(loaded['noisy_channels'])} 가 쉴 때부터 노이즈가 큼")
        sessions.append(loaded)
    fixed = [s for s in sessions if s["group"] == "fist"]
    rand = [s for s in sessions if s["group"] == "random"]
    return sos, fixed, rand


def cross_validate(fixed, cfg, cfg2, n_ch, folds=5):
    """고정 순서 세션만으로 하는 교차검증 (Random 시험 세트를 보지 않고 설정을 고를 때 사용)"""
    n_cls = len(cfg["class_names"])
    preds = []
    rf = 0
    for fi in range(folds):
        test_idx = list(range(fi, len(fixed), folds))
        train_s = [s for i, s in enumerate(fixed) if i not in test_idx]
        test_s = [fixed[i] for i in test_idx]
        rp = rest_power(train_s)
        model, norm = train_model(prepare(train_s, cfg, cfg2, rp, cfg2["aug_force_copies"], seed=fi),
                                  n_cls, n_ch, cfg2, seed=fi)
        preds += [(d, predict_probs(model, norm, d["X"])) for d in prepare(test_s, cfg, cfg2, rp)]
        rf = model.rf
        print(f"  교차검증 {fi + 1}/{folds} 완료")
    return evaluate(preds, cfg, cfg2, rf)


def run(cfg, cfg2, do_cv=False, export=True):
    os.makedirs(cfg2["out_dir"], exist_ok=True)
    print("[1/5] 데이터 읽는 중...")
    sos, fixed, rand = load_sessions(cfg)
    names = cfg["class_names"]
    n_cls, n_ch = len(names), len(cfg["channels"])
    print(f"  동작 {n_cls}가지: {', '.join(names)} / 고정 순서 {len(fixed)}개, Random {len(rand)}개")
    if len(fixed) < 3 or not rand:
        raise SystemExit("데이터가 부족합니다. data/Fixed, data/Random 폴더를 확인하세요.")

    lines = []
    if do_cv:
        print("[2/5] 교차검증 중 (고정 순서 세션)...")
        cv = cross_validate(fixed, cfg, cfg2, n_ch)
        lines += report_lines(f"교차검증: 고정 순서 세션 {len(fixed)}개를 5묶음으로 나눠 번갈아 시험", cv, cfg, cfg2)
    else:
        print("[2/5] 교차검증 건너뜀 (--cv 를 붙이면 실행)")

    print(f"[3/5] 학습 중 (고정 순서 세션 {len(fixed)}개 전부 사용)...")
    rest_pow = rest_power(fixed)
    train = prepare(fixed, cfg, cfg2, rest_pow, cfg2["aug_force_copies"], seed=cfg2["seed"])
    model, norm = train_model(train, n_cls, n_ch, cfg2, seed=cfg2["seed"], verbose=True)
    cal_logamp, strength_full = class_tables(train, cfg, n_ch)

    print("[4/5] 평가 중 (학습에 쓰지 않은 Random 세션)...")
    test = prepare(rand, cfg, cfg2, rest_pow)
    ev = evaluate([(d, predict_probs(model, norm, d["X"])) for d in test], cfg, cfg2, model.rf)
    lines += report_lines("Random 세션(순서 섞은 데이터, 학습에 쓰지 않음) 시험 결과", ev, cfg, cfg2)

    if len(test) >= 2:
        lines.append("=== 전극을 다시 붙인 상황 흉내 (채널별 이득만 바꿈, 경계 구간까지 포함한 프레임 단위 정확도) ===")
        for lo, hi in ((0.6, 1.6), (0.4, 2.5)):
            a_base, a_none, a_cal = calibration_simulation(model, norm, test, cfg, cfg2, cal_logamp, n_ch, lo, hi)
            lines.append(f"이득 {lo}~{hi}배: 변화 없을 때 {a_base * 100:.1f}% -> 보정 없이 {a_none * 100:.1f}% / 사용자 보정 후 {a_cal * 100:.1f}%")
        lines += ["(이득 변화만 흉내 낸 것으로, 전극 위치가 달라진 경우는 포함하지 않음)", ""]

    ops, params = count_ops(model, n_cls)
    hop_ms = cfg2["hop"] * 1000 / cfg["fs"]
    lines += ["=== ESP32 부담 ===",
              f"가중치 수: {params}개 (float 기준 약 {params * 4 / 1024:.0f}KB)",
              f"한 번 판단({hop_ms:.0f}ms마다)에 곱셈 약 {ops / 1000:.1f}천 번 -> 1초에 약 {ops * 1000 / hop_ms / 1e6:.2f}백만 번",
              f"한 번 판단이 보는 범위: {model.rf}프레임 = {model.rf * hop_ms / 1000:.3f}초",
              "동작별 힘 세기 100% 기준(배): " + ", ".join(f"{n} {v:.1f}" for n, v in zip(names, strength_full))]
    report = "\n".join(lines)
    print("\n" + report + "\n")

    if export:
        print("[5/5] 결과 저장 + ESP32용 파일 내보내는 중...")
        out = cfg2["out_dir"]
        with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        base.plot_confusion(ev["cm"], names, f"Random test, v2 (acc {ev['win_acc'] * 100:.1f}%)",
                            os.path.join(out, "confusion_window.png"))
        base.plot_realtime(ev["results"], cfg, os.path.join(out, "realtime_random.png"))
        # 폴더 경로는 PC마다 다르므로 저장하지 않음 (host_check.py가 실행하는 PC의 경로를 다시 채움)
        no_paths = lambda d: {k: v for k, v in d.items() if not k.endswith(("_dir", "_dirs"))}
        torch.save({"state_dict": model.state_dict(), "cfg": no_paths(cfg), "cfg2": no_paths(cfg2), "norm": norm,
                    "sos": sos, "rest_pow": rest_pow}, os.path.join(out, "emg2_model.pt"))
        model_h = os.path.join(out, "emg2_model.h")
        vec_h = os.path.join(out, "emg2_test_vectors.h")
        export_model_header(model, cfg, cfg2, sos, norm, rest_pow, cal_logamp, strength_full, model_h)
        export_test_vectors(model, norm, rand, cfg, cfg2, sos, rest_pow, vec_h)
        for d in cfg2["sketch_dirs"]:
            if os.path.isdir(d):
                shutil.copy(model_h, os.path.join(d, "emg2_model.h"))
                shutil.copy(vec_h, os.path.join(d, "emg2_test_vectors.h"))
                print(f"  {os.path.basename(d)} 폴더에 .h 파일 복사 완료")
        print(f"\n완료! 결과는 {out} 폴더에 있어요.")
    return {"model": model, "norm": norm, "sos": sos, "rest_pow": rest_pow, "eval": ev, "test": test,
            "cal_logamp": cal_logamp, "strength_full": strength_full}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="고정 순서 세션으로 5겹 교차검증도 실행")
    ap.add_argument("--no-export", action="store_true", help="파일을 저장하지 않고 결과만 출력")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run(copy.deepcopy(base.CFG), copy.deepcopy(CFG2), do_cv=args.cv, export=not args.no_export)
