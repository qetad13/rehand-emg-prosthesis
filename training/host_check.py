"""
펌웨어의 판단 코드(firmware/emg_board_v2/emg2.cpp)가 PC(PyTorch) 계산과 같은지, 보드 없이 PC에서 확인합니다.

    python host_check.py

- emg2.cpp를 PC용으로 그대로 컴파일해서 Random 세션 전체(약 6.7분)를 흘려보내고,
  25ms마다의 확률 / 힘 세기 / 최종 출력을 train_v2.py의 계산과 하나하나 비교합니다.
- 사용자 보정의 "마지막 층 다시 맞추기"(emg2_adapt_fc)도 train_v2.py의 adapt_fc와 비교합니다.
  (채널 순서를 바꿔 착용 위치가 달라진 상황을 흉내 내고, 다시 맞춘 층으로 세션을 흘려보내 확인)
- train_v2.py를 먼저 실행해 output_v2/emg2_model.pt 와 firmware/emg_board_v2/emg2_model.h 가 있어야 합니다.
- C++ 컴파일러가 필요합니다. g++ 또는 clang++가 없으면:  pip install ziglang
- Windows가 새로 만든 실행 파일을 차단하는 PC에서는 자동으로 WebAssembly + Node.js로 바꿔 실행합니다
  (이때는 ziglang과 Node.js가 필요).
  (보드에서 직접 확인하려면 이 스크립트 대신 emg_board_v2.ino를 MODE_SELFTEST로 올리면 됩니다)
"""

import os
import sys
import copy
import shutil
import subprocess

import numpy as np
import torch

import train_emg as base
import train_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
EMG_NUM_CH_GUESS = 4          # 실행 파일이 막혔는지 보려고 잠깐 돌릴 때만 씀
FW_DIR = os.path.join(HERE, "..", "firmware", "emg_board_v2")
BUILD_DIR = os.path.join(HERE, "output_v2", "host_build")


def find_compiler():
    if os.environ.get("CXX"):
        return os.environ["CXX"].split()
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return [name]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        return None


def build(wasm=False):
    """wasm=False: PC용 실행 파일 / wasm=True: WebAssembly (Node로 실행, zig 컴파일러일 때만 가능)"""
    cxx = find_compiler()
    if cxx is None:
        raise SystemExit("C++ 컴파일러를 찾지 못했습니다. g++/clang++를 설치하거나  pip install ziglang  을 실행하세요.")
    os.makedirs(BUILD_DIR, exist_ok=True)
    if wasm:
        if "ziglang" not in " ".join(cxx):
            raise SystemExit("WebAssembly로 빌드하려면 zig가 필요합니다:  pip install ziglang")
        if not shutil.which("node"):
            raise SystemExit("WebAssembly로 실행하려면 Node.js가 필요합니다.")
        exe = os.path.join(BUILD_DIR, "emg2_host.wasm")
        extra = ["-target", "wasm32-wasi", "-fno-exceptions"]
    else:
        exe = os.path.join(BUILD_DIR, "emg2_host.exe" if os.name == "nt" else "emg2_host")
        extra = []
    cmd = cxx + extra + ["-O2", "-I", FW_DIR, os.path.join(HERE, "host_check", "emg2_host.cpp"),
                         os.path.join(FW_DIR, "emg2.cpp"), "-o", exe]
    print("빌드:", os.path.basename(exe), "...")
    subprocess.run(cmd, check=True)
    return exe


def _exec(exe, args):
    """emg2_host 실행. args 중 '@이름'은 BUILD_DIR 안의 파일"""
    if exe.endswith(".wasm"):
        conv = ["/work/" + a[1:] if a.startswith("@") else a for a in args]
        cmd = ["node", "--no-warnings", os.path.join(HERE, "host_check", "run_wasi.js"), exe, BUILD_DIR] + conv
    else:
        cmd = [exe] + [os.path.join(BUILD_DIR, a[1:]) if a.startswith("@") else a for a in args]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def run_host(exe, raw_i16, gain=None, fc=None):
    """세션 전체를 흘려보냄. fc = (W, b)면 마지막 층을 그 값으로 바꿔서"""
    raw_i16.astype("<i2").tofile(os.path.join(BUILD_DIR, "in.bin"))
    gains = [f"{g:.9g}" for g in gain] if gain is not None else []
    fc_arg = "-"
    if fc is not None:
        np.concatenate([fc[0].ravel(), fc[1].ravel()]).astype("<f4").tofile(os.path.join(BUILD_DIR, "fc.bin"))
        fc_arg = "@fc.bin"
    _exec(exe, ["run", "@in.bin", "@out.bin", fc_arg] + gains)
    return os.path.join(BUILD_DIR, "out.bin")


def adapt_host(exe, H, Y, n_cls):
    """emg2_adapt_fc 실행 -> (W, b, 보정 녹화 정답률)"""
    H.astype("<f4").tofile(os.path.join(BUILD_DIR, "h.bin"))
    Y.astype(np.uint8).tofile(os.path.join(BUILD_DIR, "y.bin"))
    _exec(exe, ["adapt", "@h.bin", "@y.bin", "@adapt.bin"])
    r = np.fromfile(os.path.join(BUILD_DIR, "adapt.bin"), dtype="<f4")
    e = H.shape[1]
    return r[:n_cls * e].reshape(n_cls, e), r[n_cls * e:n_cls * e + n_cls], float(r[-1])


def build_runnable():
    """PC용 실행 파일을 만들어 보고, Windows 보안 정책이 실행을 막으면 WebAssembly + Node로 바꿈"""
    exe = build()
    try:
        run_host(exe, np.zeros((50, EMG_NUM_CH_GUESS), np.int16))
        return exe
    except OSError as e:
        print(f"  새로 만든 실행 파일을 이 PC가 실행하지 못하게 막았습니다 ({e}).")
        print("  보안 설정은 그대로 두고, 같은 코드를 WebAssembly로 빌드해 Node 안에서 실행합니다.")
        return build(wasm=True)


def main():
    ckpt_path = os.path.join(HERE, "output_v2", "emg2_model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("output_v2/emg2_model.pt 가 없습니다. 먼저 python train_v2.py 를 실행하세요.")
    ck = torch.load(ckpt_path, weights_only=False)      # 직접 만든 파일
    cfg, cfg2, norm, rest_pow = ck["cfg"], ck["cfg2"], ck["norm"], ck["rest_pow"]
    cfg.update({k: v for k, v in base.CFG.items() if k.endswith("_dir")})   # 데이터 폴더는 이 PC의 경로로
    names = cfg["class_names"]
    n_cls, n_ch = len(names), len(cfg["channels"])
    model = v2.build_model(len(norm[0]), n_cls, cfg2)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    exe = build_runnable()
    sos = base.design_sos(cfg)
    sessions = [s for s in (base.load_session(s, cfg, sos) for s in base.list_sessions(cfg))
                if s is not None and s["group"] == "random"]
    rec = np.dtype([("probs", "<f4", n_cls), ("logamp", "<f4", n_ch), ("strength", "<f4"), ("out", "<i4")])

    worst_p = worst_s = 0.0
    frames = pred_diff = out_diff = 0
    for case, gain in (("보정 없음", None), ("사용자 보정값 적용", np.array([0.3, -0.2, 0.1, -0.4][:n_ch], np.float32))):
        for s in sessions:
            data = v2.prepare([s], cfg, cfg2, rest_pow)[0]
            x = data["X"].copy()
            strength = data["strength"]
            if gain is not None:
                x[:, :n_ch] -= gain
                strength = v2.compute_strength(s[f"acc{cfg2['hop']}"], cfg2, rest_pow, gain)
            probs = v2.predict_probs(model, norm, x)
            voted = v2.decide_sequence(probs, strength, cfg, cfg2)

            got = np.fromfile(run_host(exe, np.round(s["raw"]).astype(np.int16), gain), dtype=rec)
            n = min(len(got), len(probs))
            dp = np.abs(got["probs"][:n] - probs[:n]).max()
            ds = (np.abs(got["strength"][:n] - strength[:n]) / (1.0 + strength[:n])).max()
            pd = int((got["probs"][:n].argmax(axis=1) != probs[:n].argmax(axis=1)).sum())
            od = int((got["out"][:n] != voted[:n]).sum())
            worst_p, worst_s = max(worst_p, dp), max(worst_s, ds)
            frames += n
            pred_diff += pd
            out_diff += od
            print(f"  {s['name']} ({case}): {n}프레임, 확률 차이 최대 {dp:.2e}, 힘 세기 차이 최대 {ds * 100:.3f}%, "
                  f"판단 다름 {pd}, 최종 출력 다름 {od}")

    ok = worst_p < 5e-3 and worst_s < 5e-3 and out_diff == 0
    print(f"\n총 {frames}프레임: 확률 차이 최대 {worst_p:.2e}, 판단(확률 최대 클래스) 다름 {pred_diff}, 최종 출력 다름 {out_diff}")

    # ---- 사용자 보정: 마지막 층 다시 맞추기 ----
    # 채널 순서를 한 칸씩 돌려 '착용 위치가 달라진' 상황을 만들고, 첫 Random 세션으로 보정, 나머지로 시험
    print("\n[사용자 보정: 마지막 층 다시 맞추기] 채널 순서를 바꿔 착용 위치가 달라진 상황 흉내")
    rot = []
    for s in sessions:
        r = dict(s)
        r["raw"] = np.roll(s["raw"], 1, axis=1)
        r["filt"] = np.roll(s["filt"], 1, axis=1)
        r.pop(f"acc{cfg2['hop']}", None)
        rot.append(r)
    cal = v2.prepare([rot[0]], cfg, cfg2, rest_pow)[0]
    sel = v2.cal_frames(cal, cfg, cfg2)
    H = v2.embed_sequence(model, norm, cal["X"])[sel]
    Y = cal["y"][sel]
    w0, b0 = model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy()
    w_py, b_py, acc_py = v2.adapt_fc(w0, b0, H, Y, n_cls, cfg2)
    w_c, b_c, acc_c = adapt_host(exe, H, Y, n_cls)
    dw = max(np.abs(w_c - w_py).max(), np.abs(b_c - b_py).max()) / max(np.abs(w_py).max(), 1e-6)
    same_pred = float(((H @ w_c.T + b_c).argmax(1) == (H @ w_py.T + b_py).argmax(1)).mean())
    base_acc = float((v2.predict_probs(model, norm, cal["X"]).argmax(1)[sel] == Y).mean())
    print(f"  보정 녹화 {len(Y)}프레임: 원래 모델 정답 {base_acc * 100:.0f}% -> 다시 맞춘 뒤 C {acc_c * 100:.1f}% / PC {acc_py * 100:.1f}%")
    print(f"  C와 PC의 다시 맞춘 가중치 차이 최대 {dw * 100:.3f}% (가중치 크기 대비), 보정 녹화 판단 일치 {same_pred * 100:.1f}%")
    adapted = copy.deepcopy(model)
    with torch.no_grad():
        adapted.fc.weight.copy_(torch.from_numpy(w_c))
        adapted.fc.bias.copy_(torch.from_numpy(b_c))
    a_worst, a_out, a_frames = 0.0, 0, 0
    before_ok = after_ok = n_eval = 0
    for s in rot[1:]:
        d = v2.prepare([s], cfg, cfg2, rest_pow)[0]
        probs = v2.predict_probs(adapted, norm, d["X"])
        voted = v2.decide_sequence(probs, d["strength"], cfg, cfg2)
        got = np.fromfile(run_host(exe, np.round(s["raw"]).astype(np.int16), None, (w_c, b_c)), dtype=rec)
        n = min(len(got), len(probs))
        a_worst = max(a_worst, float(np.abs(got["probs"][:n] - probs[:n]).max()))
        a_out += int((got["out"][:n] != voted[:n]).sum())
        a_frames += n
        m = d["m"]
        before_ok += int((v2.predict_probs(model, norm, d["X"]).argmax(1)[m] == d["y"][m]).sum())
        after_ok += int((probs.argmax(1)[m] == d["y"][m]).sum())
        n_eval += int(m.sum())
    print(f"  다시 맞춘 층으로 {len(rot) - 1}세션 {a_frames}프레임: 확률 차이 최대 {a_worst:.2e}, 최종 출력 다름 {a_out}")
    print(f"  (참고) 이 흉내에서 프레임 정확도: 보정 전 {before_ok / max(n_eval, 1) * 100:.1f}% -> 다시 맞춘 뒤 {after_ok / max(n_eval, 1) * 100:.1f}%")
    ok_adapt = dw < 1e-2 and same_pred > 0.99 and abs(acc_c - acc_py) < 0.01 and a_worst < 5e-3 and a_out == 0

    print("=> 통과: 펌웨어 C 코드와 PC 계산이 일치합니다." if ok and ok_adapt
          else "=> 실패: 차이가 큽니다. emg2.cpp와 train_v2.py를 확인하세요.")
    return 0 if ok and ok_adapt else 1


if __name__ == "__main__":
    sys.exit(main())
