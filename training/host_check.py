"""
펌웨어의 판단 코드(firmware/emg_board_v2/emg2.cpp)가 PC(PyTorch) 계산과 같은지, 보드 없이 PC에서 확인합니다.

    python host_check.py

- emg2.cpp를 PC용으로 그대로 컴파일해서 Random 세션 전체(약 6.7분)를 흘려보내고,
  25ms마다의 확률 / 힘 세기 / 최종 출력을 train_v2.py의 계산과 하나하나 비교합니다.
- train_v2.py를 먼저 실행해 output_v2/emg2_model.pt 와 firmware/emg_board_v2/emg2_model.h 가 있어야 합니다.
- C++ 컴파일러가 필요합니다. g++ 또는 clang++가 없으면:  pip install ziglang
- Windows가 새로 만든 실행 파일을 차단하는 PC에서는 자동으로 WebAssembly + Node.js로 바꿔 실행합니다
  (이때는 ziglang과 Node.js가 필요).
  (보드에서 직접 확인하려면 이 스크립트 대신 emg_board_v2.ino를 MODE_SELFTEST로 올리면 됩니다)
"""

import os
import sys
import shutil
import subprocess

import numpy as np
import torch

import train_emg as base
import train_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
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


def run_host(exe, raw_i16, gain=None):
    inp = os.path.join(BUILD_DIR, "in.bin")
    out = os.path.join(BUILD_DIR, "out.bin")
    raw_i16.astype("<i2").tofile(inp)
    gains = [f"{g:.9g}" for g in gain] if gain is not None else []
    if exe.endswith(".wasm"):
        args = ["node", "--no-warnings", os.path.join(HERE, "host_check", "run_wasi.js"), exe, BUILD_DIR,
                "in.bin", "out.bin"] + gains
    else:
        args = [exe, inp, out] + gains
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return out


def build_runnable():
    """PC용 실행 파일을 만들어 보고, Windows 보안 정책이 실행을 막으면 WebAssembly + Node로 바꿈"""
    exe = build()
    try:
        run_host(exe, np.zeros((50, 4), np.int16))
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
    print("=> 통과: 펌웨어 C 코드와 PC 계산이 일치합니다." if ok else "=> 실패: 차이가 큽니다. emg2.cpp와 train_v2.py를 확인하세요.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
