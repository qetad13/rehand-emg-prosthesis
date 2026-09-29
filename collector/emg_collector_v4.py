import sys
import random
import serial.tools.list_ports
import csv
import os
import time
import serial

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
)

# ------------------------------------------------------------
# NUM_CHANNELS   : 아두이노가 실제로 보내는 데이터 값 개수 (Ch00~Ch03, 4개 그대로 전송)
# TOTAL_CHANNELS : 엑셀 메타데이터에 적을 "실사용 채널 수" (지금은 3개만 사용)
# ------------------------------------------------------------
NUM_CHANNELS = 4
TOTAL_CHANNELS = 4
USING_CHANNELS = 4

# ------------------------------------------------------------
# CYCLE_SECONDS : 힘주기/힘풀기 반복 주기 (초). 5초마다 전환됨.
# ------------------------------------------------------------
CYCLE_SECONDS = 5

# ------------------------------------------------------------
# 녹화 순서 (첫 번째 힘주기 구간은 항상 휴식)
#   고정 순서(fist) : 휴식 -> 엄지 -> 검지 -> 중지 -> 약지 -> 새끼
#   Random         : 휴식 -> 다섯 손가락을 무작위로 섞은 순서 (측정 시작 때 자동 생성)
# ------------------------------------------------------------
# 동작 7가지 + 휴식 = 8구간 (5초 힘주기 + 5초 쉬기) x 8 = 80초
GESTURES = ["Thumb", "Index", "Middle", "Ring", "Little", "Fist", "Pinch"]
FIXED_ORDER = ["Rest"] + GESTURES
KOREAN = {"Rest": "휴식", "Thumb": "엄지", "Index": "검지", "Middle": "중지",
          "Ring": "약지", "Little": "새끼", "Fist": "주먹", "Pinch": "집게"}
# 녹화 중 화면에 보여줄 안내 (동작을 매번 똑같이 하도록)
GUIDE = {"Fist": "주먹 (엄지로 감싸고 꽉 쥐기)",
         "Pinch": "집게 (엄지·검지 끝을 맞대고 누르기, 나머지는 힘 빼기)"}


class EMGCollector(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EMG Data Collector")
        self.resize(500, 560)

        self.saveFolder = ""

        self.initUI()

        self.serial = None

        self.startButton.clicked.connect(self.startMeasure)

    def initUI(self):

        layout = QVBoxLayout()

        # ----------------------------
        # COM Port
        # ----------------------------
        comLayout = QHBoxLayout()

        comLabel = QLabel("COM Port")

        self.comBox = QComboBox()

        self.refreshButton = QPushButton("새로고침")
        self.refreshButton.clicked.connect(self.refreshPorts)

        comLayout.addWidget(comLabel)
        comLayout.addWidget(self.comBox)
        comLayout.addWidget(self.refreshButton)

        layout.addLayout(comLayout)

        # ----------------------------
        # Sampling Time
        # ----------------------------
        sampleLayout = QHBoxLayout()

        sampleLabel = QLabel("측정 시간 (초)")

        self.timeSpin = QSpinBox()
        self.timeSpin.setRange(1, 300)
        self.timeSpin.setValue(len(FIXED_ORDER) * 2 * CYCLE_SECONDS)  # 8구간 x 10초 = 80초

        sampleLayout.addWidget(sampleLabel)
        sampleLayout.addWidget(self.timeSpin)

        layout.addLayout(sampleLayout)

        # ----------------------------
        # Output rate (아두이노 printDivider 와 맞춰야 하는 값)
        # SamplingRate = 아두이노 sampleRate / printDivider
        # ----------------------------
        rateLayout = QHBoxLayout()

        rateLabel = QLabel("출력 샘플레이트 (Hz)")

        self.rateSpin = QSpinBox()
        self.rateSpin.setRange(1, 20000)
        self.rateSpin.setValue(1000)  # 아두이노 printDivider=1 -> 실제 1000Hz 출력과 일치

        rateLayout.addWidget(rateLabel)
        rateLayout.addWidget(self.rateSpin)

        layout.addLayout(rateLayout)

        # ----------------------------
        # Save Folder
        # ----------------------------
        folderLayout = QHBoxLayout()

        self.folderLabel = QLabel("저장 폴더를 선택하세요")

        self.folderButton = QPushButton("폴더 선택")
        self.folderButton.clicked.connect(self.selectFolder)

        folderLayout.addWidget(self.folderLabel)
        folderLayout.addWidget(self.folderButton)

        layout.addLayout(folderLayout)

        # ----------------------------
        # Finger / Gesture Selection (8종)
        # ----------------------------
        layout.addWidget(QLabel("동작 선택"))

        self.group = QButtonGroup()

        self.rest = QRadioButton("Rest")
        self.fist = QRadioButton("고정 순서 (fixed: " + "-".join(KOREAN[g] for g in FIXED_ORDER) + ")")
        self.random_ = QRadioButton("Random (순서 자동으로 섞음)")

        self.fist.setChecked(True)

        for btn in (self.rest, self.fist, self.random_):
            self.group.addButton(btn)
            layout.addWidget(btn)

        # 이번 녹화의 동작 순서 표시
        self.orderLabel = QLabel("순서 : -")
        layout.addWidget(self.orderLabel)

        self.startButton = QPushButton("측정 시작")
        layout.addWidget(self.startButton)

        self.statusLabel = QLabel("상태 : 대기")
        # 안내 문구가 잘 보이도록 폰트 크게
        font = self.statusLabel.font()
        font.setPointSize(16)
        font.setBold(True)
        self.statusLabel.setFont(font)
        self.statusLabel.setWordWrap(True)   # 긴 안내 문구가 잘리지 않게 줄바꿈
        layout.addWidget(self.statusLabel)

        self.setLayout(layout)

        self.refreshPorts()

    def refreshPorts(self):

        self.comBox.clear()

        ports = serial.tools.list_ports.comports()

        for port in ports:
            self.comBox.addItem(port.device)

    def selectFolder(self):

        folder = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")

        if folder:
            self.saveFolder = folder
            self.folderLabel.setText(folder)

    def getFingerName(self):

        if self.fist.isChecked():
            return "Fixed"
        if self.random_.isChecked():
            return "Random"

        return "Rest"

    def getNextFileName(self):

        finger = self.getFingerName()

        folder = os.path.join(self.saveFolder, finger)

        os.makedirs(folder, exist_ok=True)

        number = 1

        while True:

            filename = f"{finger.lower()}{number:02d}.csv"

            path = os.path.join(folder, filename)

            if not os.path.exists(path):
                return path

            number += 1

    def getPhase(self, elapsed, order):
        """
        elapsed(초) 기준으로 현재가 '힘주기' 구간인지 '힘풀기' 구간인지 판단.
        0~5초: 힘주기, 5~10초: 힘풀기, 10~15초: 힘주기 ... 반복
        힘주기 구간에서는 order에 따라 이번에 힘줄 손가락을 알려줌
        """
        slot = int(elapsed // CYCLE_SECONDS)
        cycleIndex = slot % 2
        remaining = CYCLE_SECONDS - (elapsed % CYCLE_SECONDS)

        if cycleIndex == 0:
            k = slot // 2
            target = order[k] if k < len(order) else "Rest"
            if target == "Rest":
                return "Contract", "🖐 휴식 (그대로 쉬세요)", remaining
            return "Contract", f"💪 {GUIDE.get(target, KOREAN[target])} 힘주세요!", remaining
        else:
            return "Relax", "🖐 힘 빼세요", remaining

    def startMeasure(self):

        if self.saveFolder == "":
            self.statusLabel.setText("폴더를 먼저 선택하세요.")
            return

        port = self.comBox.currentText()

        try:
            self.serial = serial.Serial(port, 460800, timeout=1)
            time.sleep(2)

        except Exception:

            self.statusLabel.setText("COM 연결 실패")
            return

        # 이번 녹화 순서 정하기
        mode = self.getFingerName()
        if mode == "Random":
            order = ["Rest"] + random.sample(GESTURES, len(GESTURES))
        elif mode == "Fixed":
            order = list(FIXED_ORDER)
        else:
            order = ["Rest"] * 99

        # 측정 시간이 순서를 다 담기에 짧으면 자동으로 늘림
        need = len(order) * 2 * CYCLE_SECONDS
        if mode != "Rest" and self.timeSpin.value() < need:
            self.timeSpin.setValue(need)
        if mode != "Rest":
            self.orderLabel.setText("순서 : " + " → ".join(KOREAN[g] for g in order))
        else:
            self.orderLabel.setText("순서 : 계속 휴식")

        self.statusLabel.setText("3초 후 시작")
        QApplication.processEvents()

        for i in range(3, 0, -1):
            self.statusLabel.setText(f"{i}...")
            QApplication.processEvents()
            time.sleep(1)

        # 카운트다운 동안 쌓인 묵은 데이터를 여기서 비움
        # (여기서 비워야 실제 측정이 진짜 이 순간부터 시작됨)
        self.serial.reset_input_buffer()

        filename = self.getNextFileName()
        duration = self.timeSpin.value()
        outputRate = self.rateSpin.value()  # Hz
        dt = 1.0 / outputRate

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            # ------------------------------------------------------------
            # 상단 메타데이터 3줄
            #   SamplePeriod  : 실제 측정 시간 (예: "80s")
            #   SamplingRate  : 초당 출력 샘플 수 (Hz)
            #   TotalChannels : 실사용 채널 수 (3)
            # ------------------------------------------------------------
            writer.writerow(["SamplePeriod", f"{duration}s"])
            writer.writerow(["SamplingRate", outputRate])
            writer.writerow(["UsingChannels", USING_CHANNELS])

            # 컬럼 헤더 (Phase 컬럼 추가: 해당 시점이 힘주기/힘풀기 구간인지 라벨링)
            header = ["Time", "Phase"] + [f"Ch {i:02d}" for i in range(NUM_CHANNELS)]
            writer.writerow(header)

            start = time.time()
            rowIndex = 0
            lastPhase = None

            while True:

                elapsed = time.time() - start

                if elapsed >= duration:
                    break

                phase, phaseLabel, remaining = self.getPhase(elapsed, order)

                # 구간이 바뀔 때마다, 혹은 주기적으로 화면을 갱신
                if phase != lastPhase:
                    lastPhase = phase
                    self.statusLabel.setText(
                        f"{phaseLabel}  ({int(elapsed)}s / {duration}s)"
                    )
                    QApplication.processEvents()
                else:
                    # 같은 구간이어도 남은 시간 카운트다운을 보여주기 위해 갱신
                    self.statusLabel.setText(
                        f"{phaseLabel}  (남은 {remaining:.0f}s)  [{int(elapsed)}s / {duration}s]"
                    )
                    QApplication.processEvents()

                line = self.serial.readline().decode(errors="ignore").strip()

                if line == "":
                    continue

                parts = line.split(",")

                if len(parts) != NUM_CHANNELS:
                    # 아두이노 쪽 출력 형식이 안 맞으면 이 줄은 건너뜀
                    continue

                print(line)

                t = round(rowIndex * dt, 6)

                row = [t, phase] + parts
                writer.writerow(row)

                rowIndex += 1

        self.serial.close()

        # 순서를 같은 이름의 _order.txt 파일로 저장 (학습 코드가 자동으로 읽음)
        if mode in ("Random", "Fixed"):
            orderPath = os.path.splitext(filename)[0] + "_order.txt"
            with open(orderPath, "w", encoding="utf-8") as f:
                f.write(",".join(order) + "\n")

        self.statusLabel.setText(f"저장 완료 : {os.path.basename(filename)}")


app = QApplication(sys.argv)

window = EMGCollector()
window.show()

sys.exit(app.exec())