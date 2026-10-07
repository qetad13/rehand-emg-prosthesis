// =====================================================================
// EMG 녹화 + 실시간 손가락 동작 판단 (XIAO ESP32-S3)
//
// 녹화(MODE_RECORD)와 판단(MODE_RUN)이 같은 readEmgInput()을 쓰기 때문에
// 학습 데이터와 실시간 입력이 항상 똑같습니다. 필터는 AI 쪽(emg_infer)에서 겁니다.
//
//   MODE_RECORD : GUI로 녹화할 때 (예전 녹화 코드 대신 이걸 올리면 됨)
//   MODE_CHECK  : 센서 상태 확인 (채널별 평균/흔들림/클리핑)
//   MODE_RUN    : 0.5초마다 판단 (새 데이터로 학습한 모델이 들어간 뒤에 사용)
//   MODE_REPLAY : 센서 없이 저장된 데이터로 판단 흐름 확인
//
// MODE_RUN에서는 판단 결과(어떤 손가락)와 함께 힘 세기(1~100)도 의수로 보냅니다.
// (아래 "힘 세기" 참고. 세게 주면 큰 값 -> 의수 보드가 모터 속도로 사용)
//
// MODE_RUN 안에서는 스위치로 두 가지 동작 모드를 오갑니다. (아래 "모드 전환" 참고)
//   근전도 모드 : 근전도 판단 결과를 의수로 보냄 (기본)
//   매크로 모드 : 근전도를 쓰지 않고, 스위치로 고른 동작(명령 11~19)을 의수로 보냄
// =====================================================================
#include "emg_infer.h"
#include "emg_test_vectors.h"
#include "EMGFilters.h"   // 채널마다 필터 상태를 따로 갖도록 수정된 버전
#include <WiFi.h>
#include <esp_now.h>

// ---------------------------------------------------------------------
// 사용자 설정
// ---------------------------------------------------------------------
#define MODE_REPLAY 0
#define MODE_CHECK  1
#define MODE_RUN    2
#define MODE_RECORD 3
#define RUN_MODE    MODE_RUN

// 녹화는 항상 4채널 (CSV의 Ch 00, Ch 01, Ch 02, Ch 03 순서)
// 판단에는 그중 모델이 쓰는 채널(emg_model.h의 EMG_CHANNEL_IDX)만 사용
#define REC_CH 4
static const int EMG_PINS[REC_CH] = {A0, A1, A2, A3};

// 녹화(MODE_RECORD)와 판단(MODE_RUN)이 아래 두 함수를 똑같이 사용합니다.
// 여기를 바꾸면 반드시 데이터를 새로 녹화하고 새로 학습해야 합니다.
static EMGFilters s_emgFilter[REC_CH];

static void setupEmgInput() {
  analogReadResolution(12);
  for (int c = 0; c < REC_CH; ++c) {
    pinMode(EMG_PINS[c], INPUT);
    s_emgFilter[c].init(SAMPLE_FREQ_1000HZ, NOTCH_FREQ_60HZ, true, true, true);
  }
}

static int s_lastRaw[REC_CH];   // 필터 전 원본 값 (상태 확인용)

[[maybe_unused]] static int readEmgInput(int ch) {
  const int raw = analogRead(EMG_PINS[ch]);
  s_lastRaw[ch] = raw;
  return s_emgFilter[ch].update(raw);
}

// ---------------------------------------------------------------------
// 의수 보드로 무선 전송 (ESP-NOW, MODE_RUN에서만 동작)
// ---------------------------------------------------------------------
#define USE_ESPNOW 1

// 보내는 내용: 2바이트 {명령 번호, 힘 세기}
//   힘 세기 1~100 : 근전도 모드에서 손가락 동작 중일 때 (클수록 세게 -> 의수는 빠르게)
//   힘 세기 0     : 세기 정보 없음 (휴식, 매크로 모드) -> 의수는 기본 속도로
// 의수 보드가 아직 1바이트만 받는 코드라면 0으로 바꾸면 예전처럼 명령 번호만 보냅니다.
#define SEND_STRENGTH 1

// 판단 결과 -> 의수 보드 명령 번호 (팀에서 정한 번호, 이름으로 연결해서 동작 순서가 바뀌어도 안전)
//   1 엄지, 2 검지, 3 중지, 4 약지, 5 소지, 6 휴식, 7 주먹, 8 집게
static uint8_t handCmdFor(int gesture) {
  const char *n = EMG_CLASS_NAMES[gesture];
  if (!strcmp(n, "Thumb"))  return 1;
  if (!strcmp(n, "Index"))  return 2;
  if (!strcmp(n, "Middle")) return 3;
  if (!strcmp(n, "Ring"))   return 4;
  if (!strcmp(n, "Little")) return 5;
  if (!strcmp(n, "Rest"))   return 6;
  if (!strcmp(n, "Fist"))   return 7;
  if (!strcmp(n, "Pinch"))  return 8;
  return 6;  // 모르는 동작이면 휴식
}

// 의수 보드 MAC 주소. 모르면 FF로 두면 근처 모든 ESP-NOW 수신기로 방송(broadcast)
static uint8_t HAND_MAC[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

[[maybe_unused]] static bool s_espnowReady = false;

[[maybe_unused]] static void setupEspNow() {
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] 초기화 실패");
    return;
  }
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, HAND_MAC, 6);
  peer.channel = 0;          // 현재 채널 사용 (두 보드 모두 WiFi 미연결이면 같은 채널)
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("[ESP-NOW] 수신 보드 등록 실패");
    return;
  }
  s_espnowReady = true;
  Serial.print("[ESP-NOW] 준비 완료, 이 보드 MAC: ");
  Serial.println(WiFi.macAddress());
}

[[maybe_unused]] static void sendHandCommand(uint8_t cmd, uint8_t strength) {
  if (!s_espnowReady) return;
  const uint8_t pkt[2] = {cmd, strength};
  esp_err_t r = esp_now_send(HAND_MAC, pkt, SEND_STRENGTH ? 2 : 1);
  if (r != ESP_OK) Serial.printf("[ESP-NOW] 전송 실패 (%d)\n", (int)r);
}

// 판단 결과가 바뀌었을 때 호출됨
[[maybe_unused]] static void onOutputChanged(int gesture) {
  Serial.printf(">>> 출력 변경: %s (의수 명령 %d)\n", EMG_CLASS_NAMES[gesture],
                handCmdFor(gesture));
}

// ---------------------------------------------------------------------
// 힘 세기 (세게 주면 빠르게, 약하게 주면 천천히)
//
// "어떤 손가락인지"는 AI가 판단하고, "얼마나 세게 주는지"는 신호 크기로 따로 잽니다.
// (AI는 창마다 신호 크기를 똑같이 맞춘 뒤에 모양만 보기 때문에 세기를 알 수 없음)
//
//   세기(배) = 최근 0.2초 신호 크기가 쉴 때의 몇 배인지 (4채널 중 가장 큰 값)
//   세기(%)  = 휴식 게이트 크기를 0%, 아래 동작별 기준을 100%로 놓고 바꾼 값
//
// 같은 힘이라도 동작마다 신호 크기가 많이 달라서(약지 약 40배, 엄지 약 6배) 기준을 동작별로 둡니다.
// ---------------------------------------------------------------------
#define STRENGTH_WIN_MS  200    // 세기를 재는 구간 (최근 몇 ms)
#define STRENGTH_SMOOTH  0.5f   // 0~1. 작을수록 값이 부드럽지만 반응이 느림 (1이면 그대로)
#define STRENGTH_LEVELS  0      // 0 = 1~100 연속값 / 2 = 약,강 (50,100) / 3 = 약,중,강 (33,66,100)
#define STRENGTH_HYST    8.0f   // 단계로 나눌 때, 경계를 이만큼(%) 더 넘어야 단계가 바뀜 (깜빡임 방지)
#define STRENGTH_REPORT  1      // 동작이 끝날 때마다 세기 요약을 출력 (아래 기준 값을 맞출 때 사용)

// 동작별 100% 기준 = 평소 녹화하던 힘으로 줬을 때, 인식 직후 0.5초 동안의 평균 세기(배)
// (지금 data 폴더의 녹화 20개에서 구한 중앙값)
// - 센서를 다시 붙였거나 새로 녹화했으면: 평소 힘으로 몇 번 해보고 시리얼의
//   ">>> 세기 요약 ... 처음 0.5초 평균" 값으로 바꾸세요.
// - 평소 힘이 100%가 아니라 중간쯤이 되게 하려면 숫자를 키우면 됩니다 (예: 1.5배).
[[maybe_unused]] static float strengthFullFor(int gesture) {
  const char *n = EMG_CLASS_NAMES[gesture];
  if (!strcmp(n, "Thumb"))  return 6.4f;
  if (!strcmp(n, "Index"))  return 9.2f;
  if (!strcmp(n, "Middle")) return 23.0f;
  if (!strcmp(n, "Ring"))   return 40.0f;
  if (!strcmp(n, "Little")) return 7.8f;
  if (!strcmp(n, "Fist"))   return 21.0f;
  if (!strcmp(n, "Pinch"))  return 5.7f;
  return 10.0f;
}

// 세기(배) -> 0~100%
[[maybe_unused]] static float strengthPercent(int gesture, float ratio) {
  const float lo = (EMG_REST_GATE > 1.0f) ? EMG_REST_GATE : 1.5f;   // 이 크기 아래는 어차피 휴식
  float hi = strengthFullFor(gesture);
  if (hi < lo + 0.5f) hi = lo + 0.5f;
  const float p = (ratio - lo) / (hi - lo) * 100.0f;
  return p < 0.0f ? 0.0f : (p > 100.0f ? 100.0f : p);
}

// 세기(%) -> 의수로 보낼 값 (1~100). level에는 현재 단계를 기억해 둠 (0 = 아직 없음)
[[maybe_unused]] static uint8_t strengthToByte(float pct, uint8_t *level) {
#if STRENGTH_LEVELS >= 2
  const float w = 100.0f / STRENGTH_LEVELS;          // 한 단계의 폭 (%)
  int lv = (int)(pct / w) + 1;
  if (lv > STRENGTH_LEVELS) lv = STRENGTH_LEVELS;
  const int prev = *level;
  if (prev >= 1) {
    if (lv > prev && pct < prev * w + STRENGTH_HYST) lv = prev;          // 살짝 넘은 정도면 유지
    if (lv < prev && pct > (prev - 1) * w - STRENGTH_HYST) lv = prev;
  }
  *level = (uint8_t)lv;
  return (uint8_t)(lv * 100 / STRENGTH_LEVELS);
#else
  (void)level;
  const int v = (int)(pct + 0.5f);
  return (uint8_t)(v < 1 ? 1 : (v > 100 ? 100 : v));
#endif
}

// ---------------------------------------------------------------------
// 모드 전환 (MODE_RUN에서만 동작)
//   SW1 : 근전도 모드 <-> 매크로 모드
//   SW2 : 매크로 모드에서 다음 동작으로 넘김
//         (근전도 모드에서는 나중에 '사용자 보정'을 시작할 자리, 지금은 안내만 출력)
//
// 스위치가 없어도 아래 두 가지로 대신 누를 수 있습니다.
//   - 시리얼 모니터에서 1 전송 = SW1, 2 전송 = SW2
//   - 보드의 BOOT 버튼: 짧게 = SW2, 길게(0.8초) = SW1
// 스위치 핀: SW1 = D7(8번 패드), SW2 = D8(9번 패드). 서로 반대로 달렸으면 두 번호만 바꾸면 됩니다.
// (D7은 UART 수신 핀이기도 해서, Arduino IDE의 USB CDC On Boot가 Enabled여야 합니다)
// ---------------------------------------------------------------------
#define SW1_PIN         D7    // PCB 8번 패드. 스위치가 없는 보드면 -1
#define SW2_PIN         D8    // PCB 9번 패드. 스위치가 없는 보드면 -1
#define SW_PRESSED      LOW   // 누르면 GND에 연결되는 회로 기준 (내부 풀업 사용).
                              // 누르면 3.3V에 연결되는 회로면 HIGH로 변경
#define USE_SERIAL_KEYS 1     // 시리얼 1, 2 키로 스위치 흉내
#define USE_BOOT_BTN    1     // BOOT 버튼으로 스위치 흉내 (PCB가 오면 0으로 꺼도 됨)
#define BOOT_BTN_PIN    0     // XIAO ESP32-S3의 BOOT 버튼 (누르면 LOW)
#define LONG_PRESS_MS   800   // BOOT 버튼을 이 시간 이상 누르면 SW1로 처리

// 매크로 모드에서 SW2를 누를 때마다 이 순서로 넘어갑니다. {이름, 의수 명령 번호}
//   근전도 모드가 보내는 번호: 1~8   (엄지, 검지, 중지, 약지, 소지, 휴식, 주먹, 집게)
//   매크로 모드가 보내는 번호: 11~19 (의수 보드에서 번호별 동작을 정함)
// 매크로 모드에 들어가면 바로 첫 줄(11)을 보내고, 마지막(19) 다음은 다시 11입니다.
// 이름은 시리얼 출력용이라 동작이 정해지면 자유롭게 바꿔도 됩니다.
struct MacroItem { const char *name; uint8_t cmd; };
static const MacroItem MACROS[] = {
  {"매크로 1", 11},
  {"매크로 2", 12},
  {"매크로 3", 13},
  {"매크로 4", 14},
  {"매크로 5", 15},
  {"매크로 6", 16},
  {"매크로 7", 17},
  {"매크로 8", 18},
  {"매크로 9", 19},
};
#define MACRO_COUNT ((int)(sizeof(MACROS) / sizeof(MACROS[0])))

#define CTRL_EMG   0   // 근전도 모드
#define CTRL_MACRO 1   // 매크로 모드

#if RUN_MODE == MODE_RUN
static volatile uint8_t s_ctrlMode = CTRL_EMG;
static volatile uint8_t s_macroIdx = 0;

// SW1이 눌렸을 때: 모드 전환. LED가 켜져 있으면 매크로 모드
static void onSw1Pressed() {
  if (s_ctrlMode == CTRL_EMG) {
    s_macroIdx = 0;
    s_ctrlMode = CTRL_MACRO;
    Serial.printf("=== 매크로 모드 === 동작 1/%d: %s (의수 명령 %d)\n", MACRO_COUNT,
                  MACROS[0].name, MACROS[0].cmd);
  } else {
    s_ctrlMode = CTRL_EMG;
    Serial.println("=== 근전도 모드 === (휴식에서 다시 시작)");
  }
  digitalWrite(LED_BUILTIN, s_ctrlMode == CTRL_MACRO ? LOW : HIGH);
}

// SW2가 눌렸을 때
static void onSw2Pressed() {
  if (s_ctrlMode == CTRL_MACRO) {
    const uint8_t i = (uint8_t)((s_macroIdx + 1) % MACRO_COUNT);
    s_macroIdx = i;
    Serial.printf(">>> 매크로 동작 %d/%d: %s (의수 명령 %d)\n", i + 1, MACRO_COUNT,
                  MACROS[i].name, MACROS[i].cmd);
  } else {
    // TODO: 사용자 보정(마지막 층 다시 학습) 시작. 아직 구현 전
    Serial.println("[SW2] 근전도 모드: 사용자 보정은 아직 구현 전입니다");
  }
}

// 버튼 하나의 상태 (10ms마다 확인, 3번 연속 같은 값이어야 인정 = 떨림 제거)
#define BTN_NONE  0
#define BTN_SHORT 1
#define BTN_LONG  2
struct Button {
  int      pin;            // -1이면 없는 버튼
  int      pressedLevel;   // 눌렸을 때 핀 값 (LOW 또는 HIGH)
  bool     down;
  uint8_t  same;
  uint32_t downMs;
  bool     longDone;

  void begin(int p, int level) {
    pin = p; pressedLevel = level;
    down = false; same = 0; downMs = 0; longDone = false;
    if (pin >= 0) pinMode(pin, level == LOW ? INPUT_PULLUP : INPUT_PULLDOWN);
  }

  // useLong = false : 누르는 순간 BTN_SHORT
  // useLong = true  : 짧게 눌렀다 떼면 BTN_SHORT, LONG_PRESS_MS 이상 누르고 있으면 BTN_LONG
  int poll(bool useLong) {
    if (pin < 0) return BTN_NONE;
    const bool raw = (digitalRead(pin) == pressedLevel);
    if (raw == down) {
      same = 0;
    } else if (++same >= 3) {
      same = 0;
      down = raw;
      if (raw) {                                 // 눌림
        downMs = millis();
        longDone = false;
        if (!useLong) return BTN_SHORT;
      } else if (useLong && !longDone) {         // 짧게 눌렀다 뗌
        return BTN_SHORT;
      }
    }
    if (useLong && down && !longDone && millis() - downMs >= LONG_PRESS_MS) {
      longDone = true;
      return BTN_LONG;
    }
    return BTN_NONE;
  }
};

// 스위치 확인 전용 태스크 (코어 0). 1kHz 측정 루프(코어 1)는 건드리지 않음
static void ctrlTaskFn(void *arg) {
  (void)arg;
  Button sw1, sw2, boot;
  sw1.begin(SW1_PIN, SW_PRESSED);
  sw2.begin(SW2_PIN, SW_PRESSED);
  boot.begin(USE_BOOT_BTN ? BOOT_BTN_PIN : -1, LOW);

  for (;;) {
#if USE_SERIAL_KEYS
    while (Serial.available() > 0) {
      const int ch = Serial.read();
      if (ch == '1') onSw1Pressed();
      else if (ch == '2') onSw2Pressed();   // 줄바꿈 등 다른 글자는 무시
    }
#endif
    if (sw1.poll(false) == BTN_SHORT) onSw1Pressed();
    if (sw2.poll(false) == BTN_SHORT) onSw2Pressed();
    const int ev = boot.poll(true);
    if (ev == BTN_LONG) onSw1Pressed();
    else if (ev == BTN_SHORT) onSw2Pressed();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}
#endif  // RUN_MODE == MODE_RUN

// ---------------------------------------------------------------------
// 내부 동작 (보통 수정할 필요 없음)
// ---------------------------------------------------------------------
static const uint32_t SAMPLE_US = 1000000UL / EMG_FS;
static const uint32_t FIRST_DECISION = 500 + EMG_EPOCH_LEN;  // 필터 안정화 0.5초 + 1초

static EmgFilter s_filters[EMG_NUM_CH];
static uint32_t  s_count = 0;
static uint32_t  s_nextUs = 0;
static uint32_t  s_overruns = 0;

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
static float             s_ring[EMG_NUM_CH][EMG_EPOCH_LEN];
static uint32_t          s_ringPos = 0;
static float             s_epoch[EMG_NUM_CH * EMG_EPOCH_LEN];
static volatile bool     s_busy = false;
static volatile uint32_t s_epochEnd = 0;
static volatile uint32_t s_skipped = 0;   // 이전 판단이 안 끝나서 건너뛴 횟수 (0이어야 정상)
static TaskHandle_t      s_inferTask = NULL;
#endif

#if RUN_MODE == MODE_RECORD
// 측정과 전송을 분리: 측정은 코어 1에서 1kHz로, 전송은 코어 0에서
// (USB 전송이 잠깐 막혀도 샘플을 잃지 않게 하기 위함)
#define REC_BUF_SAMPLES 16384                 // 버퍼 크기
#define REC_MAX_BACKLOG 1000                  // 1초 넘게 밀리면 PC가 안 읽는 중으로 보고 최신 데이터부터 보냄
static int16_t      s_recBuf[REC_BUF_SAMPLES][REC_CH];
static volatile uint32_t s_recHead = 0;       // 측정이 쓴 위치
static volatile uint32_t s_recTail = 0;       // 전송이 읽은 위치
static volatile uint32_t s_recLost = 0;       // 버퍼가 가득 차서 버린 샘플 수
static TaskHandle_t s_sendTask = NULL;
static volatile uint32_t s_lastDropMs = 0;

static void sendTaskFn(void *arg) {
  (void)arg;
  char line[48];
  for (;;) {
    const uint32_t head = s_recHead;
    // GUI가 아직 안 읽고 있으면 데이터가 쌓임 -> 오래된 건 버리고 실시간으로 맞춤
    // (쌓인 옛날 데이터가 녹화 앞부분에 섞여 라벨이 밀리는 것을 막음)
    if (head - s_recTail > REC_MAX_BACKLOG) {
      s_recTail = head;
      s_lastDropMs = millis();
    }
    // LED: 데이터를 버리는 중이면 켜짐. 녹화 중에는 꺼져 있어야 정상
    digitalWrite(LED_BUILTIN, (millis() - s_lastDropMs < 300) ? LOW : HIGH);
    if (s_recTail == head) {                  // 보낼 게 없으면 잠깐 쉼
      vTaskDelay(1);
      continue;
    }
    const uint32_t idx = s_recTail % REC_BUF_SAMPLES;
    int len = snprintf(line, sizeof(line), "%d.0,%d.0,%d.0,%d.0\r\n",
                       s_recBuf[idx][0], s_recBuf[idx][1], s_recBuf[idx][2], s_recBuf[idx][3]);
    Serial.write((const uint8_t *)line, len);
    s_recTail = s_recTail + 1;
  }
}
#endif

#if RUN_MODE == MODE_REPLAY
static uint32_t s_replayVec = 0;
static uint32_t s_replayIdx = 0;
#endif

#if RUN_MODE == MODE_CHECK
static double   s_sum[REC_CH];
static double   s_sq[REC_CH];
static double   s_rawSum[REC_CH];
static uint32_t s_clip[REC_CH];
static uint32_t s_statN = 0;
#endif

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
static void inferTaskFn(void *arg) {
  (void)arg;
  EmgVote vote;
  emg_vote_reset(&vote);
  int lastOutput = EMG_REST_CLASS;
  float probs[EMG_NUM_CLASSES];

  // 힘 세기
  float   strengthPct = 0.0f;     // 부드럽게 만든 세기 (%)
  uint8_t strengthLevel = 0;      // 단계로 나눌 때의 현재 단계
  const int earlyN = (500 / EMG_STEP_LEN) > 0 ? (500 / EMG_STEP_LEN) : 1;   // 처음 0.5초 = 판단 몇 번
  float sumEarly = 0.0f, sumLate = 0.0f;   // 세기 요약용
  int   nEarly = 0, nLate = 0;

#if RUN_MODE == MODE_RUN
  uint8_t prevMode = CTRL_EMG;
#endif

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

#if RUN_MODE == MODE_RUN
    const uint8_t mode = s_ctrlMode;
    if (mode == CTRL_MACRO) {
      // 매크로 모드: 근전도 판단은 하지 않고, 고른 동작을 판단 주기마다 계속 보냄
      // (한 번 못 받아도 다음 전송에서 복구됨. 의수 보드는 같은 명령이 반복되면 무시)
      prevMode = mode;
#if USE_ESPNOW
      sendHandCommand(MACROS[s_macroIdx].cmd, 0);
#endif
      s_busy = false;
      continue;
    }
    if (prevMode != mode) {
      // 매크로 -> 근전도로 돌아온 직후: 예전 판단 기록을 지우고 휴식에서 다시 시작
      emg_vote_reset(&vote);
      lastOutput = EMG_REST_CLASS;
      strengthPct = 0.0f;
      strengthLevel = 0;
      prevMode = mode;
    }
#endif

    uint32_t t0 = micros();
    int pred = emg_classify(s_epoch, probs);
    pred = emg_rest_gate(s_epoch, pred);        // 신호가 휴식 수준이면 무조건 휴식
    int output = emg_vote_update(&vote, pred);
    uint32_t dtMs = (micros() - t0) / 1000UL;

    // ---- 힘 세기 ----
    const float ratio = emg_strength(s_epoch, STRENGTH_WIN_MS * EMG_FS / 1000);   // 쉴 때의 몇 배
    uint8_t strengthOut = 0;                       // 의수로 보낼 값 (0 = 세기 정보 없음)
    if (output != lastOutput) strengthLevel = 0;
    if (output == EMG_REST_CLASS) {
      strengthPct = 0.0f;
    } else {
      const float p = strengthPercent(output, ratio);
      if (output != lastOutput) strengthPct = p;                    // 동작이 시작된 순간에는 바로 반영
      else strengthPct += STRENGTH_SMOOTH * (p - strengthPct);      // 그 뒤로는 부드럽게 따라감
      strengthOut = strengthToByte(strengthPct, &strengthLevel);
    }

    static uint32_t s_decisions = 0;
    s_decisions++;
    const bool printAll = (EMG_STEP_LEN >= 200);                   // 느린 모드: 매번 출력
    const bool printNow = printAll || output != lastOutput ||
                          (s_decisions % (1000 / EMG_STEP_LEN) == 0); // 빠른 모드: 바뀔 때 + 1초마다
    if (printNow) {
    Serial.printf("[%7.2fs] 판단 %-6s %3.0f%% | 출력 %-6s | RMS",
                  s_epochEnd / (float)EMG_FS, EMG_CLASS_NAMES[pred], probs[pred] * 100.0f,
                  EMG_CLASS_NAMES[output]);
    for (int c = 0; c < EMG_NUM_CH; ++c) {
      double acc = 0.0;
      const float *x = s_epoch + c * EMG_EPOCH_LEN;
      for (int n = 0; n < EMG_EPOCH_LEN; ++n) acc += (double)x[n] * x[n];
      Serial.printf(" %4.0f", sqrt(acc / EMG_EPOCH_LEN));
    }
    Serial.printf(" | 세기 %4.1f배 %3d%%", ratio, (int)strengthOut);
    Serial.printf(" | %lu ms", (unsigned long)dtMs);
#if RUN_MODE == MODE_REPLAY
    Serial.printf(" | 재생 중: %s", EMG_CLASS_NAMES[EMG_TEST_LABEL[s_replayVec]]);
#endif
    if (s_overruns) Serial.printf(" | 측정 지연 %lu회", (unsigned long)s_overruns);
    if (s_skipped) Serial.printf(" | 판단 건너뜀 %lu회", (unsigned long)s_skipped);
    Serial.println();
    }

    if (output != lastOutput) {
      onOutputChanged(output);
#if STRENGTH_REPORT
      if (lastOutput != EMG_REST_CLASS && nEarly > 0) {   // 방금 끝난 동작의 세기 요약
        Serial.printf(">>> 세기 요약 %s: 처음 0.5초 평균 %.1f배 (100%% 기준 %.1f배)",
                      EMG_CLASS_NAMES[lastOutput], sumEarly / nEarly, strengthFullFor(lastOutput));
        if (nLate > 0) Serial.printf(", 그 뒤 평균 %.1f배", sumLate / nLate);
        Serial.println();
      }
#endif
      sumEarly = sumLate = 0.0f;
      nEarly = nLate = 0;
      lastOutput = output;
    }
    if (output != EMG_REST_CLASS) {                // 세기 요약용 누적
      if (nEarly < earlyN) { sumEarly += ratio; nEarly++; }
      else                 { sumLate += ratio;  nLate++; }
    }
#if RUN_MODE == MODE_RUN && USE_ESPNOW
    // 판단할 때마다 현재 명령과 힘 세기를 보냄 (바뀌면 바로 반영되고, 한 번 못 받아도 다음에 복구됨)
    sendHandCommand(handCmdFor(output), strengthOut);
#endif
    s_busy = false;
  }
}
#endif

static void sampleOnce() {
  int x[REC_CH] = {0};
#if RUN_MODE == MODE_REPLAY
  for (int c = 0; c < EMG_NUM_CH; ++c) x[EMG_CHANNEL_IDX[c]] = EMG_TEST_RAW[s_replayVec][c][s_replayIdx];
#else
  for (int c = 0; c < REC_CH; ++c) x[c] = readEmgInput(c);
#endif
  s_count++;

#if RUN_MODE == MODE_RECORD
  // 버퍼에 넣기만 하고, 전송은 다른 코어가 담당 (측정이 멈추지 않도록)
  if (s_recHead - s_recTail < REC_BUF_SAMPLES) {
    const uint32_t idx = s_recHead % REC_BUF_SAMPLES;
    for (int c = 0; c < REC_CH; ++c) s_recBuf[idx][c] = (int16_t)x[c];
    s_recHead = s_recHead + 1;
  } else {
    s_recLost = s_recLost + 1;                // 버퍼까지 가득 차면 그때만 손실
  }
  return;
#endif

#if RUN_MODE == MODE_CHECK
  for (int c = 0; c < REC_CH; ++c) {
    s_sum[c] += x[c];
    s_sq[c] += (double)x[c] * x[c];
    s_rawSum[c] += s_lastRaw[c];
    // 클리핑은 필터 전 원본 값(0~4095) 기준으로 확인
    if (s_lastRaw[c] <= 5 || s_lastRaw[c] >= 4090) s_clip[c]++;
  }
  if (++s_statN >= (uint32_t)EMG_FS) {
    Serial.printf("[%6.1fs] 원본평균", s_count / (float)EMG_FS);
    for (int c = 0; c < REC_CH; ++c) Serial.printf(" %5.0f", s_rawSum[c] / s_statN);
    Serial.print(" | 흔들림(표준편차)");
    for (int c = 0; c < REC_CH; ++c) {
      double m = s_sum[c] / s_statN;
      double v = s_sq[c] / s_statN - m * m;
      Serial.printf(" %4.0f", v > 0 ? sqrt(v) : 0.0);
    }
    Serial.print(" | 클리핑");
    for (int c = 0; c < REC_CH; ++c) Serial.printf(" %3lu", (unsigned long)s_clip[c]);
    if (s_overruns) Serial.printf(" | 측정 지연 %lu회", (unsigned long)s_overruns);
    Serial.println();
    for (int c = 0; c < REC_CH; ++c) { s_sum[c] = 0.0; s_sq[c] = 0.0; s_rawSum[c] = 0.0; s_clip[c] = 0; }
    s_statN = 0;
  }
  return;
#endif

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
  for (int c = 0; c < EMG_NUM_CH; ++c) {
    s_ring[c][s_ringPos] = emg_filter_step(&s_filters[c], (float)x[EMG_CHANNEL_IDX[c]]);
  }
  s_ringPos = (s_ringPos + 1) % EMG_EPOCH_LEN;

#if RUN_MODE == MODE_REPLAY
  if (++s_replayIdx >= (uint32_t)EMG_TEST_LEN) {
    s_replayIdx = 0;
    s_replayVec = (s_replayVec + 1) % EMG_TEST_COUNT;
  }
#endif

  if (s_count >= FIRST_DECISION && (s_count - FIRST_DECISION) % EMG_STEP_LEN == 0 && s_busy) {
    s_skipped = s_skipped + 1;   // 판단 주기가 계산 시간보다 짧음
  }
  if (s_count >= FIRST_DECISION && (s_count - FIRST_DECISION) % EMG_STEP_LEN == 0 && !s_busy) {
    for (int c = 0; c < EMG_NUM_CH; ++c) {
      float *dst = s_epoch + c * EMG_EPOCH_LEN;
      const uint32_t tail = EMG_EPOCH_LEN - s_ringPos;
      memcpy(dst, &s_ring[c][s_ringPos], tail * sizeof(float));
      memcpy(dst + tail, &s_ring[c][0], s_ringPos * sizeof(float));
    }
    s_epochEnd = s_count;
    s_busy = true;
    xTaskNotifyGive(s_inferTask);
  }
#endif
}

void setup() {
  Serial.begin(115200);
  setupEmgInput();

#if RUN_MODE == MODE_RECORD
  // 녹화 모드는 GUI가 읽기 때문에 안내 문구를 출력하지 않음
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);            // 꺼진 상태 (PC가 안 읽어서 데이터를 버리는 중이면 켜짐)
  xTaskCreatePinnedToCore(sendTaskFn, "emg_send", 4096, NULL, 1, &s_sendTask, 0);
#else
  delay(2000);
  Serial.println();
#if RUN_MODE == MODE_REPLAY
  Serial.println("===== MODE_REPLAY: 저장된 테스트 데이터를 실시간처럼 흘려보냅니다 =====");
#elif RUN_MODE == MODE_CHECK
  Serial.println("===== MODE_CHECK: 1초마다 채널별 상태를 출력합니다 =====");
  Serial.println("- 쉴 때 흔들림이 작고, 힘줄 때 해당 채널의 흔들림이 커지면 정상");
  Serial.println("- EMGFilters를 거친 값이라 평균은 0 근처입니다");
  Serial.println("- 원본평균이 4095나 0에 가깝거나 클리핑 숫자가 0보다 크면 센서 연결/전압 확인");
#else
  Serial.println("===== MODE_RUN: 근전도 모드로 시작합니다 =====");
  Serial.println("- 모드 전환(SW1): 시리얼에서 1 전송, 또는 BOOT 버튼 길게");
  Serial.println("- 다음 동작(SW2): 시리얼에서 2 전송, 또는 BOOT 버튼 짧게 (매크로 모드에서)");
#endif
#endif

  for (int c = 0; c < EMG_NUM_CH; ++c) emg_filter_reset(&s_filters[c]);

#if RUN_MODE == MODE_RUN && USE_ESPNOW
  setupEspNow();
#endif
#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
#if ARDUINO_USB_CDC_ON_BOOT && ARDUINO_USB_MODE
  // 판단 모드에서는 시리얼 출력 때문에 판단이 늦어지지 않게, PC가 안 읽으면 출력을 버림
  Serial.setTxTimeoutMs(0);
#endif
  emg_init();   // 첫 판단이 늦어서 건너뛰지 않도록 계산용 표를 미리 준비
  xTaskCreatePinnedToCore(inferTaskFn, "emg_infer", 8192, NULL, 1, &s_inferTask, 0);
#endif
#if RUN_MODE == MODE_RUN
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);            // 꺼짐 = 근전도 모드, 켜짐 = 매크로 모드
  xTaskCreatePinnedToCore(ctrlTaskFn, "emg_ctrl", 4096, NULL, 2, NULL, 0);
#endif

  s_nextUs = micros();
}

void loop() {
  const uint32_t now = micros();
  if ((int32_t)(now - s_nextUs) < 0) return;  // 아직 1ms가 안 됨
  s_nextUs += SAMPLE_US;
  if ((int32_t)(now - s_nextUs) > (int32_t)(200 * SAMPLE_US)) {
    s_nextUs = now + SAMPLE_US;  // 0.2초 넘게 밀리면 포기하고 다시 맞춤
    s_overruns++;
  }
  sampleOnce();
}
