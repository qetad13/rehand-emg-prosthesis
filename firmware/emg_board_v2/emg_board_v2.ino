// =====================================================================
// ReHand 근전도 보드 v2 (XIAO ESP32-S3)
//
// 녹화 / 센서 점검 / 실시간 판단 / 재생 / 자가 점검을 이 스케치 하나로 합니다. (RUN_MODE로 선택)
// 녹화(MODE_RECORD)와 판단(MODE_RUN)이 같은 readEmgInput()을 쓰기 때문에 학습 데이터와 실시간 입력이 똑같습니다.
// 녹화 형식은 예전(v1-stft-cnn 브랜치의 esp32_emg_realtime)과 같아서, 예전 녹화 데이터와 수집 GUI를 그대로 씁니다.
//
// 예전 스케치(esp32_emg_realtime)와 달라진 점
//   1. 판단 방식 v2 (emg2.*): 0.75초 창 STFT+CNN -> 25ms마다 스트리밍 특징 + 작은 TCN
//      정확도는 올라가고 반응은 빨라지고, 계산은 1회 판단 기준 약 1/50 · 1초 기준 약 1/13 (docs/v2_results.md 참고)
//   2. 의수와 양방향 연결 (RehandLink.h): 의수의 연결/과열/전압/서보 이상을 LED와 시리얼로 알려 줌
//   3. 사용자 보정 (SW1 3초): 의수가 동작을 하나씩 보여주면 따라 하기만 하면 됨 (약 40초)
//      전극을 다시 붙여 채널별 신호 크기가 달라진 것을 맞추고, 동작별 힘 세기 기준도 다시 잡아 저장
//   4. 잠금 모드 (근전도 모드에서 SW2 짧게): 한 번 힘주면 그 자세로 유지, 다시 힘주면 풀림
//      (물건을 오래 쥐고 있을 때 계속 힘주지 않아도 됨)
//
// 스위치
//   SW1 짧게 : 근전도 모드 <-> 매크로 모드 (LED가 켜져 있으면 매크로 모드)
//   SW1 3초  : 사용자 보정 시작 (근전도 모드에서)
//   SW2 짧게 : 근전도 모드 = 잠금 모드 켜기/끄기 (켜면 LED 2번, 끄면 1번 깜빡임) / 매크로 모드 = 다음 동작
//   SW2 3초  : 의수 호밍 (근전도 모드에서, LED 3번 깜빡임)
// 스위치가 없어도 시리얼 모니터에서 글자를 보내면 됩니다.
//   1 = SW1, 2 = SW2, h = 호밍, c = 사용자 보정, x = 보정 지우기, l = 잠금 모드, ? = 상태
//   (BOOT 버튼: 짧게 = SW2, 0.8초 = SW1)
//
// LED (근전도 모드에서 평소에는 꺼져 있음)
//   2초마다 1번 짧게 : 의수와 연결 안 됨 (의수 전원/거리 확인)
//   2초마다 2번 짧게 : 의수 이상 (과열, 전압 낮음, 서보 응답 없음) -> 시리얼에 원인 출력
//   보정 중 켜짐     : 의수가 보여주는 동작을 따라 힘주기 / 꺼짐 : 힘 빼기
// =====================================================================
#include "emg2.h"
#include "emg2_test_vectors.h"
#include "EMGFilters.h"   // 채널마다 필터 상태를 따로 갖도록 수정된 버전
#include <Preferences.h>

// ---------------------------------------------------------------------
// 사용자 설정
// ---------------------------------------------------------------------
#define MODE_REPLAY   0   // 센서 없이 저장된 데이터로 판단 흐름 확인
#define MODE_CHECK    1   // 센서 상태 확인 (채널별 평균/흔들림/클리핑)
#define MODE_RUN      2   // 실시간 판단 + 의수로 전송
#define MODE_RECORD   3   // GUI로 녹화할 때
#define MODE_SELFTEST 4   // PC 계산 결과와 일치하는지 + 계산 시간 확인 (모델을 새로 학습한 뒤 한 번 실행)
#define RUN_MODE      MODE_RUN

// 녹화는 항상 4채널 (CSV의 Ch 00, Ch 01, Ch 02, Ch 03 순서)
// 판단에는 그중 모델이 쓰는 채널(emg2_model.h의 EMG2_CHANNEL_IDX)만 사용
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
#define USE_LINK 1

#if RUN_MODE == MODE_RUN && USE_LINK
#define REHAND_ROLE_EMG
#include "RehandLink.h"
#endif

// 판단 결과 -> 의수 보드 명령 번호 (이름으로 연결해서 동작 순서가 바뀌어도 안전)
//   1 엄지, 2 검지, 3 중지, 4 약지, 5 소지, 6 휴식, 7 주먹, 8 집게
[[maybe_unused]] static uint8_t handCmdFor(int gesture) {
  const char *n = EMG2_CLASS_NAMES[gesture];
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
#define HAND_CMD_REST   6
#define HAND_CMD_HOMING 20
#define HOMING_SEND_N   3     // 무선은 가끔 빠지므로 여러 번 보냄 (의수 쪽에서 1번만 실행)
#define SEND_EVERY_N    4     // 명령이 안 바뀌어도 이 판단 횟수마다 다시 보냄 (4번 = 0.1초)

// ---------------------------------------------------------------------
// 힘 세기 (세게 주면 빠르고 세게, 약하게 주면 천천히 약하게)
//   세기(배) = 최근 0.2초 신호 크기가 쉴 때의 몇 배인지 (4채널 중 가장 큰 값)
//   세기(%)  = 휴식 수준을 0%, 동작별 기준을 100%로 놓고 바꾼 값 -> 1~100으로 의수에 전송
// 동작별 100% 기준은 학습 데이터에서 자동으로 구해 emg2_model.h에 들어 있고(EMG2_STRENGTH_FULL),
// 사용자 보정을 하면 그때 준 힘이 새 100% 기준이 됩니다.
// ---------------------------------------------------------------------
#define STRENGTH_SMOOTH  0.2f   // 0~1. 작을수록 값이 부드럽지만 반응이 느림 (25ms마다 적용)
#define STRENGTH_FLOOR   1.5f   // 이 배수 이하는 0%
#define STRENGTH_REPORT  1      // 동작이 끝날 때마다 세기 요약을 출력

// ---------------------------------------------------------------------
// 스위치
// ---------------------------------------------------------------------
#define SW1_PIN         D7    // PCB 8번 패드. 스위치가 없는 보드면 -1
#define SW2_PIN         D8    // PCB 9번 패드. 스위치가 없는 보드면 -1
#define SW_PRESSED      LOW   // 누르면 GND에 연결되는 회로 기준
#define USE_SERIAL_KEYS 1     // 시리얼 글자로 스위치 흉내
#define USE_BOOT_BTN    1     // BOOT 버튼으로 스위치 흉내
#define BOOT_BTN_PIN    0     // XIAO ESP32-S3의 BOOT 버튼 (누르면 LOW)
#define LONG_PRESS_MS   800   // BOOT 버튼을 이 시간 이상 누르면 SW1로 처리
#define HOLD_PRESS_MS   3000  // SW1/SW2를 이 시간 이상 누르면 보정/호밍

// 매크로 모드에서 SW2를 누를 때마다 이 순서로 넘어갑니다. {이름, 의수 명령 번호}
struct MacroItem { const char *name; uint8_t cmd; };
static const MacroItem MACROS[] = {
  {"전체 쥐기", 11},
  {"반쯤 펴기", 12},
  {"가리키기", 13},
  {"집게 파지", 14},
  {"세 손가락 파지", 15},
  {"브이", 16},
  {"가위", 17},
  {"따봉", 18},
  {"네 손가락 파지", 19},
};
#define MACRO_COUNT ((int)(sizeof(MACROS) / sizeof(MACROS[0])))

#define CTRL_EMG   0   // 근전도 모드
#define CTRL_MACRO 1   // 매크로 모드

// ---------------------------------------------------------------------
// 사용자 보정
//   동작마다: 의수가 동작을 보여줌(LED 켜짐) -> 1.2초 뒤부터 2초 동안 측정 -> 휴식 2초(LED 꺼짐)
// ---------------------------------------------------------------------
#define CAL_PROMPT_FRAMES 48     // 따라 할 시간 (1.2초)
#define CAL_RECORD_FRAMES 80     // 측정 (2초)
#define CAL_REST_FRAMES   80     // 휴식 (2초, 뒤쪽 1초로 쉴 때 잡음 확인)
#define CAL_MIN_RATIO     1.5f   // 측정한 신호가 쉴 때의 이 배수보다 작으면 그 동작은 실패로 봄
#define CAL_MIN_GESTURES  4      // 성공한 동작이 이보다 적으면 보정을 적용하지 않음
#define CAL_MAX_GAIN      0.7f   // 채널별 보정 한계 (로그 값, 0.7 = 약 2배)
#define CAL_MAGIC         0x32434C52UL

struct UserCal {
  uint32_t magic;
  uint8_t  numCh, numClasses;
  float    gain[EMG2_NUM_CH];          // 채널별 로그 진폭 보정값
  float    full[EMG2_NUM_CLASSES];     // 동작별 힘 세기 100% 기준 (0이면 기본값 사용)
};

// ---------------------------------------------------------------------
// 내부 동작 (보통 수정할 필요 없음)
// ---------------------------------------------------------------------
static const uint32_t SAMPLE_US = 1000000UL / EMG2_FS;
static uint32_t s_count = 0;
static uint32_t s_nextUs = 0;
static uint32_t s_overruns = 0;

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
static Emg2Front     s_front;                 // 측정 루프(코어 1)만 사용
static Emg2Back      s_back;                  // 판단 태스크(코어 0)만 사용
static Emg2Decider   s_dec;
static QueueHandle_t s_hopQ = NULL;           // 측정 루프 -> 판단 태스크 (25ms마다 하나)
static volatile uint32_t s_dropped = 0;       // 판단 태스크가 밀려서 버린 횟수 (0이어야 정상)
#endif

#if RUN_MODE == MODE_REPLAY
static uint32_t s_replayVec = 0;
static uint32_t s_replayIdx = 0;
#endif

#if RUN_MODE == MODE_RUN
static volatile uint8_t s_ctrlMode = CTRL_EMG;
static volatile uint8_t s_macroIdx = 0;
static volatile bool    s_lockMode = false;   // 잠금 모드
static volatile bool    s_reqHoming = false;  // 스위치 태스크 -> 판단 태스크 요청 (무선 전송은 판단 태스크만 함)
static volatile bool    s_reqCal = false;
static volatile bool    s_reqCalReset = false;
static volatile bool    s_reqStatus = false;
static volatile bool    s_calActive = false;
static volatile bool    s_calLed = false;
static volatile uint8_t s_blinkReq = 0;       // LED를 이 횟수만큼 깜빡여 달라는 요청
static UserCal          s_cal;
static Preferences      s_prefs;

static void calSetDefault() {
  memset(&s_cal, 0, sizeof(s_cal));
  s_cal.magic = CAL_MAGIC;
  s_cal.numCh = EMG2_NUM_CH;
  s_cal.numClasses = EMG2_NUM_CLASSES;
}

static bool calLoad() {
  UserCal c;
  s_prefs.begin("emg2", true);
  const size_t n = s_prefs.getBytes("cal", &c, sizeof(c));
  s_prefs.end();
  if (n != sizeof(c) || c.magic != CAL_MAGIC || c.numCh != EMG2_NUM_CH || c.numClasses != EMG2_NUM_CLASSES) {
    calSetDefault();
    return false;
  }
  s_cal = c;
  return true;
}

static void calSave() {
  s_prefs.begin("emg2", false);
  s_prefs.putBytes("cal", &s_cal, sizeof(s_cal));
  s_prefs.end();
}

static float strengthFullFor(int gesture) {
  const float f = s_cal.full[gesture];
  return f > 0.0f ? f : EMG2_STRENGTH_FULL[gesture];
}

// 세기(배) -> 0~100%
static float strengthPercent(int gesture, float ratio) {
  float hi = strengthFullFor(gesture);
  if (hi < STRENGTH_FLOOR + 0.5f) hi = STRENGTH_FLOOR + 0.5f;
  const float p = (ratio - STRENGTH_FLOOR) / (hi - STRENGTH_FLOOR) * 100.0f;
  return p < 0.0f ? 0.0f : (p > 100.0f ? 100.0f : p);
}

static uint8_t strengthToByte(float pct) {
  const int v = (int)(pct + 0.5f);
  return (uint8_t)(v < 1 ? 1 : (v > 100 ? 100 : v));
}

static void sendHand(uint8_t cmd, uint8_t strength) {
#if USE_LINK
  rhEmgSend(cmd, strength);
#else
  (void)cmd; (void)strength;
#endif
}

// ---- 사용자 보정 진행 (판단 태스크 안에서 25ms마다 호출) ----
struct CalRun {
  int   g;                                        // 지금 하는 동작 (클래스 번호)
  int   phase;                                    // 0 따라 하기, 1 측정, 2 휴식
  int   n;                                        // 이 단계에서 지난 프레임 수
  float rec[CAL_RECORD_FRAMES][EMG2_NUM_CH];
  float med[EMG2_NUM_CLASSES][EMG2_NUM_CH];       // 동작별 채널별 로그 진폭 중앙값
  bool  ok[EMG2_NUM_CLASSES];
  float noise[EMG2_NUM_CH];                       // 쉴 때 로그 진폭 (0 근처여야 정상) 중 가장 큰 값
};
static CalRun s_calRun;

static float medianOf(float *v, int n) {
  for (int i = 1; i < n; ++i) {                   // 삽입 정렬 (n이 작음)
    const float x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; --j; }
    v[j + 1] = x;
  }
  return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static int calNextGesture(int g) {
  for (++g; g < EMG2_NUM_CLASSES; ++g) if (g != EMG2_REST_CLASS) return g;
  return -1;
}

static void calStart() {
  memset(&s_calRun, 0, sizeof(s_calRun));
  for (int c = 0; c < EMG2_NUM_CH; ++c) s_calRun.noise[c] = -10.0f;
  s_calRun.g = calNextGesture(-1);
  s_calActive = true;
  s_calLed = true;
  Serial.println("=== 사용자 보정 시작 ===");
  Serial.println("LED가 켜지면 의수가 보여주는 동작을 평소 힘으로 따라 하고, 꺼지면 힘을 빼세요.");
  Serial.printf(">>> 따라 하기: %s\n", EMG2_CLASS_NAMES[s_calRun.g]);
}

static void calFinish() {
  s_calActive = false;
  s_calLed = false;
  int nOk = 0;
  for (int g = 0; g < EMG2_NUM_CLASSES; ++g) nOk += s_calRun.ok[g] ? 1 : 0;
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    if (s_calRun.noise[c] > 0.7f)
      Serial.printf("[보정] 주의: Ch %02d가 쉴 때도 평소의 %.1f배로 큼 -> 전극 접촉 확인\n",
                    EMG2_CHANNEL_IDX[c], expf(s_calRun.noise[c]));
  }
  if (nOk < CAL_MIN_GESTURES) {
    Serial.printf("=== 보정 실패: 신호가 잡힌 동작이 %d개뿐 (최소 %d개). 예전 보정을 그대로 씁니다 ===\n",
                  nOk, CAL_MIN_GESTURES);
    s_blinkReq = 6;
    return;
  }
  // 채널별 보정값: (측정 - 학습 기준)의 동작 중앙값. 한두 동작이 어긋나도 영향이 작음
  float diffs[EMG2_NUM_CLASSES];
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    int n = 0;
    for (int g = 0; g < EMG2_NUM_CLASSES; ++g) {
      if (s_calRun.ok[g]) diffs[n++] = s_calRun.med[g][c] - EMG2_CAL_LOGAMP[g * EMG2_NUM_CH + c];
    }
    float d = medianOf(diffs, n);
    if (d > CAL_MAX_GAIN) d = CAL_MAX_GAIN;
    if (d < -CAL_MAX_GAIN) d = -CAL_MAX_GAIN;
    s_cal.gain[c] = d;
  }
  // 동작별 힘 세기 100% 기준: 방금 준 힘
  for (int g = 0; g < EMG2_NUM_CLASSES; ++g) {
    s_cal.full[g] = 0.0f;
    if (!s_calRun.ok[g]) continue;
    float best = 0.0f;
    for (int c = 0; c < EMG2_NUM_CH; ++c) {
      const float r = expf(s_calRun.med[g][c] - s_cal.gain[c]);
      if (r > best) best = r;
    }
    s_cal.full[g] = best;
  }
  emg2_back_set_gain(&s_back, s_cal.gain);
  calSave();
  Serial.printf("=== 보정 완료 (%d개 동작 사용) - 저장됨 ===\n", nOk);
  Serial.print("채널별 신호 크기 (학습 때 대비):");
  for (int c = 0; c < EMG2_NUM_CH; ++c) Serial.printf(" Ch %02d x%.2f", EMG2_CHANNEL_IDX[c], expf(s_cal.gain[c]));
  Serial.println();
  Serial.print("동작별 힘 세기 100% 기준(배):");
  for (int g = 0; g < EMG2_NUM_CLASSES; ++g) {
    if (g != EMG2_REST_CLASS) Serial.printf(" %s %.1f%s", EMG2_CLASS_NAMES[g], strengthFullFor(g), s_calRun.ok[g] ? "" : "(기본)");
  }
  Serial.println();
  s_blinkReq = 2;
}

// 프레임마다 호출. 보정이 진행 중이면 true (그동안 평소 판단/전송은 하지 않음)
static bool calFrame(uint32_t frameNo) {
  if (!s_calActive) return false;
  CalRun &r = s_calRun;
  const bool gesturePhase = (r.phase < 2);
  if (frameNo % SEND_EVERY_N == 0 || r.n == 0) {
    sendHand(gesturePhase ? handCmdFor(r.g) : HAND_CMD_REST, 0);     // 의수가 동작을 보여줌
  }
  if (r.phase == 1) {
    memcpy(r.rec[r.n], s_back.logamp, sizeof(r.rec[0]));
  } else if (r.phase == 2 && r.n >= CAL_REST_FRAMES / 2) {
    for (int c = 0; c < EMG2_NUM_CH; ++c) {
      // 휴식 후반부의 프레임 중 가장 조용한 쪽이 아니라 평균적인 수준을 보려고 느리게 따라감
      const float v = s_back.logamp[c];
      if (r.n == CAL_REST_FRAMES / 2 && r.noise[c] < -5.0f) r.noise[c] = v;
      else r.noise[c] += 0.05f * (v - r.noise[c]);
    }
  }
  r.n++;

  if (r.phase == 0 && r.n >= CAL_PROMPT_FRAMES) { r.phase = 1; r.n = 0; }
  else if (r.phase == 1 && r.n >= CAL_RECORD_FRAMES) {
    float col[CAL_RECORD_FRAMES];
    float best = -10.0f;
    for (int c = 0; c < EMG2_NUM_CH; ++c) {
      for (int i = 0; i < CAL_RECORD_FRAMES; ++i) col[i] = r.rec[i][c];
      r.med[r.g][c] = medianOf(col, CAL_RECORD_FRAMES);
      if (r.med[r.g][c] > best) best = r.med[r.g][c];
    }
    r.ok[r.g] = (best >= logf(CAL_MIN_RATIO));
    Serial.printf("    %s: 쉴 때의 %.1f배 %s\n", EMG2_CLASS_NAMES[r.g], expf(best),
                  r.ok[r.g] ? "" : "-> 너무 약해서 이 동작은 건너뜀");
    r.phase = 2; r.n = 0;
    s_calLed = false;
    Serial.println(">>> 힘 빼기");
  } else if (r.phase == 2 && r.n >= CAL_REST_FRAMES) {
    r.g = calNextGesture(r.g);
    if (r.g < 0) {
      calFinish();
    } else {
      r.phase = 0; r.n = 0;
      s_calLed = true;
      Serial.printf(">>> 따라 하기: %s\n", EMG2_CLASS_NAMES[r.g]);
    }
  }
  return true;
}

// ---- 스위치 처리 (스위치 태스크에서 호출) ----
static void ledSet(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }
static void ledBlink(int n) {
  for (int i = 0; i < n; ++i) {
    ledSet(true);  vTaskDelay(pdMS_TO_TICKS(100));
    ledSet(false); vTaskDelay(pdMS_TO_TICKS(100));
  }
}

static void onModeToggle() {
  if (s_calActive) return;
  if (s_ctrlMode == CTRL_EMG) {
    s_macroIdx = 0;
    s_ctrlMode = CTRL_MACRO;
    Serial.printf("=== 매크로 모드 === 동작 1/%d: %s (의수 명령 %d)\n", MACRO_COUNT, MACROS[0].name, MACROS[0].cmd);
  } else {
    s_ctrlMode = CTRL_EMG;
    Serial.println("=== 근전도 모드 === (휴식에서 다시 시작)");
  }
}

static void onSw2Short() {
  if (s_calActive) return;
  if (s_ctrlMode == CTRL_MACRO) {
    const uint8_t i = (uint8_t)((s_macroIdx + 1) % MACRO_COUNT);
    s_macroIdx = i;
    Serial.printf(">>> 매크로 동작 %d/%d: %s (의수 명령 %d)\n", i + 1, MACRO_COUNT, MACROS[i].name, MACROS[i].cmd);
  } else {
    s_lockMode = !s_lockMode;
    Serial.println(s_lockMode ? "=== 잠금 모드 켬: 한 번 힘주면 유지, 다시 힘주면 풀림 ==="
                              : "=== 잠금 모드 끔: 힘주는 동안만 동작 ===");
    ledBlink(s_lockMode ? 2 : 1);
  }
}

static void onHomingRequested() {
  if (s_calActive || s_ctrlMode != CTRL_EMG) return;
  s_reqHoming = true;
}

static void onCalRequested() {
  if (s_calActive) return;
  if (s_ctrlMode != CTRL_EMG) {
    Serial.println("[보정] 근전도 모드에서만 할 수 있습니다");
    return;
  }
  s_reqCal = true;
}

// 버튼 하나의 상태 (10ms마다 확인, 3번 연속 같은 값이어야 인정 = 떨림 제거)
#define BTN_NONE  0
#define BTN_SHORT 1
#define BTN_LONG  2
struct Button {
  int      pin;            // -1이면 없는 버튼
  int      pressedLevel;
  bool     down;
  uint8_t  same;
  uint32_t downMs;
  bool     longDone;

  void begin(int p, int level) {
    pin = p; pressedLevel = level;
    down = false; same = 0; downMs = 0; longDone = false;
    if (pin >= 0) pinMode(pin, level == LOW ? INPUT_PULLUP : INPUT_PULLDOWN);
  }

  // longMs = 0 : 누르는 순간 BTN_SHORT
  // longMs > 0 : 짧게 눌렀다 떼면 BTN_SHORT, longMs 이상 누르고 있으면 BTN_LONG
  int poll(uint32_t longMs) {
    const bool useLong = (longMs > 0);
    if (pin < 0) return BTN_NONE;
    const bool raw = (digitalRead(pin) == pressedLevel);
    if (raw == down) {
      same = 0;
    } else if (++same >= 3) {
      same = 0;
      down = raw;
      if (raw) {
        downMs = millis();
        longDone = false;
        if (!useLong) return BTN_SHORT;
      } else if (useLong && !longDone) {
        return BTN_SHORT;
      }
    }
    if (useLong && down && !longDone && millis() - downMs >= longMs) {
      longDone = true;
      return BTN_LONG;
    }
    return BTN_NONE;
  }
};

// 스위치 + LED 전용 태스크 (코어 0). 1kHz 측정 루프(코어 1)는 건드리지 않음
static void ctrlTaskFn(void *arg) {
  (void)arg;
  Button sw1, sw2, boot;
  sw1.begin(SW1_PIN, SW_PRESSED);
  sw2.begin(SW2_PIN, SW_PRESSED);
  boot.begin(USE_BOOT_BTN ? BOOT_BTN_PIN : -1, LOW);
  uint8_t lastWarn = 0xFF;

  for (;;) {
#if USE_SERIAL_KEYS
    while (Serial.available() > 0) {
      const int ch = Serial.read();
      if (ch == '1') onModeToggle();
      else if (ch == '2') onSw2Short();
      else if (ch == 'h' || ch == 'H') onHomingRequested();
      else if (ch == 'c' || ch == 'C') onCalRequested();
      else if (ch == 'x' || ch == 'X') s_reqCalReset = true;
      else if (ch == 'l' || ch == 'L') { if (s_ctrlMode == CTRL_EMG) onSw2Short(); }
      else if (ch == '?') s_reqStatus = true;                    // 줄바꿈 등 다른 글자는 무시
    }
#endif
    // SW1: 짧게(뗄 때) = 모드 전환, 3초 = 사용자 보정
    const int e1 = sw1.poll(HOLD_PRESS_MS);
    if (e1 == BTN_LONG) onCalRequested();
    else if (e1 == BTN_SHORT) onModeToggle();
    // SW2: 근전도 모드에서는 짧게(뗄 때) = 잠금 모드, 3초 = 호밍 / 매크로 모드에서는 누르는 순간 다음 동작
    const int e2 = sw2.poll(s_ctrlMode == CTRL_EMG ? HOLD_PRESS_MS : 0);
    if (e2 == BTN_LONG) onHomingRequested();
    else if (e2 == BTN_SHORT) onSw2Short();
    const int ev = boot.poll(LONG_PRESS_MS);
    if (ev == BTN_LONG) onModeToggle();
    else if (ev == BTN_SHORT) onSw2Short();

    // ---- LED ----
    if (s_blinkReq) {
      const uint8_t n = s_blinkReq;
      s_blinkReq = 0;
      ledBlink(n);
    }
    bool led = false;
    uint8_t warn = 0;                       // 0 정상, 1 의수 연결 안 됨, 2 의수 이상
#if USE_LINK
    if (!rhEmgHandAlive()) warn = 1;
    else if (rhEmgHandFlags() & (HAND_FLAG_HOT | HAND_FLAG_LOW_VOLT | HAND_FLAG_SERVO_ERR)) warn = 2;
#endif
    if (s_calActive) {
      led = s_calLed;
    } else if (s_ctrlMode == CTRL_MACRO) {
      led = true;
    } else if (warn) {
      const uint32_t ph = millis() % 2000;
      led = (ph < 80) || (warn == 2 && ph >= 240 && ph < 320);
    }
    ledSet(led);

#if USE_LINK
    if (warn != lastWarn && millis() > 3000) {            // 상태가 바뀔 때만 출력
      lastWarn = warn;
      if (warn == 0) Serial.println("[의수] 연결됨, 정상");
      else if (warn == 1) Serial.println("[의수] 연결 안 됨 - 의수 전원과 거리를 확인하세요");
      else {
        const uint8_t f = rhEmgHandFlags();
        Serial.print("[의수] 이상:");
        if (f & HAND_FLAG_HOT) Serial.printf(" 과열(%u°C, 힘을 줄여 동작 중)", rhEmgHandTemp());
        if (f & HAND_FLAG_LOW_VOLT) Serial.printf(" 전압 낮음(%u.%uV)", rhEmgHandVolt() / 10, rhEmgHandVolt() % 10);
        if (f & HAND_FLAG_SERVO_ERR) Serial.print(" 서보 응답 없음");
        Serial.println();
      }
    }
#endif
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}
#endif  // RUN_MODE == MODE_RUN

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
    if (head - s_recTail > REC_MAX_BACKLOG) {
      s_recTail = head;
      s_lastDropMs = millis();
    }
    // LED: 데이터를 버리는 중이면 켜짐. 녹화 중에는 꺼져 있어야 정상
    digitalWrite(LED_BUILTIN, (millis() - s_lastDropMs < 300) ? LOW : HIGH);
    if (s_recTail == head) {
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

#if RUN_MODE == MODE_CHECK
static double   s_sum[REC_CH];
static double   s_sq[REC_CH];
static double   s_rawSum[REC_CH];
static uint32_t s_clip[REC_CH];
static uint32_t s_statN = 0;
#endif

// ---------------------------------------------------------------------
// 판단 태스크 (코어 0): 25ms마다 하나씩 오는 누적값으로 판단하고 의수로 보냄
// ---------------------------------------------------------------------
#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
static void inferTaskFn(void *arg) {
  (void)arg;
  Emg2Hop hop;
  uint32_t frameNo = 0;
  uint32_t maxUs = 0;
  int   lastOutput = EMG2_REST_CLASS;
  float strengthPct = 0.0f;
  const int earlyN = 500 * EMG2_FS / 1000 / EMG2_HOP;      // 처음 0.5초 = 판단 몇 번
  float sumEarly = 0.0f, sumLate = 0.0f;                   // 세기 요약용
  int   nEarly = 0, nLate = 0;
#if RUN_MODE == MODE_RUN
  uint8_t prevMode = CTRL_EMG;
  int     lockLatched = EMG2_REST_CLASS;   // 잠금 모드에서 유지 중인 동작
  bool    lockOwn = false;                 // 지금 수축이 그 동작을 잠근 수축인가
  uint8_t lockStrength = 0;
  uint8_t lastSentCmd = 0;
#endif

  for (;;) {
    xQueueReceive(s_hopQ, &hop, portMAX_DELAY);
    const uint32_t t0 = micros();
    const int pred = emg2_back_frame(&s_back, &hop);
    const uint32_t dtUs = micros() - t0;
    if (dtUs > maxUs) maxUs = dtUs;
    frameNo++;

#if RUN_MODE == MODE_RUN
    // ---- 스위치 태스크가 맡긴 일 ----
    if (s_reqHoming) {
      s_reqHoming = false;
      Serial.println("=== 의수 호밍 요청 전송 ===");
#if USE_LINK
      rhEmgSendEvent(HAND_CMD_HOMING, HOMING_SEND_N);
#endif
      lastSentCmd = HAND_CMD_HOMING;
      s_blinkReq = 3;
    }
    if (s_reqCalReset) {
      s_reqCalReset = false;
      calSetDefault();
      calSave();
      emg2_back_set_gain(&s_back, NULL);
      Serial.println("=== 사용자 보정을 지우고 기본값(학습 데이터 기준)으로 되돌림 ===");
    }
    if (s_reqCal) {
      s_reqCal = false;
      calStart();
    }
    if (s_reqStatus) {
      s_reqStatus = false;
      Serial.printf("[상태] %s 모드, 잠금 %s, 판단 1회 최대 %lu us, 밀린 판단 %lu, 측정 지연 %lu\n",
                    s_ctrlMode == CTRL_MACRO ? "매크로" : "근전도", s_lockMode ? "켬" : "끔",
                    (unsigned long)maxUs, (unsigned long)s_dropped, (unsigned long)s_overruns);
      Serial.print("  보정(채널별 배율):");
      for (int c = 0; c < EMG2_NUM_CH; ++c) Serial.printf(" x%.2f", expf(s_cal.gain[c]));
      Serial.print(" | 지금 신호(쉴 때 대비 배):");
      for (int c = 0; c < EMG2_NUM_CH; ++c) Serial.printf(" %.1f", expf(s_back.logamp[c]));
      Serial.println();
#if USE_LINK
      if (rhEmgHandAlive()) Serial.printf("  의수: 연결됨, 최고 온도 %u°C, 전압 %u.%uV, 상태 0x%02X\n", rhEmgHandTemp(),
                                          rhEmgHandVolt() / 10, rhEmgHandVolt() % 10, rhEmgHandFlags());
      else Serial.println("  의수: 연결 안 됨");
#endif
    }

    if (calFrame(frameNo)) {                 // 보정 중에는 평소 판단/전송을 하지 않음
      lastSentCmd = 0;
      prevMode = 0xFF;                       // 끝나면 아래에서 휴식부터 다시 시작
      continue;
    }

    const uint8_t mode = s_ctrlMode;
    if (mode == CTRL_MACRO) {
      // 매크로 모드: 근전도 판단은 쓰지 않고, 고른 동작을 계속 보냄 (한 번 못 받아도 다음 전송에서 복구됨)
      prevMode = mode;
      const uint8_t cmd = MACROS[s_macroIdx].cmd;
      if (cmd != lastSentCmd || frameNo % SEND_EVERY_N == 0) {
        sendHand(cmd, 0);
        lastSentCmd = cmd;
      }
      continue;
    }
    if (prevMode != mode) {
      // 매크로/보정 -> 근전도로 돌아온 직후: 예전 판단 기록을 지우고 휴식에서 다시 시작
      emg2_decide_reset(&s_dec);
      s_dec.frames = EMG2_WARMUP_FRAMES;     // 필터는 이미 안정된 상태
      lastOutput = EMG2_REST_CLASS;
      strengthPct = 0.0f;
      lockLatched = EMG2_REST_CLASS;
      lockOwn = false;
      prevMode = mode;
    }
#endif

    const int output = emg2_decide(&s_dec, s_back.probs, s_back.strength);

    // ---- 힘 세기 ----
    const float ratio = s_back.strength;                 // 쉴 때의 몇 배
    uint8_t strengthOut = 0;                             // 의수로 보낼 값 (0 = 세기 정보 없음)
#if RUN_MODE == MODE_RUN
    if (output == EMG2_REST_CLASS) {
      strengthPct = 0.0f;
    } else {
      const float p = strengthPercent(output, ratio);
      if (output != lastOutput) strengthPct = p;                    // 동작이 시작된 순간에는 바로 반영
      else strengthPct += STRENGTH_SMOOTH * (p - strengthPct);      // 그 뒤로는 부드럽게 따라감
      strengthOut = strengthToByte(strengthPct);
    }

    // ---- 잠금 모드: 한 번 힘주면 유지, 다시 힘주면 풀림 ----
    int     cmdGesture = output;
    uint8_t cmdStrength = strengthOut;
    if (s_lockMode) {
      if (output != EMG2_REST_CLASS && lastOutput == EMG2_REST_CLASS) {       // 새로 힘주기 시작
        if (lockLatched == EMG2_REST_CLASS) {
          lockLatched = output; lockOwn = true; lockStrength = strengthOut;   // 잠금
        } else {
          lockLatched = EMG2_REST_CLASS; lockOwn = false;                     // 풀림
        }
      } else if (output != EMG2_REST_CLASS && lockOwn) {                      // 잠근 수축이 이어지는 동안
        lockLatched = output;                                                 // (주먹으로 바뀌면 따라감)
        if (strengthOut > lockStrength) lockStrength = strengthOut;           // 가장 세게 준 힘으로 유지
      } else if (output == EMG2_REST_CLASS) {
        lockOwn = false;
      }
      cmdGesture = lockLatched;
      cmdStrength = (lockLatched == EMG2_REST_CLASS) ? 0 : lockStrength;
    } else {
      lockLatched = EMG2_REST_CLASS;
      lockOwn = false;
    }
#endif

    // ---- 출력 ----
    const bool printNow = (output != lastOutput) || (frameNo % (1000 / (EMG2_HOP * 1000 / EMG2_FS)) == 0);
    if (printNow) {
      Serial.printf("[%7.2fs] 판단 %-6s %3.0f%% | 출력 %-6s | 세기 %4.1f배 %3d%% | %lu us",
                    frameNo * EMG2_HOP / (float)EMG2_FS, EMG2_CLASS_NAMES[pred], s_back.probs[pred] * 100.0f,
                    EMG2_CLASS_NAMES[output], ratio, (int)strengthOut, (unsigned long)dtUs);
#if RUN_MODE == MODE_REPLAY
      Serial.printf(" | 재생 중: %s", EMG2_CLASS_NAMES[EMG2_TEST_LABEL[s_replayVec]]);
#endif
#if RUN_MODE == MODE_RUN
      if (s_lockMode) Serial.printf(" | 잠금: %s", EMG2_CLASS_NAMES[lockLatched]);
#endif
      if (s_overruns) Serial.printf(" | 측정 지연 %lu회", (unsigned long)s_overruns);
      if (s_dropped) Serial.printf(" | 밀린 판단 %lu회", (unsigned long)s_dropped);
      Serial.println();
    }

    if (output != lastOutput) {
#if STRENGTH_REPORT && RUN_MODE == MODE_RUN
      if (lastOutput != EMG2_REST_CLASS && nEarly > 0) {   // 방금 끝난 동작의 세기 요약
        Serial.printf(">>> 세기 요약 %s: 처음 0.5초 평균 %.1f배 (100%% 기준 %.1f배)",
                      EMG2_CLASS_NAMES[lastOutput], sumEarly / nEarly, strengthFullFor(lastOutput));
        if (nLate > 0) Serial.printf(", 그 뒤 평균 %.1f배", sumLate / nLate);
        Serial.println();
      }
#endif
      sumEarly = sumLate = 0.0f;
      nEarly = nLate = 0;
      lastOutput = output;
    }
    if (output != EMG2_REST_CLASS) {
      if (nEarly < earlyN) { sumEarly += ratio; nEarly++; }
      else                 { sumLate += ratio;  nLate++; }
    }

#if RUN_MODE == MODE_RUN
    // 명령이 바뀌면 바로, 아니면 0.1초마다 현재 명령과 힘 세기를 보냄 (한 번 못 받아도 다음에 복구됨)
    const uint8_t cmd = handCmdFor(cmdGesture);
    if (cmd != lastSentCmd || frameNo % SEND_EVERY_N == 0) {
      sendHand(cmd, cmdStrength);
      lastSentCmd = cmd;
    }
#endif
  }
}
#endif

static void sampleOnce() {
  int v[REC_CH] = {0};
#if RUN_MODE == MODE_REPLAY
  for (int c = 0; c < EMG2_NUM_CH; ++c) v[EMG2_CHANNEL_IDX[c]] = EMG2_TEST_RAW[s_replayVec][c][s_replayIdx];
#elif RUN_MODE != MODE_SELFTEST
  for (int c = 0; c < REC_CH; ++c) v[c] = readEmgInput(c);
#endif
  s_count++;

#if RUN_MODE == MODE_RECORD
  // 버퍼에 넣기만 하고, 전송은 다른 코어가 담당 (측정이 멈추지 않도록)
  if (s_recHead - s_recTail < REC_BUF_SAMPLES) {
    const uint32_t idx = s_recHead % REC_BUF_SAMPLES;
    for (int c = 0; c < REC_CH; ++c) s_recBuf[idx][c] = (int16_t)v[c];
    s_recHead = s_recHead + 1;
  } else {
    s_recLost = s_recLost + 1;
  }
#endif

#if RUN_MODE == MODE_CHECK
  for (int c = 0; c < REC_CH; ++c) {
    s_sum[c] += v[c];
    s_sq[c] += (double)v[c] * v[c];
    s_rawSum[c] += s_lastRaw[c];
    // 클리핑은 필터 전 원본 값(0~4095) 기준으로 확인
    if (s_lastRaw[c] <= 5 || s_lastRaw[c] >= 4090) s_clip[c]++;
  }
  if (++s_statN >= (uint32_t)EMG2_FS) {
    Serial.printf("[%6.1fs] 원본평균", s_count / (float)EMG2_FS);
    for (int c = 0; c < REC_CH; ++c) Serial.printf(" %5.0f", s_rawSum[c] / s_statN);
    Serial.print(" | 흔들림(표준편차)");
    for (int c = 0; c < REC_CH; ++c) {
      double m = s_sum[c] / s_statN;
      double var = s_sq[c] / s_statN - m * m;
      Serial.printf(" %4.0f", var > 0 ? sqrt(var) : 0.0);
    }
    Serial.print(" | 클리핑");
    for (int c = 0; c < REC_CH; ++c) Serial.printf(" %3lu", (unsigned long)s_clip[c]);
    if (s_overruns) Serial.printf(" | 측정 지연 %lu회", (unsigned long)s_overruns);
    Serial.println();
    for (int c = 0; c < REC_CH; ++c) { s_sum[c] = 0.0; s_sq[c] = 0.0; s_rawSum[c] = 0.0; s_clip[c] = 0; }
    s_statN = 0;
  }
#endif

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
  float x[EMG2_NUM_CH];
  for (int c = 0; c < EMG2_NUM_CH; ++c) x[c] = (float)v[EMG2_CHANNEL_IDX[c]];
  Emg2Hop hop;
  if (emg2_front_push(&s_front, x, &hop)) {
    if (xQueueSend(s_hopQ, &hop, 0) != pdTRUE) s_dropped = s_dropped + 1;
  }
#if RUN_MODE == MODE_REPLAY
  if (++s_replayIdx >= (uint32_t)EMG2_TEST_LEN) {
    s_replayIdx = 0;
    s_replayVec = (s_replayVec + 1) % EMG2_TEST_COUNT;
  }
#endif
#endif
}

#if RUN_MODE == MODE_SELFTEST
// PC(train_v2.py)가 계산한 확률과 이 보드가 계산한 확률을 비교하고, 계산 시간을 잽니다.
static Emg2Front s_tFront;
static Emg2Back  s_tBack;

static void runSelfTest() {
  int pass = 0;
  uint32_t sampleUs = 0, frameUs = 0, frameMax = 0, frames = 0;
  emg2_back_set_gain(&s_tBack, NULL);
  for (int t = 0; t < EMG2_TEST_COUNT; ++t) {
    emg2_front_reset(&s_tFront);
    emg2_back_reset(&s_tBack);
    Emg2Hop hop;
    float x[EMG2_NUM_CH];
    for (int n = 0; n < EMG2_TEST_LEN; ++n) {
      for (int c = 0; c < EMG2_NUM_CH; ++c) x[c] = (float)EMG2_TEST_RAW[t][c][n];
      const uint32_t t0 = micros();
      const int ready = emg2_front_push(&s_tFront, x, &hop);
      const uint32_t t1 = micros();
      sampleUs += t1 - t0;
      if (ready) {
        emg2_back_frame(&s_tBack, &hop);
        const uint32_t d = micros() - t1;
        frameUs += d;
        if (d > frameMax) frameMax = d;
        frames++;
      }
    }
    float maxDiff = 0.0f;
    for (int k = 0; k < EMG2_NUM_CLASSES; ++k) {
      const float d = fabsf(s_tBack.probs[k] - EMG2_TEST_EXPECTED_PROB[t][k]);
      if (d > maxDiff) maxDiff = d;
    }
    const float sDiff = fabsf(s_tBack.strength - EMG2_TEST_EXPECTED_STRENGTH[t]);
    const bool ok = (s_tBack.pred == EMG2_TEST_EXPECTED_PRED[t]) && maxDiff < 0.02f &&
                    sDiff < 0.02f * (1.0f + EMG2_TEST_EXPECTED_STRENGTH[t]);
    pass += ok ? 1 : 0;
    Serial.printf("[%d] 정답 %-6s | PC %-6s | 보드 %-6s %5.1f%% | 확률 차이 %.4f | 세기 %.2f배 (PC %.2f) | %s\n", t,
                  EMG2_CLASS_NAMES[EMG2_TEST_LABEL[t]], EMG2_CLASS_NAMES[EMG2_TEST_EXPECTED_PRED[t]],
                  EMG2_CLASS_NAMES[s_tBack.pred], s_tBack.probs[s_tBack.pred] * 100.0f, maxDiff,
                  s_tBack.strength, EMG2_TEST_EXPECTED_STRENGTH[t], ok ? "통과" : "실패");
  }
  Serial.printf("===== %d / %d 통과 =====\n", pass, EMG2_TEST_COUNT);
  Serial.printf("샘플 1개 처리(필터+누적, 4채널): 평균 %.1f us  (여유: 1000 us마다 1번)\n",
                sampleUs / (float)(EMG2_TEST_COUNT * EMG2_TEST_LEN));
  Serial.printf("판단 1회(특징+TCN): 평균 %.0f us, 최대 %lu us  (여유: %d ms마다 1번)\n",
                frameUs / (float)frames, (unsigned long)frameMax, EMG2_HOP * 1000 / EMG2_FS);
}
#endif

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
  Serial.printf("ReHand 근전도 보드 v2 | 모델: %d프레임(%.3f초)을 보고 %dms마다 판단, 동작 %d가지\n", EMG2_RF_FRAMES,
                EMG2_RF_FRAMES * EMG2_HOP / (float)EMG2_FS, EMG2_HOP * 1000 / EMG2_FS, EMG2_NUM_CLASSES);
#if RUN_MODE == MODE_REPLAY
  Serial.println("===== MODE_REPLAY: 저장된 테스트 데이터를 실시간처럼 흘려보냅니다 =====");
#elif RUN_MODE == MODE_CHECK
  Serial.println("===== MODE_CHECK: 1초마다 채널별 상태를 출력합니다 =====");
  Serial.println("- 쉴 때 흔들림이 작고, 힘줄 때 해당 채널의 흔들림이 커지면 정상");
  Serial.println("- 원본평균이 4095나 0에 가깝거나 클리핑 숫자가 0보다 크면 센서 연결/전압 확인");
#elif RUN_MODE == MODE_SELFTEST
  Serial.println("===== MODE_SELFTEST: PC 계산 결과와 비교합니다 =====");
  runSelfTest();
#else
  Serial.println("===== MODE_RUN: 근전도 모드로 시작합니다 =====");
  Serial.println("- 시리얼 키: 1 모드 전환, 2 다음 동작/잠금, h 호밍, c 사용자 보정, x 보정 지우기, l 잠금 모드, ? 상태");
#endif
#endif

#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
#if ARDUINO_USB_CDC_ON_BOOT && ARDUINO_USB_MODE
  // 판단 모드에서는 시리얼 출력 때문에 판단이 늦어지지 않게, PC가 안 읽으면 출력을 버림
  Serial.setTxTimeoutMs(0);
#endif
  emg2_front_reset(&s_front);
  emg2_back_set_gain(&s_back, NULL);
  emg2_back_reset(&s_back);
  emg2_decide_reset(&s_dec);
  s_hopQ = xQueueCreate(16, sizeof(Emg2Hop));
#endif

#if RUN_MODE == MODE_RUN
  if (calLoad()) {
    emg2_back_set_gain(&s_back, s_cal.gain);
    Serial.print("저장된 사용자 보정 사용 (채널별 배율):");
    for (int c = 0; c < EMG2_NUM_CH; ++c) Serial.printf(" x%.2f", expf(s_cal.gain[c]));
    Serial.println();
  } else {
    Serial.println("사용자 보정 없음 (학습 데이터 기준). 전극을 다시 붙였으면 SW1 3초 또는 c 로 보정하세요.");
  }
#if USE_LINK
  rhEmgBegin();
#endif
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  xTaskCreatePinnedToCore(ctrlTaskFn, "emg_ctrl", 4096, NULL, 2, NULL, 0);
#endif
#if RUN_MODE == MODE_RUN || RUN_MODE == MODE_REPLAY
  xTaskCreatePinnedToCore(inferTaskFn, "emg_infer", 8192, NULL, 1, NULL, 0);
#endif

  s_nextUs = micros();
}

void loop() {
#if RUN_MODE == MODE_SELFTEST
  delay(1000);
#else
  const uint32_t now = micros();
  if ((int32_t)(now - s_nextUs) < 0) return;  // 아직 1ms가 안 됨
  s_nextUs += SAMPLE_US;
  if ((int32_t)(now - s_nextUs) > (int32_t)(200 * SAMPLE_US)) {
    s_nextUs = now + SAMPLE_US;  // 0.2초 넘게 밀리면 포기하고 다시 맞춤
    s_overruns++;
  }
  sampleOnce();
#endif
}
