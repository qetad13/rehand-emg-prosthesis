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

[[maybe_unused]] static void sendHandCommand(uint8_t cmd) {
  if (!s_espnowReady) return;
  esp_err_t r = esp_now_send(HAND_MAC, &cmd, 1);
  if (r != ESP_OK) Serial.printf("[ESP-NOW] 전송 실패 (%d)\n", (int)r);
}

// 판단 결과가 바뀌었을 때 호출됨
[[maybe_unused]] static void onOutputChanged(int gesture) {
  Serial.printf(">>> 출력 변경: %s (의수 명령 %d)\n", EMG_CLASS_NAMES[gesture],
                handCmdFor(gesture));
}

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

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint32_t t0 = micros();
    int pred = emg_classify(s_epoch, probs);
    pred = emg_rest_gate(s_epoch, pred);        // 신호가 휴식 수준이면 무조건 휴식
    int output = emg_vote_update(&vote, pred);
    uint32_t dtMs = (micros() - t0) / 1000UL;

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
      lastOutput = output;
    }
#if RUN_MODE == MODE_RUN && USE_ESPNOW
    // 0.5초마다 현재 명령을 보냄 (바뀌면 바로 반영되고, 한 번 못 받아도 다음에 복구됨)
    sendHandCommand(handCmdFor(output));
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
  Serial.println("===== MODE_RUN: 0.5초마다 판단합니다 =====");
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
