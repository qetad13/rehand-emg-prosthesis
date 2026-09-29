// ESP32 정답 맞추기 테스트
// PC(PyTorch)에서 계산한 답과 ESP32에서 C로 계산한 답이 같은지 확인합니다.
// 센서 연결은 필요 없습니다. 보드만 USB로 연결하고 업로드 -> 시리얼 모니터(115200)
#include "emg_infer.h"
#include "emg_test_vectors.h"

static EmgFilter s_filters[EMG_NUM_CH];
static float s_epoch[EMG_NUM_CH * EMG_EPOCH_LEN];
static float s_probs[EMG_NUM_CLASSES];

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println();
  Serial.println("===== EMG golden test =====");

  int passed = 0;
  unsigned long total_us = 0;

  for (int v = 0; v < EMG_TEST_COUNT; ++v) {
    // 1) 필터를 0부터 시작해서 2초 전체에 적용하고, 마지막 1초를 판단에 사용
    for (int c = 0; c < EMG_NUM_CH; ++c) emg_filter_reset(&s_filters[c]);
    const int skip = EMG_TEST_LEN - EMG_EPOCH_LEN;
    for (int n = 0; n < EMG_TEST_LEN; ++n) {
      for (int c = 0; c < EMG_NUM_CH; ++c) {
        float y = emg_filter_step(&s_filters[c], (float)EMG_TEST_RAW[v][c][n]);
        if (n >= skip) s_epoch[c * EMG_EPOCH_LEN + (n - skip)] = y;
      }
    }

    // 2) 판단 + 시간 측정
    unsigned long t0 = micros();
    int pred = emg_classify(s_epoch, s_probs);
    unsigned long dt = micros() - t0;
    total_us += dt;

    // 3) PC 결과와 비교
    float max_diff = 0.0f;
    for (int k = 0; k < EMG_NUM_CLASSES; ++k) {
      float d = fabs(s_probs[k] - EMG_TEST_EXPECTED_PROB[v][k]);
      if (d > max_diff) max_diff = d;
    }
    bool ok = (pred == EMG_TEST_EXPECTED_PRED[v]) && (max_diff < 0.01f);
    if (ok) passed++;

    Serial.printf("[%d] 정답 %-6s | ESP32 %-6s (%.1f%%) | PC와 확률 차이 %.5f | %lu ms | %s\n",
                  v, EMG_CLASS_NAMES[EMG_TEST_LABEL[v]], EMG_CLASS_NAMES[pred],
                  s_probs[pred] * 100.0f, max_diff, dt / 1000UL, ok ? "OK" : "FAIL");
  }

  Serial.printf("\n결과: %d / %d 통과, 판단 1번 평균 %lu ms\n",
                passed, EMG_TEST_COUNT, total_us / EMG_TEST_COUNT / 1000UL);
  if (passed == EMG_TEST_COUNT) {
    Serial.println("-> PC와 ESP32 계산이 일치합니다. 실시간 연결 단계로 넘어가면 됩니다.");
  } else {
    Serial.println("-> 불일치가 있습니다. 시리얼 출력 전체를 캡처해서 보내주세요.");
  }
}

void loop() {
  delay(1000);
}
