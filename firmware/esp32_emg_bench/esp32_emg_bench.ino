// =====================================================================
// 판단 속도 측정 (센서 없이 실행)
//   1) 필터            : 1초치(4채널 1000샘플) 필터 계산
//   2) STFT 특징       : 정규화 + FFT + 주파수 묶기
//   3) CNN             : 전체 판단 시간에서 특징 시간을 뺀 값
//   4) 시간영역 방식   : 특징(MAV/RMS/파형길이/영교차/기울기) + 작은 신경망
//                        (속도 비교용이라 가중치는 임의값)
// =====================================================================
#include "emg_infer.h"
#include "emg_test_vectors.h"

static float s_epoch[EMG_NUM_CH * EMG_EPOCH_LEN];
static float s_feat[EMG_NUM_CH * EMG_NUM_BANDS * EMG_NUM_FRAMES];
static float s_probs[EMG_NUM_CLASSES];

// ---- 시간영역 방식 (비교용) ----
#define TD_SUB    4                       // 1초를 4구간으로 나눔
#define TD_IN     (EMG_NUM_CH * TD_SUB * 5)  // 채널 x 구간 x 특징 5개
#define TD_HIDDEN 48
static float td_feat[TD_IN];
static float td_w1[TD_IN * TD_HIDDEN], td_b1[TD_HIDDEN];
static float td_w2[TD_HIDDEN * TD_HIDDEN], td_b2[TD_HIDDEN];
static float td_w3[TD_HIDDEN * EMG_NUM_CLASSES], td_b3[EMG_NUM_CLASSES];
static float td_h1[TD_HIDDEN], td_h2[TD_HIDDEN];

static uint32_t rnd = 1;
static float frand() { rnd = rnd * 1664525UL + 1013904223UL; return ((rnd >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static void td_init() {
  for (int i = 0; i < TD_IN * TD_HIDDEN; ++i) td_w1[i] = frand() * 0.1f;
  for (int i = 0; i < TD_HIDDEN; ++i) td_b1[i] = 0.0f;
  for (int i = 0; i < TD_HIDDEN * TD_HIDDEN; ++i) td_w2[i] = frand() * 0.1f;
  for (int i = 0; i < TD_HIDDEN; ++i) td_b2[i] = 0.0f;
  for (int i = 0; i < TD_HIDDEN * EMG_NUM_CLASSES; ++i) td_w3[i] = frand() * 0.1f;
  for (int i = 0; i < EMG_NUM_CLASSES; ++i) td_b3[i] = 0.0f;
}

static void td_features(const float *epoch) {
  const int sub = EMG_EPOCH_LEN / TD_SUB;
  int k = 0;
  for (int c = 0; c < EMG_NUM_CH; ++c) {
    const float *x = epoch + c * EMG_EPOCH_LEN;
    for (int s = 0; s < TD_SUB; ++s) {
      const float *p = x + s * sub;
      float mav = 0, sq = 0, wl = 0;
      int zc = 0, ssc = 0;
      for (int n = 0; n < sub; ++n) { mav += fabsf(p[n]); sq += p[n] * p[n]; }
      mav /= sub;
      float rms = sqrtf(sq / sub);
      float th = 0.15f * rms;
      for (int n = 1; n < sub; ++n) {
        float d = p[n] - p[n - 1];
        wl += fabsf(d);
        if (p[n] * p[n - 1] < 0 && fabsf(d) > th) zc++;
        if (n > 1 && (p[n] - p[n - 1]) * (p[n - 1] - p[n - 2]) < 0 && fabsf(d) > th) ssc++;
      }
      wl /= (sub - 1);
      td_feat[k++] = logf(mav + 1e-3f);
      td_feat[k++] = logf(rms + 1e-3f);
      td_feat[k++] = logf(wl + 1e-3f);
      td_feat[k++] = (float)zc / sub;
      td_feat[k++] = (float)ssc / sub;
    }
  }
}

static int td_classify(const float *epoch) {
  td_features(epoch);
  for (int j = 0; j < TD_HIDDEN; ++j) {
    float a = td_b1[j];
    for (int i = 0; i < TD_IN; ++i) a += td_w1[j * TD_IN + i] * td_feat[i];
    td_h1[j] = a > 0 ? a : 0;
  }
  for (int j = 0; j < TD_HIDDEN; ++j) {
    float a = td_b2[j];
    for (int i = 0; i < TD_HIDDEN; ++i) a += td_w2[j * TD_HIDDEN + i] * td_h1[i];
    td_h2[j] = a > 0 ? a : 0;
  }
  int best = 0;
  float bestv = -1e30f;
  for (int j = 0; j < EMG_NUM_CLASSES; ++j) {
    float a = td_b3[j];
    for (int i = 0; i < TD_HIDDEN; ++i) a += td_w3[j * TD_HIDDEN + i] * td_h2[i];
    if (a > bestv) { bestv = a; best = j; }
  }
  return best;
}

static const int REPEAT = 20;

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println();
  Serial.println("===== 판단 속도 측정 =====");
  td_init();

  // 테스트 데이터 1초치를 필터에 통과시켜 준비
  EmgFilter f[EMG_NUM_CH];
  for (int c = 0; c < EMG_NUM_CH; ++c) emg_filter_reset(&f[c]);
  for (int n = 0; n < EMG_TEST_LEN; ++n) {
    for (int c = 0; c < EMG_NUM_CH; ++c) {
      float y = emg_filter_step(&f[c], (float)EMG_TEST_RAW[4][c][n]);
      int m = n - (EMG_TEST_LEN - EMG_EPOCH_LEN);
      if (m >= 0) s_epoch[c * EMG_EPOCH_LEN + m] = y;
    }
  }

  uint32_t t0 = micros();
  for (int r = 0; r < REPEAT; ++r) {
    for (int c = 0; c < EMG_NUM_CH; ++c) emg_filter_reset(&f[c]);
    for (int n = 0; n < EMG_EPOCH_LEN; ++n)
      for (int c = 0; c < EMG_NUM_CH; ++c) emg_filter_step(&f[c], (float)EMG_TEST_RAW[4][c][n]);
  }
  float tFilter = (micros() - t0) / 1000.0f / REPEAT;

  t0 = micros();
  for (int r = 0; r < REPEAT; ++r) emg_features(s_epoch, s_feat);
  float tFeat = (micros() - t0) / 1000.0f / REPEAT;

  t0 = micros();
  for (int r = 0; r < REPEAT; ++r) emg_classify(s_epoch, s_probs);
  float tAll = (micros() - t0) / 1000.0f / REPEAT;

  t0 = micros();
  for (int r = 0; r < REPEAT; ++r) td_classify(s_epoch);
  float tTd = (micros() - t0) / 1000.0f / REPEAT;

  Serial.printf("필터 (1초치, 1000샘플 x 4채널) : %7.2f ms\n", tFilter);
  Serial.printf("STFT 특징 계산                 : %7.2f ms\n", tFeat);
  Serial.printf("CNN 계산                       : %7.2f ms\n", tAll - tFeat);
  Serial.printf("합계 (지금 방식, 판단 1번)     : %7.2f ms\n", tAll);
  Serial.printf("시간영역 특징 + 작은 신경망     : %7.2f ms\n", tTd);
  Serial.printf("\n0.5초(500ms)마다 판단하므로 여유: 지금 방식 %.0f%%, 시간영역 %.0f%% 사용\n",
                100.0f * tAll / 500.0f, 100.0f * tTd / 500.0f);
}

void loop() { delay(1000); }
