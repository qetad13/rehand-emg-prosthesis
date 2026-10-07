// EMG 손가락 동작 판단 구현
// 계산 순서는 train_emg.py와 똑같습니다:
//   필터 -> 1초 창 채널별 정규화 -> STFT -> 주파수 묶기 -> log -> 정규화
//   -> Conv/ReLU/MaxPool x2 -> Conv/ReLU -> 평균 -> 전결합층 -> 확률
#define EMG_MODEL_DATA
#include "emg_infer.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------
// 작업용 버퍼 (스택이 작은 ESP32를 위해 전역으로 둠)
// ---------------------------------------------------------------
static float    s_hamming[EMG_WIN];
static float    s_cos[EMG_NFFT / 2];
static float    s_sin[EMG_NFFT / 2];
static uint16_t s_rev[EMG_NFFT];
static int      s_ready = 0;

static float s_norm[EMG_EPOCH_LEN];
static float s_re[EMG_NFFT];
static float s_im[EMG_NFFT];
static float s_buf_a[EMG_BUF_SIZE];
static float s_buf_b[EMG_BUF_SIZE];

static void init_tables(void) {
  const double two_pi = 6.283185307179586;
  for (int i = 0; i < EMG_WIN; ++i) {
    s_hamming[i] = (float)(0.54 - 0.46 * cos(two_pi * i / (EMG_WIN - 1)));
  }
  for (int k = 0; k < EMG_NFFT / 2; ++k) {
    s_cos[k] = (float)cos(two_pi * k / EMG_NFFT);
    s_sin[k] = (float)sin(two_pi * k / EMG_NFFT);
  }
  int bits = 0;
  while ((1 << bits) < EMG_NFFT) ++bits;
  for (int i = 0; i < EMG_NFFT; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    }
    s_rev[i] = (uint16_t)r;
  }
  s_ready = 1;
}

void emg_init(void) {
  if (!s_ready) init_tables();
}

// ---------------------------------------------------------------
// 필터 (Direct Form II Transposed, scipy sosfilt와 같은 식)
// ---------------------------------------------------------------
void emg_filter_reset(EmgFilter *f) {
  memset(f, 0, sizeof(*f));
}

float emg_filter_step(EmgFilter *f, float x) {
  for (int s = 0; s < EMG_NUM_SOS; ++s) {
    const float *c = &EMG_SOS[s * 6];  // b0 b1 b2 a0 a1 a2
    float y = c[0] * x + f->z1[s];
    f->z1[s] = c[1] * x - c[4] * y + f->z2[s];
    f->z2[s] = c[2] * x - c[5] * y;
    x = y;
  }
  return x;
}

// ---------------------------------------------------------------
// FFT (크기 EMG_NFFT, 2의 거듭제곱)
// ---------------------------------------------------------------
static void fft(float *re, float *im) {
  const int n = EMG_NFFT;
  for (int i = 0; i < n; ++i) {
    int j = s_rev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    const int step = n / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half; ++k) {
        const float wr = s_cos[k * step];
        const float wi = -s_sin[k * step];
        const int a = i + k;
        const int b = a + half;
        const float tr = wr * re[b] - wi * im[b];
        const float ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// ---------------------------------------------------------------
// 특징: (채널, 주파수 묶음, 시간 프레임)
// ---------------------------------------------------------------
void emg_features(const float *epoch, float *feat) {
  if (!s_ready) init_tables();

  for (int c = 0; c < EMG_NUM_CH; ++c) {
    const float *x = epoch + c * EMG_EPOCH_LEN;

#if EMG_ZSCORE_EPOCH
    double sum = 0.0;
    for (int n = 0; n < EMG_EPOCH_LEN; ++n) sum += x[n];
    const double mean = sum / EMG_EPOCH_LEN;
    double sq = 0.0;
    for (int n = 0; n < EMG_EPOCH_LEN; ++n) {
      const double d = x[n] - mean;
      sq += d * d;
    }
    double sd = sqrt(sq / EMG_EPOCH_LEN);
    if (sd < 1e-6) sd = 1e-6;
    for (int n = 0; n < EMG_EPOCH_LEN; ++n) s_norm[n] = (float)((x[n] - mean) / sd);
#else
    for (int n = 0; n < EMG_EPOCH_LEN; ++n) s_norm[n] = x[n];
#endif

    for (int t = 0; t < EMG_NUM_FRAMES; ++t) {
      const int off = t * EMG_HOP;
      for (int i = 0; i < EMG_NFFT; ++i) {
        s_re[i] = (i < EMG_WIN) ? s_norm[off + i] * s_hamming[i] : 0.0f;
        s_im[i] = 0.0f;
      }
      fft(s_re, s_im);
      for (int b = 0; b < EMG_NUM_BANDS; ++b) {
        float acc = 0.0f;
        for (int w = 0; w < EMG_BAND_WIDTH; ++w) {
          const int k = EMG_BAND_START + b * EMG_BAND_WIDTH + w;
          acc += s_re[k] * s_re[k] + s_im[k] * s_im[k];
        }
        const float p = acc / (float)EMG_BAND_WIDTH;
        const float v = (float)(10.0 * log10((double)(p + 1e-10f)));
        feat[(c * EMG_NUM_BANDS + b) * EMG_NUM_FRAMES + t] =
            (v - EMG_FEAT_MEAN[c]) / EMG_FEAT_STD[c];
      }
    }
  }
}

// ---------------------------------------------------------------
// CNN 연산
// ---------------------------------------------------------------
static void conv3x3_relu(const float *in, int cin, int h, int w,
                         const float *weight, const float *bias, int cout, float *out) {
  for (int o = 0; o < cout; ++o) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float acc = bias[o];
        for (int i = 0; i < cin; ++i) {
          const float *k = weight + (o * cin + i) * 9;
          const float *p = in + i * h * w;
          for (int dy = -1; dy <= 1; ++dy) {
            const int yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int xx = x + dx;
              if (xx < 0 || xx >= w) continue;
              acc += k[(dy + 1) * 3 + (dx + 1)] * p[yy * w + xx];
            }
          }
        }
        out[(o * h + y) * w + x] = acc > 0.0f ? acc : 0.0f;
      }
    }
  }
}

static void maxpool2(const float *in, int c, int h, int w, float *out) {
  const int h2 = h / 2;
  const int w2 = w / 2;
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h2; ++y) {
      for (int x = 0; x < w2; ++x) {
        const float *r0 = in + (ch * h + 2 * y) * w + 2 * x;
        const float *r1 = r0 + w;
        float m = r0[0];
        if (r0[1] > m) m = r0[1];
        if (r1[0] > m) m = r1[0];
        if (r1[1] > m) m = r1[1];
        out[(ch * h2 + y) * w2 + x] = m;
      }
    }
  }
}

int emg_classify(const float *epoch, float *probs) {
  if (!s_ready) init_tables();

  const int h0 = EMG_NUM_BANDS, w0 = EMG_NUM_FRAMES;
  const int h1 = h0 / 2, w1 = w0 / 2;
  const int h2 = h1 / 2, w2 = w1 / 2;

  emg_features(epoch, s_buf_a);
  conv3x3_relu(s_buf_a, EMG_NUM_CH, h0, w0, EMG_CONV1_W, EMG_CONV1_B, EMG_C1, s_buf_b);
  maxpool2(s_buf_b, EMG_C1, h0, w0, s_buf_a);
  conv3x3_relu(s_buf_a, EMG_C1, h1, w1, EMG_CONV2_W, EMG_CONV2_B, EMG_C2, s_buf_b);
  maxpool2(s_buf_b, EMG_C2, h1, w1, s_buf_a);
  conv3x3_relu(s_buf_a, EMG_C2, h2, w2, EMG_CONV3_W, EMG_CONV3_B, EMG_C3, s_buf_b);

  float gap[EMG_C3];
  for (int c = 0; c < EMG_C3; ++c) {
    float acc = 0.0f;
    const float *p = s_buf_b + c * h2 * w2;
    for (int i = 0; i < h2 * w2; ++i) acc += p[i];
    gap[c] = acc / (float)(h2 * w2);
  }

  float logits[EMG_NUM_CLASSES];
  int best = 0;
  for (int k = 0; k < EMG_NUM_CLASSES; ++k) {
    float acc = EMG_FC_B[k];
    for (int j = 0; j < EMG_C3; ++j) acc += EMG_FC_W[k * EMG_C3 + j] * gap[j];
    logits[k] = acc;
    if (acc > logits[best]) best = k;
  }

  if (probs) {
    float sum = 0.0f;
    for (int k = 0; k < EMG_NUM_CLASSES; ++k) {
      probs[k] = (float)exp((double)(logits[k] - logits[best]));
      sum += probs[k];
    }
    for (int k = 0; k < EMG_NUM_CLASSES; ++k) probs[k] /= sum;
  }
  return best;
}

// ---------------------------------------------------------------
// 휴식 게이트 (train_emg.py의 rest_gate_mask와 같음)
// ---------------------------------------------------------------
int emg_rest_gate(const float *epoch, int pred) {
  if (EMG_REST_GATE <= 0.0f) return pred;
  for (int c = 0; c < EMG_NUM_CH; ++c) {
    const float *x = epoch + c * EMG_EPOCH_LEN;
    float acc = 0.0f;
    for (int n = 0; n < EMG_EPOCH_LEN; ++n) acc += x[n] * x[n];
    const float rms = sqrtf(acc / (float)EMG_EPOCH_LEN);
    if (rms >= EMG_REST_GATE * EMG_REST_RMS[c]) return pred;   // 한 채널이라도 크면 통과
  }
  return EMG_REST_CLASS;
}

// ---------------------------------------------------------------
// 힘 세기 (휴식 게이트와 같은 계산, 다만 최근 n_last개 샘플만 사용)
// ---------------------------------------------------------------
float emg_strength(const float *epoch, int n_last) {
  if (n_last <= 0 || n_last > EMG_EPOCH_LEN) n_last = EMG_EPOCH_LEN;
  float best = 0.0f;
  for (int c = 0; c < EMG_NUM_CH; ++c) {
    if (EMG_REST_RMS[c] <= 0.0f) continue;
    const float *x = epoch + c * EMG_EPOCH_LEN + (EMG_EPOCH_LEN - n_last);
    float acc = 0.0f;
    for (int n = 0; n < n_last; ++n) acc += x[n] * x[n];
    const float ratio = sqrtf(acc / (float)n_last) / EMG_REST_RMS[c];
    if (ratio > best) best = ratio;
  }
  return best;
}

// ---------------------------------------------------------------
// 최종 출력 규칙 (train_emg.py의 vote_sequence와 같음)
// ---------------------------------------------------------------
void emg_vote_reset(EmgVote *v) {
  memset(v, 0, sizeof(*v));
  v->output = EMG_REST_CLASS;
  v->since = 1L << 30;
}

int emg_vote_update(EmgVote *v, int pred) {
  v->hist[v->pos] = pred;
  v->pos = (v->pos + 1) % EMG_HIST_N;
  if (v->count < EMG_HIST_N) v->count++;
  v->since++;

  int cand = v->output;
  if (pred == EMG_REST_CLASS) {
    v->rest_run++;
    if (v->rest_run >= EMG_VOTE_RELEASE_N) cand = EMG_REST_CLASS;   // 끄기
  } else {
    v->rest_run = 0;
    const int resting = (v->output == EMG_REST_CLASS);
    const int n    = resting ? EMG_VOTE_N : EMG_VOTE_SWITCH_K;
    const int need = resting ? EMG_VOTE_K : EMG_VOTE_SWITCH_K;
    const int m    = v->count < n ? v->count : n;

    int counts[EMG_NUM_CLASSES];
    memset(counts, 0, sizeof(counts));
    for (int i = 0; i < m; ++i) {
      const int idx = (v->pos - 1 - i + EMG_HIST_N) % EMG_HIST_N;   // 최근 것부터
      counts[v->hist[idx]]++;
    }
    int best = 0;
    for (int k = 1; k < EMG_NUM_CLASSES; ++k) {
      if (counts[k] > counts[best]) best = k;
    }
    if (counts[best] >= need && best != EMG_REST_CLASS && best != v->output) cand = best;
  }

  if (cand != v->output && v->since >= EMG_VOTE_HOLD_N) {   // 바뀐 지 얼마 안 됐으면 유지
    v->output = cand;
    v->since = 0;
  }
  return v->output;
}
