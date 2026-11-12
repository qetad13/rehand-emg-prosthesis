// EMG 손가락 동작 판단 v2 구현
// 계산 순서는 train_v2.py와 똑같습니다:
//   필터 -> 25ms 누적 -> 특징(로그 진폭, 주파수 비, 영교차, 채널 상관) -> 정규화
//   -> 1D 시간 합성곱(TCN) x N층 -> 전결합층 -> 확률
#define EMG2_MODEL_DATA
#include "emg2.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------
// 1) 샘플 단계: 필터(Direct Form II Transposed, scipy sosfilt와 같은 식) + 누적
// ---------------------------------------------------------------
void emg2_front_reset(Emg2Front *f) {
  memset(f, 0, sizeof(*f));
}

int emg2_front_push(Emg2Front *f, const float *x, Emg2Hop *out) {
  float y[EMG2_NUM_CH];
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    float v = x[c];
    for (int s = 0; s < EMG2_NUM_SOS; ++s) {
      const float *k = &EMG2_SOS[s * 6];       // b0 b1 b2 a0 a1 a2
      const float o = k[0] * v + f->z1[c][s];
      f->z1[c][s] = k[1] * v - k[4] * o + f->z2[c][s];
      f->z2[c][s] = k[2] * v - k[5] * o;
      v = o;
    }
    y[c] = v;
    const float d1 = v - f->prev[c];
    const float d2 = d1 - f->prev_d1[c];
    f->acc.s0[c] += v * v;
    f->acc.s1[c] += d1 * d1;
    f->acc.s2[c] += d2 * d2;
    if (v * f->prev[c] < 0.0f) f->acc.zc[c] += 1.0f;
    f->prev[c] = v;
    f->prev_d1[c] = d1;
  }
  int p = 0;
  for (int a = 0; a < EMG2_NUM_CH; ++a) {
    for (int b = a + 1; b < EMG2_NUM_CH; ++b) f->acc.cr[p++] += y[a] * y[b];
  }
  if (++f->n < EMG2_HOP) return 0;
  *out = f->acc;
  memset(&f->acc, 0, sizeof(f->acc));
  f->n = 0;
  return 1;
}

// ---------------------------------------------------------------
// 2) 프레임 단계
// ---------------------------------------------------------------
void emg2_back_set_gain(Emg2Back *b, const float *gain_off) {
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    b->gain_off[c] = gain_off ? gain_off[c] : 0.0f;
    b->gain_inv[c] = expf(-b->gain_off[c]);
  }
}

void emg2_back_reset(Emg2Back *b) {
  float keep[EMG2_NUM_CH];
  memcpy(keep, b->gain_off, sizeof(keep));
  memset(b, 0, sizeof(*b));
  emg2_back_set_gain(b, keep);
  b->pred = EMG2_REST_CLASS;
}

int emg2_back_frame(Emg2Back *b, const Emg2Hop *hop) {
  // ---- 최근 EMG2_K개 hop을 더함 ----
  b->hops[b->hop_pos] = *hop;
  b->hop_pos = (b->hop_pos + 1) % EMG2_K;
  Emg2Hop s;
  memset(&s, 0, sizeof(s));
  for (int k = 0; k < EMG2_K; ++k) {
    const Emg2Hop *h = &b->hops[k];
    for (int c = 0; c < EMG2_NUM_CH; ++c) {
      s.s0[c] += h->s0[c]; s.s1[c] += h->s1[c]; s.s2[c] += h->s2[c]; s.zc[c] += h->zc[c];
    }
    for (int p = 0; p < EMG2_NUM_PAIRS; ++p) s.cr[p] += h->cr[p];
  }

  // ---- 특징 (순서: 로그 진폭, 주파수 비 1, 주파수 비 2, 영교차, 채널 상관) ----
  const float N = (float)(EMG2_K * EMG2_HOP);
  const float eps = 1e-3f * N;
  float feat[EMG2_NUM_FEAT];
  int n = 0;
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    b->logamp[c] = 0.5f * logf((s.s0[c] / N + 1e-3f) / EMG2_REST_POW[c]);
    feat[n++] = b->logamp[c] - b->gain_off[c];
  }
#if EMG2_FEAT_F1
  for (int c = 0; c < EMG2_NUM_CH; ++c) feat[n++] = 0.5f * logf((s.s1[c] + eps) / (s.s0[c] + eps));
#endif
#if EMG2_FEAT_F2
  for (int c = 0; c < EMG2_NUM_CH; ++c) feat[n++] = 0.5f * logf((s.s2[c] + eps) / (s.s1[c] + eps));
#endif
#if EMG2_FEAT_ZC
  for (int c = 0; c < EMG2_NUM_CH; ++c) feat[n++] = s.zc[c] / N;
#endif
#if EMG2_FEAT_CR
  {
    int p = 0;
    for (int a = 0; a < EMG2_NUM_CH; ++a) {
      for (int c = a + 1; c < EMG2_NUM_CH; ++c) {
        feat[n++] = s.cr[p++] / sqrtf(s.s0[a] * s.s0[c] + eps * eps);
      }
    }
  }
#endif
  for (int i = 0; i < EMG2_NUM_FEAT; ++i) feat[i] = (feat[i] - EMG2_FEAT_MEAN[i]) / EMG2_FEAT_STD[i];

  // ---- TCN: 층마다 과거 입력을 링 버퍼에 두고, 이번 시점의 출력 하나만 계산 ----
  float buf_a[EMG2_MAX_WIDTH], buf_b[EMG2_MAX_WIDTH];
  const float *x = feat;
  float *y = buf_a;
  float *ring = b->hist;
  for (int l = 0; l < EMG2_NUM_LAYERS; ++l) {
    const Emg2LayerDef *L = &EMG2_LAYERS[l];
    const int pos = b->pos[l];
    memcpy(ring + pos * L->in, x, (size_t)L->in * sizeof(float));
    for (int o = 0; o < L->out; ++o) {
      float acc = L->b[o];
      for (int j = 0; j < L->k; ++j) {
        // 탭 j는 (k-1-j)*dil 프레임 전의 입력
        int idx = pos - (L->k - 1 - j) * L->dil;
        if (idx < 0) idx += L->hist;
        const float *xp = ring + idx * L->in;
        const float *w = L->w + ((size_t)o * L->k + j) * L->in;   // 가중치 순서: [출력][탭][입력]
        for (int i = 0; i < L->in; ++i) acc += w[i] * xp[i];
      }
      y[o] = acc > 0.0f ? acc : 0.0f;
    }
    b->pos[l] = (pos + 1) % L->hist;
    ring += L->hist * L->in;
    x = y;
    y = (y == buf_a) ? buf_b : buf_a;
  }

  // ---- 전결합층 + 확률 ----
  float logits[EMG2_NUM_CLASSES];
  int best = 0;
  for (int k = 0; k < EMG2_NUM_CLASSES; ++k) {
    float acc = EMG2_FC_B[k];
    const float *w = EMG2_FC_W + k * EMG2_EMBED;
    for (int j = 0; j < EMG2_EMBED; ++j) acc += w[j] * x[j];
    logits[k] = acc;
    if (acc > logits[best]) best = k;
  }
  float sum = 0.0f;
  for (int k = 0; k < EMG2_NUM_CLASSES; ++k) {
    b->probs[k] = expf(logits[k] - logits[best]);
    sum += b->probs[k];
  }
  for (int k = 0; k < EMG2_NUM_CLASSES; ++k) b->probs[k] /= sum;
  b->pred = best;

  // ---- 힘 세기: 최근 EMG2_STRENGTH_HOPS개 hop의 신호 크기 / 쉴 때 크기, 채널 중 최댓값 ----
  memcpy(b->pow_hist[b->pow_pos], hop->s0, sizeof(hop->s0));
  b->pow_pos = (b->pow_pos + 1) % EMG2_STRENGTH_HOPS;
  float strongest = 0.0f;
  for (int c = 0; c < EMG2_NUM_CH; ++c) {
    float acc = 0.0f;
    for (int h = 0; h < EMG2_STRENGTH_HOPS; ++h) acc += b->pow_hist[h][c];
    const float r = sqrtf(acc / (float)(EMG2_STRENGTH_HOPS * EMG2_HOP) / EMG2_REST_POW[c]) * b->gain_inv[c];
    if (r > strongest) strongest = r;
  }
  b->strength = strongest;
  b->frames++;
  return best;
}

// ---------------------------------------------------------------
// 3) 최종 출력 규칙
// ---------------------------------------------------------------
void emg2_decide_reset(Emg2Decider *d) {
  memset(d, 0, sizeof(*d));
  d->ps[EMG2_REST_CLASS] = 1.0f;
  d->output = EMG2_REST_CLASS;
  d->since = 1L << 30;
  d->sw_c = -1;
}

static void decide_set_rest(float *p) {
  for (int k = 0; k < EMG2_NUM_CLASSES; ++k) p[k] = 0.0f;
  p[EMG2_REST_CLASS] = 1.0f;
}

int emg2_decide(Emg2Decider *d, const float *probs, float strength) {
  float p[EMG2_NUM_CLASSES];
  memcpy(p, probs, sizeof(p));
  const int forced_rest = (d->frames < EMG2_WARMUP_FRAMES) ||
                          (EMG2_REST_GATE > 0.0f && strength < EMG2_REST_GATE);
  if (forced_rest) decide_set_rest(p);
  d->frames++;

  int k = 0;                                   // 이번 판단 (평활 전)
  for (int i = 1; i < EMG2_NUM_CLASSES; ++i) if (p[i] > p[k]) k = i;
  int c = -1;                                  // 평활한 확률이 가장 큰 동작 (휴식 제외)
  for (int i = 0; i < EMG2_NUM_CLASSES; ++i) {
    d->ps[i] = EMG2_A_ON * d->ps[i] + (1.0f - EMG2_A_ON) * p[i];
    if (i != EMG2_REST_CLASS && (c < 0 || d->ps[i] > d->ps[c])) c = i;
  }
  if (d->since < (1L << 30)) d->since++;
  d->rest_run = (k == EMG2_REST_CLASS) ? d->rest_run + 1 : 0;

  int cand = d->output;
  if (d->output != EMG2_REST_CLASS && d->rest_run >= EMG2_REL_N) {
    cand = EMG2_REST_CLASS;                                          // 끄기
  } else if (d->output == EMG2_REST_CLASS) {
    if (d->ps[c] >= EMG2_TH_ON && k != EMG2_REST_CLASS) cand = c;    // 켜기
  } else if (c != d->output && d->ps[c] >= EMG2_TH_SW && ((EMG2_SWITCH_FREE_MASK >> c) & 1)) {
    d->sw_run = (d->sw_c == c) ? d->sw_run + 1 : 1;                  // 휴식 없이 바꾸기 (주먹)
    d->sw_c = c;
    if (d->sw_run >= EMG2_SW_N) cand = c;
  } else {
    d->sw_run = 0;
  }

  if (cand != d->output &&
      (d->since >= EMG2_HOLD_N || (cand != EMG2_REST_CLASS && d->output != EMG2_REST_CLASS))) {
    if (cand == EMG2_REST_CLASS) decide_set_rest(d->ps);             // 끈 뒤에는 증거를 처음부터 다시 모음
    d->output = cand;
    d->since = 0;
    d->sw_run = 0;
  }
  return d->output;
}
