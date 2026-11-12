// =====================================================================
// EMG 손가락 동작 판단 v2 (스트리밍 방식, 라이브러리 없이 C로 직접 계산)
// train_v2.py가 만든 emg2_model.h의 가중치를 사용합니다.
//
// 예전 방식(v1): 0.75초 창을 통째로 STFT -> 2D CNN, 0.1초마다 판단 (1회 약 30만 곱셈)
// 새 방식(v2)  : 샘플이 들어올 때마다 누적 -> 25ms마다 특징 22개 -> 작은 1D 시간 합성곱(TCN)
//                25ms마다 판단 (1회 약 6천 곱셈), 창 전체를 저장하지 않음
//
// 계산은 세 단계로 나뉘어 있고, 단계마다 다른 코어/태스크에서 불러도 됩니다.
//   1) emg2_front_push : 샘플 1개(4채널)를 넣음. 필터 + 누적. 25ms마다 Emg2Hop 하나가 나옴  (1kHz 측정 루프)
//   2) emg2_back_frame : Emg2Hop -> 특징 -> TCN -> 클래스별 확률, 힘 세기                    (판단 태스크)
//   3) emg2_decide     : 확률 -> 최종 출력 (켜기/끄기/바꾸기 규칙)                           (판단 태스크)
// =====================================================================
#ifndef EMG2_H
#define EMG2_H

#include <stdint.h>
#include "emg2_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------
// 1) 샘플 단계
// ---------------------------------------------------------------
typedef struct {              // 25ms(EMG2_HOP 샘플) 동안의 누적값
  float s0[EMG2_NUM_CH];      // 신호 제곱합            (크기)
  float s1[EMG2_NUM_CH];      // 1차 차분 제곱합        (주파수 정보 1)
  float s2[EMG2_NUM_CH];      // 2차 차분 제곱합        (주파수 정보 2)
  float zc[EMG2_NUM_CH];      // 부호가 바뀐 횟수
  float cr[EMG2_NUM_PAIRS];   // 채널 쌍의 곱의 합      (채널 사이 관계)
} Emg2Hop;

typedef struct {
  float   z1[EMG2_NUM_CH][EMG2_NUM_SOS];
  float   z2[EMG2_NUM_CH][EMG2_NUM_SOS];
  float   prev[EMG2_NUM_CH];      // 직전 샘플
  float   prev_d1[EMG2_NUM_CH];   // 직전 1차 차분
  Emg2Hop acc;
  int     n;
} Emg2Front;

void emg2_front_reset(Emg2Front *f);
// x: 채널별 입력 값 EMG2_NUM_CH개 (모델이 쓰는 채널 순서). Emg2Hop이 완성되면 1을 돌려주고 out에 채움
int  emg2_front_push(Emg2Front *f, const float *x, Emg2Hop *out);

// ---------------------------------------------------------------
// 2) 프레임 단계
// ---------------------------------------------------------------
typedef struct {
  Emg2Hop  hops[EMG2_K];                               // 최근 hop들 (특징은 최근 EMG2_K개 hop의 합으로 계산)
  int      hop_pos;
  float    pow_hist[EMG2_STRENGTH_HOPS][EMG2_NUM_CH];  // 힘 세기용 최근 신호 제곱합
  int      pow_pos;
  float    hist[EMG2_HIST_FLOATS];                     // TCN 층별 과거 입력
  int      pos[EMG2_NUM_LAYERS];
  float    gain_off[EMG2_NUM_CH];                      // 사용자 보정: 채널별 로그 진폭 보정값 (기본 0)
  float    gain_inv[EMG2_NUM_CH];                      // exp(-gain_off)
  uint32_t frames;

  // ---- 결과 (emg2_back_frame이 채움) ----
  float    probs[EMG2_NUM_CLASSES];   // 클래스별 확률
  int      pred;                      // 확률이 가장 큰 클래스
  float    strength;                  // 힘 세기: 최근 0.2초 신호 크기가 쉴 때의 몇 배인지 (채널 중 최댓값, 보정 반영)
  float    logamp[EMG2_NUM_CH];       // 채널별 로그 진폭 (쉴 때 = 0, 보정 전 값. 사용자 보정에서 사용)
} Emg2Back;

void emg2_back_reset(Emg2Back *b);                         // 보정값(gain_off)은 그대로 둠
void emg2_back_set_gain(Emg2Back *b, const float *gain_off);  // NULL이면 보정 없음(0)
int  emg2_back_frame(Emg2Back *b, const Emg2Hop *hop);     // pred를 돌려줌

// ---------------------------------------------------------------
// 3) 최종 출력 규칙 (train_v2.py의 decide_sequence와 같음)
//    켜기   : 휴식 상태에서, 평활한 확률이 EMG2_TH_ON 이상인 동작이 있으면 켬
//    끄기   : 휴식 판단이 EMG2_REL_N번 연속이면 끔 (켠 뒤 EMG2_HOLD_N번 판단 동안은 유지)
//    바꾸기 : 손가락 -> 다른 손가락은 휴식을 거쳐야 함. EMG2_SWITCH_FREE_MASK에 든 동작(주먹)으로만
//             평활한 확률 EMG2_TH_SW 이상이 EMG2_SW_N번 연속이면 바로 바꿈
//    처음 EMG2_WARMUP_FRAMES번(필터 안정화)과 신호가 휴식 게이트보다 작을 때는 무조건 휴식
// ---------------------------------------------------------------
typedef struct {
  float    ps[EMG2_NUM_CLASSES];   // 평활한 확률
  int      output;
  int32_t  since;                  // 마지막으로 출력이 바뀐 뒤 지난 판단 수
  int      rest_run;               // 연속된 휴식 판단 수
  int      sw_run, sw_c;           // 바꾸기 후보가 연속된 횟수 / 그 후보
  uint32_t frames;
} Emg2Decider;

void emg2_decide_reset(Emg2Decider *d);
int  emg2_decide(Emg2Decider *d, const float *probs, float strength);

#ifdef __cplusplus
}
#endif

#endif  // EMG2_H
