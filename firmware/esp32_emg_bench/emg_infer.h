// EMG 손가락 동작 판단 (ESP32용, 라이브러리 없이 C로 직접 계산)
// train_emg.py가 만든 emg_model.h의 가중치를 사용합니다.
#ifndef EMG_INFER_H
#define EMG_INFER_H

#include <stdint.h>
#include "emg_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------
// 1) 필터: 채널마다 하나씩, 샘플이 들어올 때마다 호출
// ---------------------------------------------------------------
typedef struct {
  float z1[EMG_NUM_SOS];
  float z2[EMG_NUM_SOS];
} EmgFilter;

void  emg_filter_reset(EmgFilter *f);
float emg_filter_step(EmgFilter *f, float x);

// ---------------------------------------------------------------
// 2) 판단: 필터 통과한 최근 1초 신호를 넣으면 클래스 번호를 돌려줌
//    epoch: [Ch0 샘플 EMG_EPOCH_LEN개][Ch1 ...]... 순서 (시간순, 오래된 것부터)
//    probs: 클래스별 확률을 받을 배열 (EMG_NUM_CLASSES개), 필요 없으면 NULL
// ---------------------------------------------------------------
int emg_classify(const float *epoch, float *probs);

// 계산용 표(FFT, 해밍 창)를 미리 만들어 둠. setup()에서 한 번 부르면
// 첫 판단이 느려지는 것을 막을 수 있음 (안 불러도 첫 판단 때 자동으로 만듦)
void emg_init(void);

// 판단 과정 중 앞부분(정규화 + STFT)만 따로 실행 (속도 측정용)
// out: EMG_NUM_CH * EMG_NUM_BANDS * EMG_NUM_FRAMES 개
void emg_features(const float *epoch, float *out);

// ---------------------------------------------------------------
// 3) 휴식 게이트: 모든 채널 신호가 휴식 기준 x EMG_REST_GATE 보다 작으면 휴식으로 바꿈
//    (EMG_REST_GATE가 0이면 아무것도 안 함)
// ---------------------------------------------------------------
int emg_rest_gate(const float *epoch, int pred);

// ---------------------------------------------------------------
// 4) 최종 출력
//    휴식 -> 손가락: 최근 EMG_VOTE_N번 중 EMG_VOTE_K번 이상 같아야 켬
//    손가락 -> 다른 손가락: EMG_VOTE_SWITCH_K번 연속 같아야 바꿈
//    끄기: 휴식 판단이 EMG_VOTE_RELEASE_N번 연속이면
//    출력이 바뀐 뒤 EMG_VOTE_HOLD_N번 판단 동안은 다시 안 바꿈 (깜빡임 방지)
// ---------------------------------------------------------------
#define EMG_HIST_N (EMG_VOTE_N > EMG_VOTE_SWITCH_K ? EMG_VOTE_N : EMG_VOTE_SWITCH_K)
typedef struct {
  int hist[EMG_HIST_N];
  int count;
  int pos;
  int output;
  int rest_run;     // 연속된 휴식 판단 수
  long since;       // 마지막으로 출력이 바뀐 뒤 지난 판단 수
} EmgVote;

void emg_vote_reset(EmgVote *v);
int  emg_vote_update(EmgVote *v, int pred);

#ifdef __cplusplus
}
#endif

#endif  // EMG_INFER_H
