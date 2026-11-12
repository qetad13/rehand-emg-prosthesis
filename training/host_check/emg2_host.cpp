// PC에서 펌웨어의 판단 코드(firmware/emg_board_v2/emg2.cpp)를 그대로 컴파일해 돌려 보는 프로그램.
// host_check.py가 이 프로그램을 빌드하고, 녹화 세션 전체를 넣어 PyTorch 결과와 비교합니다.
//
//   입력 파일: int16 [샘플][채널] (녹화 CSV의 값 그대로, 모델이 쓰는 채널 순서)
//   출력 파일: 프레임마다 float32 x (클래스 수 + 채널 수 + 1) = 확률, 로그 진폭, 힘 세기  +  int32 출력 클래스
//   인자: 입력 출력 [채널별 보정값 ...]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "emg2.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: emg2_host in.bin out.bin [gain_off x CH]\n");
    return 2;
  }
  FILE *fi = fopen(argv[1], "rb");
  FILE *fo = fopen(argv[2], "wb");
  if (!fi || !fo) {
    fprintf(stderr, "cannot open files\n");
    return 2;
  }
  static Emg2Front front;
  static Emg2Back back;
  static Emg2Decider dec;
  float gain[EMG2_NUM_CH] = {0};
  for (int c = 0; c < EMG2_NUM_CH && 3 + c < argc; ++c) gain[c] = (float)atof(argv[3 + c]);
  emg2_front_reset(&front);
  emg2_back_set_gain(&back, gain);
  emg2_back_reset(&back);
  emg2_decide_reset(&dec);

  int16_t s[EMG2_NUM_CH];
  float x[EMG2_NUM_CH];
  Emg2Hop hop;
  long frames = 0;
  while (fread(s, sizeof(int16_t), EMG2_NUM_CH, fi) == EMG2_NUM_CH) {
    for (int c = 0; c < EMG2_NUM_CH; ++c) x[c] = (float)s[c];
    if (!emg2_front_push(&front, x, &hop)) continue;
    emg2_back_frame(&back, &hop);
    const int32_t out = emg2_decide(&dec, back.probs, back.strength);
    fwrite(back.probs, sizeof(float), EMG2_NUM_CLASSES, fo);
    fwrite(back.logamp, sizeof(float), EMG2_NUM_CH, fo);
    fwrite(&back.strength, sizeof(float), 1, fo);
    fwrite(&out, sizeof(int32_t), 1, fo);
    ++frames;
  }
  fclose(fi);
  fclose(fo);
  printf("%ld frames\n", frames);
  return 0;
}
