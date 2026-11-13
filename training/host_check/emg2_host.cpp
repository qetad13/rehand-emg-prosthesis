// PC에서 펌웨어의 판단 코드(firmware/emg_board_v2/emg2.cpp)를 그대로 컴파일해 돌려 보는 프로그램.
// host_check.py가 이 프로그램을 빌드하고, 녹화 세션 전체를 넣어 PyTorch 결과와 비교합니다.
//
//   emg2_host run   입력 출력 마지막층|- [채널별 보정값 ...]
//       입력: int16 [샘플][채널] (녹화 CSV의 값 그대로, 모델이 쓰는 채널 순서)
//       출력: 프레임마다 float32 x (클래스 수 + 채널 수 + 1) = 확률, 로그 진폭, 힘 세기  +  int32 출력 클래스
//       마지막층: float32 [클래스][임베딩] + [클래스] (사용자 보정으로 다시 맞춘 값). '-'면 원래 값
//   emg2_host adapt 임베딩 동작 출력
//       임베딩: float32 [n][임베딩], 동작: uint8 [n]
//       출력: emg2_adapt_fc 결과 float32 [클래스][임베딩] + [클래스] + 보정 녹화 정답률 1개
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "emg2.h"

static long file_size(FILE *f) {
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  return n;
}

static int run(int argc, char **argv) {
  FILE *fi = fopen(argv[2], "rb");
  FILE *fo = fopen(argv[3], "wb");
  if (!fi || !fo) {
    fprintf(stderr, "cannot open files\n");
    return 2;
  }
  static Emg2Front front;
  static Emg2Back back;
  static Emg2Decider dec;
  float gain[EMG2_NUM_CH] = {0};
  for (int c = 0; c < EMG2_NUM_CH && 5 + c < argc; ++c) gain[c] = (float)atof(argv[5 + c]);
  emg2_front_reset(&front);
  emg2_back_set_gain(&back, gain);
  emg2_back_set_fc(&back, NULL, NULL);
  if (strcmp(argv[4], "-") != 0) {
    static float w[EMG2_NUM_CLASSES * EMG2_EMBED], b[EMG2_NUM_CLASSES];
    FILE *ff = fopen(argv[4], "rb");
    if (!ff || fread(w, sizeof(float), EMG2_NUM_CLASSES * EMG2_EMBED, ff) != EMG2_NUM_CLASSES * EMG2_EMBED ||
        fread(b, sizeof(float), EMG2_NUM_CLASSES, ff) != EMG2_NUM_CLASSES) {
      fprintf(stderr, "cannot read last layer file\n");
      return 2;
    }
    fclose(ff);
    emg2_back_set_fc(&back, w, b);
  }
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

static int adapt(char **argv) {
  FILE *fh = fopen(argv[2], "rb");
  FILE *fy = fopen(argv[3], "rb");
  FILE *fo = fopen(argv[4], "wb");
  if (!fh || !fy || !fo) {
    fprintf(stderr, "cannot open files\n");
    return 2;
  }
  const long n = file_size(fy);
  float *H = (float *)malloc(sizeof(float) * (size_t)n * EMG2_EMBED);
  uint8_t *Y = (uint8_t *)malloc((size_t)n);
  if (!H || !Y || fread(H, sizeof(float) * EMG2_EMBED, (size_t)n, fh) != (size_t)n || fread(Y, 1, (size_t)n, fy) != (size_t)n) {
    fprintf(stderr, "cannot read adapt input\n");
    return 2;
  }
  static float w[EMG2_NUM_CLASSES * EMG2_EMBED], b[EMG2_NUM_CLASSES];
  const float acc = emg2_adapt_fc(H, Y, (int)n, w, b, NULL);
  fwrite(w, sizeof(float), EMG2_NUM_CLASSES * EMG2_EMBED, fo);
  fwrite(b, sizeof(float), EMG2_NUM_CLASSES, fo);
  fwrite(&acc, sizeof(float), 1, fo);
  fclose(fh);
  fclose(fy);
  fclose(fo);
  free(H);
  free(Y);
  printf("adapt %ld frames, acc %.4f\n", n, acc);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 5 && strcmp(argv[1], "run") == 0) return run(argc, argv);
  if (argc >= 5 && strcmp(argv[1], "adapt") == 0) return adapt(argv);
  fprintf(stderr, "usage: emg2_host run in.bin out.bin fc.bin|- [gain_off x CH]\n"
                  "       emg2_host adapt embed.bin labels.bin out.bin\n");
  return 2;
}
