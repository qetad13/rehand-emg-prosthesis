// 의수 보드 v2에서 쓰는 자료형
// (Arduino IDE는 .ino의 함수 선언을 파일 맨 위에 자동으로 만들기 때문에,
//  함수 인자/반환형으로 쓰는 구조체는 이렇게 헤더에 따로 두어야 컴파일됩니다)
#pragma once
#include <Arduino.h>

// 자세 옵션
#define PF_THUMB_LATE 0x01   // 네 손가락을 먼저 굽히고 엄지는 조금 뒤에
#define PF_GRIP       0x02   // 물건을 쥐는 자세 (힘 세기에 따라 쥐는 힘도 조절)

struct HandPose {
  uint8_t     cmd;        // 명령 번호
  const char *name;       // 시리얼 출력용 이름
  uint16_t    t[7];       // 서보 7개 목표: 0 = 완전히 폄, 65535 = 완전히 굽힘
  uint8_t     flags;      // PF_*
};

struct Motion {
  bool     active;          // 자세 명령을 한 번이라도 받았는가
  uint8_t  cmd;             // 현재 명령 번호
  uint8_t  flags;           // 현재 자세의 옵션
  uint16_t target[7];       // 최종 목표
  bool     thumbPending;    // 엄지 단계가 남았는가
  uint32_t thumbAtMs;
  uint8_t  strength;        // 서보에 반영된 힘 세기 (0 = 기본)
  uint16_t torqueOverride;  // 0이 아니면 이 힘 한도로 (그립 단계 21~24)
  uint32_t lastSendMs;
  bool     fromLink;        // 무선으로 받은 명령인가 (시리얼 테스트 명령이면 무선 힘 세기를 따라가지 않음)
};

struct ServoMetrics {
  uint16_t pos[7];
  int16_t  vel[7];
  int16_t  cur[7];
  uint8_t  tmp[7];
  uint8_t  volt[7];         // x0.1V
};
