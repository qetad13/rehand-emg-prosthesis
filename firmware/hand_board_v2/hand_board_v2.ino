// =====================================================================
// ReHand 의수 보드 v2 (XIAO ESP32-S3 + Feetech HLS3606M x7)
//
// 예전 코드(rehand.ino)와 달라진 점
//   1. 근전도 보드와 직접 연결 (ESP-NOW, RehandLink.h). 라즈베리파이 없이 두 보드만으로 동작
//   2. 힘 세기 비례 제어: 세게 힘주면 빠르고 세게, 약하게 주면 천천히 약하게 (움직이는 동안 계속 따라감)
//   3. 멈추지 않는 동작: '네 손가락 먼저 -> 엄지' 순서를 기다리는 동안에도 새 명령·무선 수신이 계속 처리됨
//      (예전에는 vTaskDelay(250) 동안 아무것도 못 받았음)
//   4. 호밍을 별도 태스크로: 호밍 중에도 무선이 살아 있고 상태(호밍 중)를 근전도 보드에 알림
//   5. 자세를 표(POSES)로 관리: 동작 추가/수정은 표 한 줄
//   6. 서보 전압·온도·응답 감시 -> 근전도 보드 LED와 시리얼로 알림
//   7. 시리얼 입력이 loop()를 최대 1초씩 멈추던 문제(parseInt) 수정
//   8. 연결 상대 고정: 처음 연결된 근전도 보드의 주소를 저장하고, 다른 보드에서 온 명령은 무시
//
// 명령 번호 (예전과 같음)
//   1~5 엄지/검지/중지/약지/소지, 6 펴기(휴식), 7 주먹, 8 집게
//   11~19 매크로 동작, 20 호밍, 21~24 그립 단계(21 가장 셈 ~ 24 가장 약함)
//
// 시리얼 모니터 테스트 (줄 끝에 Enter)
//   숫자      그 명령 실행 (예: 7)
//   s숫자     힘 세기 지정 후 현재 동작에 반영 (예: s30, s0 = 기본)
//   ?         상태 출력 (자세, 온도, 전압, 전류, 무선)
//   p         연결 상대 지우기 (다른 근전도 보드로 바꿀 때. 보드의 BOOT 버튼 3초도 같음)
// =====================================================================
#include <Arduino.h>
#include <HLSCL.h>
#include <Preferences.h>
#include "HandConfig.h"
#include "Homing.h"
#include "HandTypes.h"
#include "esp_system.h"
#include <WiFi.h>

// ---------------------------------------------------------------------
// 사용자 설정
// ---------------------------------------------------------------------
// 힘 세기(1~100) -> 서보 속도. 0(세기 정보 없음)이면 기본 속도(g_speed, 최고 속도)
#define PROP_SPEED       1       // 0이면 속도는 항상 기본값
#define SPEED_MIN        1200    // 세기 1일 때 속도 (호밍 속도 2400의 절반 정도)
#define SPEED_MAX        6000    // 세기 100일 때 속도 (서보 최고 속도 이상이면 최고 속도로 움직임)

// 힘 세기 -> 쥐는 힘(서보 힘 한도). 쥐는 자세(PF_GRIP)에만 적용
//   한도가 약 200 미만이면 서보가 손가락 기구 저항을 못 이겨 목표 위치에 못 갈 수 있음
#define PROP_TORQUE      1       // 0이면 힘 한도는 항상 기본값(g_torque)
#define GRIP_TORQUE_MIN  300
#define GRIP_TORQUE_MAX  1000

#define STRENGTH_STEP    8       // 힘 세기가 이만큼 바뀌어야 서보에 다시 보냄 (버스 부담/떨림 방지)
#define STRENGTH_MIN_MS  80      // 다시 보내는 최소 간격
#define THUMB_LATE_MS    250     // '네 손가락 먼저' 자세에서 엄지가 따라 움직이기까지의 시간

// 서보 전원 전압 감시 (x0.1V). 2S 리튬(7.4V)이면 66 정도를 권장. 0이면 끔
#define LOW_VOLT_DV      0

// 연결 상대 지우기 버튼: 3초 누르면 저장된 근전도 보드 주소를 지움 (XIAO의 BOOT 버튼 = GPIO0). 안 쓰려면 -1
#define PAIR_BTN_PIN     0
#define PAIR_HOLD_MS     3000

// ---------------------------------------------------------------------
// 무선 링크: 근전도 보드(emg_board_v2 또는 예전 esp32_emg_realtime)와 ESP-NOW로 직접 연결
// ---------------------------------------------------------------------
#define REHAND_ROLE_HAND
#include "RehandLink.h"
static void linkBegin() { rhHandBegin(); }
static void linkPoll(uint8_t flags, uint8_t maxTempC, uint8_t volt) { rhHandPoll(flags, maxTempC, volt); }
static bool linkTakeCommand(uint8_t *cmd, uint8_t *strength) { return rhHandTakeCommand(cmd, strength); }
static uint8_t linkStrength() { return rhHandStrength(); }
static bool linkLost() { return rhHandLinkLost(); }
static void linkUnpair() { rhHandUnpair(); }

HLSCL hlscl;
Preferences prefs;

// ---- 서보 버스 UART (ESP32-S3 XIAO: RX=2, TX=3) ----
#define SERIAL2_TX_PIN 3
#define SERIAL2_RX_PIN 2

// 서보: 0 엄지 돌리기(손바닥 쪽으로 모으기), 1 엄지 아래 마디, 2 엄지 끝 마디, 3~6 검지·중지·약지·소지
const uint8_t SERVO_IDS[7] = { 0, 1, 2, 3, 4, 5, 6 };
ServoData sd[7];

// ---- 기본 속도 / 가속 / 힘 한도 ----
static uint16_t g_speed[7]  = {32766, 32766, 32766, 32766, 32766, 32766, 32766};
static uint8_t  g_accel[7]  = {0, 0, 0, 0, 0, 0, 0};                 // 0..255
static uint16_t g_torque[7] = {700, 700, 700, 700, 700, 700, 700};   // 0..1000

// ---- 온도 보호 ----
static const uint8_t  TEMP_CUTOFF_C    = 70;    // 이 온도 이상인 서보는
static const uint16_t HOT_TORQUE_LIMIT = 500;   // 힘 한도를 이 값으로 낮춤

// ---- 서보 상태 읽기 (주소 56부터 15바이트: 위치, 속도, 부하, 전압, 온도, ..., 전류) ----
#define REG_BLOCK_LEN   15
#define REG_BLOCK_START 56

static ServoMetrics gMetrics;
static volatile uint32_t gReadOk = 0, gReadFail = 0;   // 상태 읽기 성공/실패 횟수
static volatile uint32_t gLastReadOkMs = 0;

static SemaphoreHandle_t gMetricsMux;
SemaphoreHandle_t gBusMux = nullptr;

static inline uint16_t u16_min(uint16_t a, uint16_t b) { return (a < b) ? a : b; }

static inline uint8_t getTempC(uint8_t ch) {
  uint8_t t = 0;
  if (gMetricsMux) xSemaphoreTake(gMetricsMux, portMAX_DELAY);
  t = gMetrics.tmp[ch];
  if (gMetricsMux) xSemaphoreGive(gMetricsMux);
  return t;
}
static inline bool isHot(uint8_t ch) { return getTempC(ch) >= TEMP_CUTOFF_C; }

// ---- 호밍으로 구한 폄 위치를 NVS에 저장/불러오기 ----
static void loadManualExtendsFromNVS() {
  prefs.begin("hand", false);   // 읽기 전용으로 열면 처음 쓰는 보드(저장 공간 없음)에서 오류가 남
  for (uint8_t i = 0; i < 7; ++i) {
    String key = "ext" + String(i);
    int v = prefs.getInt(key.c_str(), -1);
    if (v >= 0 && v <= 4095) sd[i].extend_count = v;
  }
  prefs.end();
}

static void saveExtendsToNVS() {
  prefs.begin("hand", false);
  for (uint8_t i = 0; i < 7; ++i) {
    String kext = "ext" + String(i);
    prefs.putInt(kext.c_str(), (int)sd[i].extend_count);
  }
  prefs.end();
}

// ---- 0~65535(폄~쥠) <-> 서보 위치(0~4095) ----
static inline uint16_t mapRawToU16(uint8_t ch, uint16_t raw) {
  int32_t ext  = sd[ch].extend_count;
  int32_t gra  = sd[ch].grasp_count;
  int32_t span = gra - ext;
  if (span == 0) return 0;
  int32_t val = ((int32_t)(raw - ext) * 65535L) / span;
  if (val < 0) val = 0;
  if (val > 65535) val = 65535;
  return (uint16_t)val;
}
static inline uint16_t mapU16ToRaw(uint8_t ch, uint16_t u16) {
  int32_t ext = sd[ch].extend_count;
  int32_t gra = sd[ch].grasp_count;
  int32_t raw32;
  if (ext == 0 && gra == 0) raw32 = ((uint64_t)u16 * 4095u) / 65535u;
  else                      raw32 = ext + ((int64_t)u16 * (gra - ext)) / 65535LL;
  if (raw32 < 0)    raw32 = 0;
  if (raw32 > 4095) raw32 = 4095;
  return (uint16_t)raw32;
}

// ---------------------------------------------------------------------
// 자세 표: {명령, 이름, {엄지 돌리기, 엄지 아래, 엄지 끝, 검지, 중지, 약지, 소지}, 옵션}
//   값: 0 = 완전히 폄, 65535 = 완전히 굽힘
//   PF_THUMB_LATE : 네 손가락을 먼저 굽히고 엄지는 THUMB_LATE_MS 뒤에 (엄지가 손가락 위로 덮이는 자세.
//                   같이 움직이면 엄지가 손가락 밑에 끼일 수 있음)
//   PF_GRIP       : 물건을 쥐는 자세. 힘 세기에 따라 쥐는 힘(힘 한도)도 조절
// ---------------------------------------------------------------------
#define FULL 65535

static const HandPose POSES[] = {
  // ---- 근전도 모드 (1~8) ----
  { 1, "엄지",            {FULL,  FULL,  FULL,  0,     0,     0,     0    }, 0 },
  { 2, "검지",            {0,     0,     0,     FULL,  0,     0,     0    }, 0 },
  { 3, "중지",            {0,     0,     0,     0,     FULL,  0,     0    }, 0 },
  { 4, "약지",            {0,     0,     0,     0,     0,     FULL,  0    }, 0 },
  { 5, "소지",            {0,     0,     0,     0,     0,     0,     FULL }, 0 },
  { 6, "펴기(휴식)",      {0,     0,     0,     0,     0,     0,     0    }, 0 },
  { 7, "주먹",            {FULL,  FULL,  FULL,  FULL,  FULL,  FULL,  FULL }, PF_THUMB_LATE | PF_GRIP },
  { 8, "집게",            {57000, 30000, 20000, 31000, 0,     0,     0    }, PF_GRIP },
  // ---- 매크로 모드 (11~19) ----
  {11, "전체 쥐기",       {FULL,  FULL,  FULL,  FULL,  FULL,  FULL,  FULL }, PF_THUMB_LATE | PF_GRIP },
  {12, "반쯤 펴기",       {30000, 30000, 30000, 30000, 30000, 30000, 30000}, 0 },
  {13, "가리키기",        {FULL,  FULL,  FULL,  0,     FULL,  FULL,  FULL }, PF_THUMB_LATE },
  {14, "집게 파지",       {57000, 30000, 20000, 31000, 0,     0,     0    }, PF_GRIP },
  {15, "세 손가락 파지",  {FULL,  30000, 20000, 31000, 36000, FULL,  FULL }, PF_GRIP },
  {16, "브이",            {FULL,  FULL,  FULL,  0,     0,     FULL,  FULL }, 0 },
  {17, "가위",            {0,     0,     0,     0,     FULL,  FULL,  FULL }, 0 },
  {18, "따봉",            {0,     0,     0,     FULL,  FULL,  FULL,  FULL }, 0 },
  {19, "네 손가락 파지",  {FULL,  30000, 20000, 31000, 36000, 35000, 34000}, PF_GRIP },
};
#define POSE_COUNT ((int)(sizeof(POSES) / sizeof(POSES[0])))

// ---- 그립 단계 (명령 21~24): 굽힘과 힘 한도를 함께 바꿈 (예전 코드와 같은 값) ----
static const float    GRIP_THUMB_ROTATE = 1.00f;                           // 엄지 돌리기는 모든 단계에서 끝까지
//                                    21 주먹  22      23      24
static const float    GRIP_CLOSE[4]  = { 1.00f,  0.95f,  0.90f,  0.90f };  // 굽힘 (0 폄 ~ 1 다 굽힘)
static const uint16_t GRIP_TORQUE[4] = { 1000,   700,    450,    300   };  // 힘 한도

// ---------------------------------------------------------------------
// 동작 실행 (기다리지 않음: 목표를 보내 두고, 엄지 단계는 motionPoll()이 시간이 되면 보냄)
// ---------------------------------------------------------------------
static Motion g_motion = {};

static uint16_t speedFor(uint8_t ch, uint8_t strength) {
#if PROP_SPEED
  if (strength > 0) {
    const uint32_t v = SPEED_MIN + (uint32_t)(SPEED_MAX - SPEED_MIN) * strength / 100;
    return (uint16_t)(v < g_speed[ch] ? v : g_speed[ch]);
  }
#else
  (void)strength;
#endif
  return g_speed[ch];
}

static uint16_t torqueFor(uint8_t ch, uint8_t strength) {
  uint16_t base = g_torque[ch];
  if (g_motion.torqueOverride) {
    base = g_motion.torqueOverride;
  }
#if PROP_TORQUE
  else if ((g_motion.flags & PF_GRIP) && strength > 0) {
    base = (uint16_t)(GRIP_TORQUE_MIN + (uint32_t)(GRIP_TORQUE_MAX - GRIP_TORQUE_MIN) * strength / 100);
  }
#else
  (void)strength;
#endif
  if (isHot(ch)) base = u16_min(base, HOT_TORQUE_LIMIT);
  return base;
}

// 현재 단계의 목표를 서보로 보냄 (엄지 단계가 남았으면 엄지 3개는 펴 둠)
static void motionSend() {
  int16_t  pos[7];
  uint16_t spd[7], tor[7];
  for (uint8_t i = 0; i < 7; ++i) {
    const uint16_t t = (g_motion.thumbPending && i < 3) ? 0 : g_motion.target[i];
    pos[i] = (int16_t)mapU16ToRaw(i, t);
    spd[i] = speedFor(i, g_motion.strength);
    tor[i] = torqueFor(i, g_motion.strength);
  }
  if (gBusMux) xSemaphoreTake(gBusMux, portMAX_DELAY);
  hlscl.SyncWritePosEx((uint8_t *)SERVO_IDS, 7, pos, spd, g_accel, tor);
  if (gBusMux) xSemaphoreGive(gBusMux);
  g_motion.lastSendMs = millis();
}

// 새 자세 시작. stage: 네 손가락 먼저 움직일지 (PF_THUMB_LATE 자세라도 이미 쥐고 있으면 false)
static void motionStart(uint8_t cmd, const uint16_t target[7], uint8_t flags, bool stage,
                        uint8_t strength, uint16_t torqueOverride, bool fromLink) {
  g_motion.active = true;
  g_motion.cmd = cmd;
  g_motion.flags = flags;
  memcpy(g_motion.target, target, sizeof(g_motion.target));
  g_motion.thumbPending = stage;
  g_motion.thumbAtMs = millis() + THUMB_LATE_MS;
  g_motion.strength = strength;
  g_motion.torqueOverride = torqueOverride;
  g_motion.fromLink = fromLink;
  motionSend();
}

// loop()에서 계속 호출: 엄지 단계 처리
static void motionPoll() {
  if (g_motion.active && g_motion.thumbPending && (int32_t)(millis() - g_motion.thumbAtMs) >= 0) {
    g_motion.thumbPending = false;
    motionSend();
  }
}

// 움직이는 중/쥐고 있는 중에 힘 세기가 바뀌면 같은 목표를 새 속도·힘으로 다시 보냄
static void motionUpdateStrength(uint8_t strength) {
  if (!g_motion.active || g_motion.torqueOverride) return;
  if (g_motion.cmd == 6) return;                                   // 펴기는 항상 기본 속도
  const int diff = abs((int)strength - (int)g_motion.strength);
  const bool edge = (strength == 0) != (g_motion.strength == 0);   // 세기 정보가 생기거나 없어짐
  if (!edge && diff < STRENGTH_STEP) return;
  if (millis() - g_motion.lastSendMs < STRENGTH_MIN_MS) return;
  g_motion.strength = strength;
  motionSend();
}

// 호밍 (별도 태스크에서 실행: 그동안에도 loop()가 돌아 무선 수신·상태 보고가 계속됨)
static volatile bool g_homingActive = false;

static void homingTaskFn(void *arg) {
  (void)arg;
  HOMING_start();
  saveExtendsToNVS();
  g_motion.active = false;            // 호밍 뒤에는 모두 편 상태
  gLastReadOkMs = millis();           // 호밍 중에는 상태를 못 읽었으므로 '응답 없음' 판단을 여기서부터 다시 시작
  g_homingActive = false;
  Serial.println("[HOMING] 완료");
  vTaskDelete(NULL);
}

static bool homingBusy() { return g_homingActive || HOMING_isBusy(); }

static void startHoming() {
  if (homingBusy()) return;
  g_homingActive = true;
  Serial.println("[CMD] 20 : 호밍 시작");
  if (xTaskCreatePinnedToCore(homingTaskFn, "homing", 8192, NULL, 1, NULL, 1) != pdPASS) {
    g_homingActive = false;
    Serial.println("[HOMING] 태스크를 만들지 못함");
  }
}

static const HandPose *findPose(uint8_t cmd) {
  for (int i = 0; i < POSE_COUNT; ++i) {
    if (POSES[i].cmd == cmd) return &POSES[i];
  }
  return nullptr;
}

static void executeHandCommand(uint8_t command, uint8_t strength, bool fromLink) {
  if (command == 20) {
    startHoming();
    return;
  }

  // 지금 엄지가 손가락 위로 덮인 채 쥐고 있는가 (그 상태에서 다시 '네 손가락 먼저'를 하면 엄지가 잠깐 펴져 물건을 놓침)
  const bool gripHeld = g_motion.active && !g_motion.thumbPending &&
                        ((g_motion.cmd >= 21 && g_motion.cmd <= 24) || g_motion.cmd == 7 || g_motion.cmd == 11);

  if (command >= 21 && command <= 24) {
    const int lv = command - 21;
    const uint16_t pos = (uint16_t)(65535.0f * GRIP_CLOSE[lv]);
    const uint16_t rot = (uint16_t)(65535.0f * GRIP_THUMB_ROTATE);
    const uint16_t target[7] = { rot, pos, pos, pos, pos, pos, pos };
    Serial.printf("[CMD] %d : 그립 단계 (굽힘 %d%%, 힘 한도 %u)\n", command,
                  (int)(GRIP_CLOSE[lv] * 100.0f + 0.5f), GRIP_TORQUE[lv]);
    motionStart(command, target, PF_GRIP, !gripHeld, 0, GRIP_TORQUE[lv], fromLink);
    return;
  }

  const HandPose *p = findPose(command);
  if (!p) {
    Serial.printf("[CMD] 모르는 명령 %u\n", command);
    return;
  }
  if (command == 6) strength = 0;                 // 펴기는 항상 기본(최고) 속도: 놓는 동작은 빨라야 안전
  const bool stage = (p->flags & PF_THUMB_LATE) && !gripHeld;
  Serial.printf("[CMD] %u : %s", command, p->name);
  if (strength) Serial.printf(" (힘 세기 %u)", strength);
  Serial.println();
  motionStart(command, p->t, p->flags, stage, strength, 0, fromLink);
}

// ---------------------------------------------------------------------
// 서보 상태 읽기 태스크 (코어 1, 100Hz). 명령이 버스를 쓰는 중이면 그 주기는 건너뜀
// ---------------------------------------------------------------------
static inline uint16_t leu_u16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static inline int16_t decode_signmag15(uint8_t lo, uint8_t hi) {
  uint16_t mag = ((uint16_t)(hi & 0x7F) << 8) | lo;
  return (hi & 0x80) ? -(int16_t)mag : (int16_t)mag;
}

static void TaskSyncRead_Core1(void *arg) {
  (void)arg;
  uint8_t  rx[REG_BLOCK_LEN];
  uint16_t pos[7];
  int16_t  vel[7], cur[7];
  uint8_t  tmp[7], volt[7];
  const TickType_t period = pdMS_TO_TICKS(10);
  TickType_t nextWake = xTaskGetTickCount();
  for (;;) {
    if (gBusMux && xSemaphoreTake(gBusMux, 0) != pdTRUE) {
      vTaskDelayUntil(&nextWake, period);
      continue;
    }
    bool ok = true;
    hlscl.syncReadPacketTx((uint8_t *)SERVO_IDS, 7, REG_BLOCK_START, REG_BLOCK_LEN);
    for (uint8_t i = 0; i < 7; ++i) {
      if (!hlscl.syncReadPacketRx(SERVO_IDS[i], rx)) { ok = false; break; }
      pos[i]  = mapRawToU16(i, leu_u16(&rx[0]));
      vel[i]  = decode_signmag15(rx[2], rx[3]);
      volt[i] = rx[6];
      tmp[i]  = rx[7];
      cur[i]  = decode_signmag15(rx[13], rx[14]);
    }
    if (gBusMux) xSemaphoreGive(gBusMux);
    if (ok) {
      xSemaphoreTake(gMetricsMux, portMAX_DELAY);
      for (int i = 0; i < 7; ++i) {
        gMetrics.pos[i] = pos[i];
        gMetrics.vel[i] = vel[i];
        gMetrics.tmp[i] = tmp[i];
        gMetrics.cur[i] = cur[i];
        gMetrics.volt[i] = volt[i];
      }
      xSemaphoreGive(gMetricsMux);
      gReadOk = gReadOk + 1;
      gLastReadOkMs = millis();
    } else {
      gReadFail = gReadFail + 1;
    }
    vTaskDelayUntil(&nextWake, period);
  }
}

// 무선으로 보낼 의수 상태
static uint8_t handStatusFlags(uint8_t *maxTempC, uint8_t *minVolt) {
  uint8_t flags = homingBusy() ? HAND_FLAG_HOMING : 0;
  uint8_t maxT = 0, minV = 255;
  xSemaphoreTake(gMetricsMux, portMAX_DELAY);
  for (uint8_t i = 0; i < 7; ++i) {
    if (gMetrics.tmp[i] > maxT) maxT = gMetrics.tmp[i];
    if (gMetrics.volt[i] < minV) minV = gMetrics.volt[i];
  }
  xSemaphoreGive(gMetricsMux);
  if (gReadOk == 0) minV = 0;                                   // 아직 한 번도 못 읽음
  if (maxT >= TEMP_CUTOFF_C) flags |= HAND_FLAG_HOT;
  if (LOW_VOLT_DV > 0 && minV > 0 && minV < LOW_VOLT_DV) flags |= HAND_FLAG_LOW_VOLT;
  // 호밍 중에는 호밍이 버스를 계속 쓰므로 상태를 못 읽는 것이 정상
  if (!homingBusy() && millis() - gLastReadOkMs > 1000) flags |= HAND_FLAG_SERVO_ERR;
  *maxTempC = maxT;
  *minVolt = minV;
  return flags;
}

// ---------------------------------------------------------------------
// 시리얼 테스트 입력 (loop()를 멈추지 않음)
// ---------------------------------------------------------------------
static uint8_t g_serialStrength = 0;

static void printStatus() {
  ServoMetrics m;
  xSemaphoreTake(gMetricsMux, portMAX_DELAY);
  m = gMetrics;
  xSemaphoreGive(gMetricsMux);
  Serial.printf("[상태] 명령 %u, 힘 세기 %u, 무선 %s, 상태 읽기 성공 %lu / 실패 %lu\n",
                g_motion.active ? g_motion.cmd : 0, g_motion.strength, linkLost() ? "끊김" : "수신 중",
                (unsigned long)gReadOk, (unsigned long)gReadFail);
  uint8_t mate[6];
  if (rhHandMate(mate)) {
    Serial.printf("  연결 상대 %02X:%02X:%02X:%02X:%02X:%02X, 다른 보드에서 와서 무시한 명령 %lu개\n",
                  mate[0], mate[1], mate[2], mate[3], mate[4], mate[5], (unsigned long)rhHandRejected());
  } else {
    Serial.println("  연결 상대 없음 (처음 명령을 보내는 근전도 보드와 연결됨)");
  }
  Serial.print("  위치(%)  ");
  for (int i = 0; i < 7; ++i) Serial.printf(" %3u", (unsigned)((uint32_t)m.pos[i] * 100 / 65535));
  Serial.print("\n  전류     ");
  for (int i = 0; i < 7; ++i) Serial.printf(" %4d", m.cur[i]);
  Serial.print("\n  온도(C)  ");
  for (int i = 0; i < 7; ++i) Serial.printf(" %3u", m.tmp[i]);
  Serial.print("\n  전압(V)  ");
  for (int i = 0; i < 7; ++i) Serial.printf(" %u.%u", m.volt[i] / 10, m.volt[i] % 10);
  Serial.println();
}

static void handleSerialLine(const char *line) {
  while (*line == ' ') ++line;
  if (*line == '\0') return;
  if (*line == '?') { printStatus(); return; }
  if (*line == 'p' || *line == 'P') { linkUnpair(); return; }
  if (*line == 's' || *line == 'S') {
    int v = atoi(line + 1);
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    g_serialStrength = (uint8_t)v;
    Serial.printf("[SERIAL] 힘 세기 %d\n", v);
    if (g_motion.active && !g_motion.fromLink) motionUpdateStrength(g_serialStrength);
    return;
  }
  const int command = atoi(line);
  if ((command >= 1 && command <= 8) || (command >= 11 && command <= 24)) {
    executeHandCommand((uint8_t)command, g_serialStrength, false);
  } else {
    Serial.println("[SERIAL] 명령: 1~8, 11~24 / s숫자 = 힘 세기 / ? = 상태 / p = 연결 상대 지우기");
  }
}

// BOOT 버튼을 3초 누르면 연결 상대를 지움 (한 번 누를 때 한 번만)
static void pairButtonPoll() {
#if PAIR_BTN_PIN >= 0
  static uint32_t downMs = 0;
  static bool fired = false;
  if (digitalRead(PAIR_BTN_PIN) == LOW) {
    if (downMs == 0) downMs = millis() | 1;
    if (!fired && millis() - downMs >= PAIR_HOLD_MS) {
      fired = true;
      linkUnpair();
    }
  } else {
    downMs = 0;
    fired = false;
  }
#endif
}

static void serialPoll() {
  static char buf[16];
  static uint8_t n = 0;
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      buf[n] = '\0';
      if (n > 0) handleSerialLine(buf);
      n = 0;
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = c;
    }
  }
}

void setup() {
  Serial.begin(921600);
  delay(100);

  Serial2.begin(1000000, SERIAL_8N1, SERIAL2_RX_PIN, SERIAL2_TX_PIN);
  hlscl.pSerial = &Serial2;

  resetSdToBaseline();
  loadManualExtendsFromNVS();
#if defined(LEFT_HAND)
  Serial.println("[BOOT] ReHand 의수 v2, 왼손");
#else
  Serial.println("[BOOT] ReHand 의수 v2, 오른손");
#endif

  // 전원을 새로 켰을 때만 호밍 (리셋 버튼/업로드 뒤에는 건너뜀)
  const esp_reset_reason_t reason = esp_reset_reason();
  if (reason == ESP_RST_POWERON || reason == ESP_RST_BROWNOUT) {
    Serial.println("[BOOT] 전원 켜짐 -> 호밍");
    HOMING_start();
    saveExtendsToNVS();
  } else {
    Serial.println("[BOOT] 전원 리셋이 아님 -> 호밍 건너뜀 (필요하면 20 입력)");
  }

  hlscl.syncReadBegin(sizeof(SERVO_IDS), REG_BLOCK_LEN, /*rx_fix*/ 8);

  gBusMux = xSemaphoreCreateMutex();
  gMetricsMux = xSemaphoreCreateMutex();
  gLastReadOkMs = millis();
  xTaskCreatePinnedToCore(TaskSyncRead_Core1, "SyncRead", 4096, NULL, 1, NULL, 1);

#if PAIR_BTN_PIN >= 0
  pinMode(PAIR_BTN_PIN, INPUT_PULLUP);
#endif
  linkBegin();
}

void loop() {
  // 무선 수신과 상태 보내기는 호밍 중에도 계속한다 (그동안 온 명령은 호밍이 끝난 뒤 가장 최근 것만 실행됨)
  uint8_t maxTempC = 0, minVolt = 0;
  const uint8_t flags = handStatusFlags(&maxTempC, &minVolt);
  linkPoll(flags, maxTempC, minVolt);
  pairButtonPoll();

  // 이상 상태는 바뀔 때만 한 번씩 출력
  static uint8_t lastFlags = 0;
  const uint8_t warn = flags & (HAND_FLAG_HOT | HAND_FLAG_LOW_VOLT | HAND_FLAG_SERVO_ERR);
  if (warn != lastFlags) {
    lastFlags = warn;
    if (warn & HAND_FLAG_HOT)       Serial.printf("[경고] 서보 과열 %u°C -> 힘 한도 %u로 제한\n", maxTempC, HOT_TORQUE_LIMIT);
    if (warn & HAND_FLAG_LOW_VOLT)  Serial.printf("[경고] 서보 전압 낮음 %u.%uV\n", minVolt / 10, minVolt % 10);
    if (warn & HAND_FLAG_SERVO_ERR) Serial.println("[경고] 서보 응답 없음 -> 배선/전원 확인");
    if (!warn)                      Serial.println("[정상] 경고 해제");
  }

  if (homingBusy()) {
    while (Serial.available()) { Serial.read(); }
    vTaskDelay(pdMS_TO_TICKS(5));
    return;
  }

  motionPoll();

  // 무선 명령: 새 명령일 때만 실행 (재전송은 무시). 연결이 끊기면 아무것도 안 함 = 현재 자세 유지
  uint8_t command, strength;
  if (linkTakeCommand(&command, &strength)) {
    executeHandCommand(command, strength, true);
  } else if (!linkLost() && g_motion.active && g_motion.fromLink) {
    motionUpdateStrength(linkStrength());         // 같은 동작을 유지하는 동안 힘 세기만 따라감
  }

  serialPoll();
  vTaskDelay(pdMS_TO_TICKS(5));
}
