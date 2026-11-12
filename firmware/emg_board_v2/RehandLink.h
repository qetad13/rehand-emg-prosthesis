// =====================================================================
// ReHand 무선 링크 v2 (ESP-NOW, 근전도 보드 <-> 의수 보드 직접 연결)
//
// 이 파일은 firmware/emg_board_v2 와 firmware/hand_board_v2 에 똑같은 내용으로 들어 있습니다.
// 한쪽을 고치면 다른 쪽도 같이 고쳐 주세요. (include 하기 전에 역할을 정의)
//   의수 보드   : #define REHAND_ROLE_HAND
//   근전도 보드 : #define REHAND_ROLE_EMG
//
//   근전도 -> 의수  RhCmdPacket    8바이트  "RH" 1 | 세션 | 번호 | 명령 | 힘 세기 | 플래그
//   의수 -> 근전도  RhStatusPacket 10바이트 "RH" 2 | 세션 | 번호 | 실행한 명령 | 상태 | 최고 온도 | 전압 | 버전
//
// - 근전도 보드는 명령이 바뀌면 바로, 그리고 0.1초마다 현재 명령을 다시 보냅니다.
//   의수는 (세션, 번호)가 바뀔 때만 실행하므로 패킷을 잃어도 복구되고 같은 동작이 반복 실행되지 않습니다.
//   힘 세기는 번호와 상관없이 매 패킷의 값을 그대로 반영합니다 (움직이는 동안 속도·힘이 따라감).
// - 처음에는 방송(broadcast)으로 보내다가, 상대 보드의 패킷을 받으면 그 주소로 1:1 전송으로 바꿉니다.
//   (1:1 전송은 무선 계층에서 자동 재전송이 되어 방송보다 훨씬 안정적)
// - 의수는 예전 근전도 펌웨어의 1~2바이트 패킷 {명령, 힘 세기}도 그대로 받습니다.
// =====================================================================
#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define RH_PROTO_VERSION   2
#define RH_WIFI_CHANNEL    1      // 두 보드가 같은 값이어야 함
#define RH_TYPE_CMD        1
#define RH_TYPE_STATUS     2

// 의수 상태 비트
#ifndef HAND_FLAG_HOMING
#define HAND_FLAG_HOMING    0x01  // 호밍 중
#define HAND_FLAG_LINK_LOST 0x02  // 명령이 끊겨 현재 자세 유지 중
#define HAND_FLAG_HOT       0x04  // 온도 제한에 걸린 서보가 있음
#endif
#define HAND_FLAG_LOW_VOLT  0x08  // 서보 전원 전압이 낮음 (배터리 교체/충전)
#define HAND_FLAG_SERVO_ERR 0x10  // 서보 응답이 없음 (배선/전원 확인)

typedef struct __attribute__((packed)) {
  uint8_t magic[2];     // 'R','H'
  uint8_t type;         // RH_TYPE_CMD
  uint8_t session;      // 근전도 보드가 켜질 때 정하는 값
  uint8_t seq;          // 명령이 바뀔 때마다 1 증가
  uint8_t cmd;          // 1~8 근전도 동작, 11~19 매크로, 20 호밍, 21~24 그립 단계
  uint8_t strength;     // 0 = 세기 정보 없음(기본 속도), 1~100
  uint8_t flags;        // 예약
} RhCmdPacket;

typedef struct __attribute__((packed)) {
  uint8_t magic[2];     // 'R','H'
  uint8_t type;         // RH_TYPE_STATUS
  uint8_t session;      // 마지막으로 실행한 명령의 세션/번호/명령
  uint8_t seq;
  uint8_t doneCmd;
  uint8_t flags;        // HAND_FLAG_*
  uint8_t maxTempC;     // 서보 최고 온도
  uint8_t voltage_dV;   // 서보 전원 전압 x10 (예: 74 = 7.4V), 모르면 0
  uint8_t version;      // RH_PROTO_VERSION
} RhStatusPacket;

static inline bool rhValidCmd(uint8_t c) { return (c >= 1 && c <= 8) || (c >= 11 && c <= 24); }

static const uint8_t RH_BROADCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static portMUX_TYPE  s_rhMux = portMUX_INITIALIZER_UNLOCKED;
static bool          s_rhReady = false;
static uint8_t       s_rhPeer[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};   // 상대 보드 주소 (알기 전에는 방송)
static volatile bool s_rhPeerSeen = false;    // 수신 콜백이 상대 주소를 알아냄
static uint8_t       s_rhPeerSeenAddr[6];
static bool          s_rhPeerAdded = false;

static bool rhAddPeer(const uint8_t *mac) {
  if (esp_now_is_peer_exist(mac)) return true;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = 0;                 // 현재 채널
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  return esp_now_add_peer(&peer) == ESP_OK;
}

// 수신 콜백에서 알아낸 상대 주소를 1:1 전송 대상으로 등록 (콜백 밖, 각 보드의 poll에서 호출)
static void rhAdoptPeer() {
  if (s_rhPeerAdded || !s_rhPeerSeen) return;
  uint8_t mac[6];
  portENTER_CRITICAL(&s_rhMux);
  memcpy(mac, s_rhPeerSeenAddr, 6);
  portEXIT_CRITICAL(&s_rhMux);
  if (rhAddPeer(mac)) {
    memcpy(s_rhPeer, mac, 6);
    s_rhPeerAdded = true;
    Serial.printf("[LINK] 상대 보드 연결: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  }
}

static bool rhRadioBegin(esp_now_recv_cb_t cb) {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);             // 절전을 켜 두면 수신이 수십~수백 ms 늦어짐
  esp_wifi_set_channel(RH_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  if (esp_now_init() != ESP_OK) {
    Serial.println("[LINK] ESP-NOW 초기화 실패");
    return false;
  }
  if (!rhAddPeer(RH_BROADCAST)) {
    Serial.println("[LINK] 방송 주소 등록 실패");
    return false;
  }
  esp_now_register_recv_cb(cb);
  s_rhReady = true;
  Serial.printf("[LINK] ESP-NOW 준비 완료 (채널 %d), 이 보드 MAC: %s\n", RH_WIFI_CHANNEL,
                WiFi.macAddress().c_str());
  return true;
}

// =====================================================================
// 의수 보드 쪽
// =====================================================================
#ifdef REHAND_ROLE_HAND

#define RH_LINK_TIMEOUT_MS 1000     // 이만큼 명령이 없으면 연결 끊김 (현재 자세 유지)
#define RH_STATUS_MS       200      // 상태 보내는 주기
#ifndef RH_LINK_LED_PIN
#define RH_LINK_LED_PIN    LED_BUILTIN   // XIAO 보드 LED (LOW가 켜짐). 안 쓰려면 -1
#endif

static volatile bool    s_rhHaveCmd = false;
static volatile uint8_t s_rhRxSession = 0, s_rhRxSeq = 0, s_rhRxCmd = 0, s_rhRxStrength = 0;
static volatile uint32_t s_rhLastRxMs = 0;
static uint8_t  s_rhLegacyLast = 0, s_rhLegacySeq = 0;      // 예전 1~2바이트 패킷용 (콜백 안에서만 사용)
static bool     s_rhExecuted = false;
static uint8_t  s_rhDoneSession = 0, s_rhDoneSeq = 0, s_rhDoneCmd = 0;
static bool     s_rhLinkLost = true;
static uint32_t s_rhLastStatusMs = 0;

// 수신 콜백은 와이파이 태스크에서 실행됨: 여기서는 값만 적어 두고, 서보는 loop()에서 움직임
static void rhHandOnRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  uint8_t session, seq, cmd, strength;
  if (len >= (int)sizeof(RhCmdPacket) && data[0] == 'R' && data[1] == 'H' && data[2] == RH_TYPE_CMD) {
    session = data[3]; seq = data[4]; cmd = data[5]; strength = data[6];
  } else if (len >= 1 && len <= 2) {                 // 예전 펌웨어: {명령} 또는 {명령, 힘 세기}
    cmd = data[0];
    strength = (len >= 2) ? data[1] : 0;
    if (cmd != s_rhLegacyLast) { s_rhLegacyLast = cmd; s_rhLegacySeq++; }
    session = 0xFE; seq = s_rhLegacySeq;
  } else {
    return;
  }
  if (!rhValidCmd(cmd)) return;
  if (strength > 100) strength = 100;
  portENTER_CRITICAL(&s_rhMux);
  s_rhRxSession = session; s_rhRxSeq = seq; s_rhRxCmd = cmd; s_rhRxStrength = strength;
  s_rhLastRxMs = millis();
  s_rhHaveCmd = true;
  if (!s_rhPeerSeen && info && info->src_addr) {
    memcpy(s_rhPeerSeenAddr, info->src_addr, 6);
    s_rhPeerSeen = true;
  }
  portEXIT_CRITICAL(&s_rhMux);
}

static void rhHandBegin() {
  if (RH_LINK_LED_PIN >= 0) {
    pinMode(RH_LINK_LED_PIN, OUTPUT);
    digitalWrite(RH_LINK_LED_PIN, LOW);            // 연결 전에는 켜 둠
  }
  rhRadioBegin(rhHandOnRecv);
}

// 매 loop()마다 호출 (호밍 중에도). 끊김 판단과 상태 보내기를 합니다.
static void rhHandPoll(uint8_t flags, uint8_t maxTempC, uint8_t voltage_dV) {
  if (!s_rhReady) return;
  rhAdoptPeer();
  const uint32_t now = millis();
  const bool lost = !s_rhHaveCmd || (now - s_rhLastRxMs > RH_LINK_TIMEOUT_MS);
  if (lost != s_rhLinkLost) {
    s_rhLinkLost = lost;
    Serial.println(lost ? "[LINK] 명령 끊김 - 현재 자세 유지" : "[LINK] 명령 수신 중");
    if (RH_LINK_LED_PIN >= 0) digitalWrite(RH_LINK_LED_PIN, lost ? LOW : HIGH);
  }
  if (now - s_rhLastStatusMs >= RH_STATUS_MS) {
    s_rhLastStatusMs = now;
    RhStatusPacket st = {{'R', 'H'}, RH_TYPE_STATUS, s_rhDoneSession, s_rhDoneSeq, s_rhDoneCmd,
                         (uint8_t)(flags | (lost ? HAND_FLAG_LINK_LOST : 0)), maxTempC, voltage_dV,
                         RH_PROTO_VERSION};
    esp_now_send(s_rhPeer, (const uint8_t *)&st, sizeof(st));
  }
}

// 아직 실행하지 않은 새 명령이 있으면 true. 호밍 중에는 부르지 마세요 (끝난 뒤 가장 최근 것만 실행됨)
static bool rhHandTakeCommand(uint8_t *cmd, uint8_t *strength) {
  if (!s_rhHaveCmd || s_rhLinkLost) return false;
  portENTER_CRITICAL(&s_rhMux);
  const uint8_t session = s_rhRxSession, seq = s_rhRxSeq, c = s_rhRxCmd, s = s_rhRxStrength;
  portEXIT_CRITICAL(&s_rhMux);
  if (s_rhExecuted && session == s_rhDoneSession && seq == s_rhDoneSeq) return false;   // 재전송
  s_rhExecuted = true;
  s_rhDoneSession = session; s_rhDoneSeq = seq; s_rhDoneCmd = c;
  *cmd = c;
  *strength = s;
  return true;
}

static inline uint8_t rhHandStrength() { return s_rhRxStrength; }   // 가장 최근 힘 세기
static inline bool    rhHandLinkLost() { return s_rhLinkLost; }

#endif  // REHAND_ROLE_HAND

// =====================================================================
// 근전도 보드 쪽
// =====================================================================
#ifdef REHAND_ROLE_EMG

#define RH_HAND_TIMEOUT_MS 1500     // 이만큼 의수 상태가 안 오면 의수 꺼짐/멀어짐으로 봄

static uint8_t  s_rhSession = 0, s_rhSeq = 0, s_rhLastCmd = 0;
static volatile uint32_t s_rhStatusMs = 0;
static volatile bool     s_rhStatusSeen = false;
static volatile uint8_t  s_rhHandFlags = 0, s_rhHandTemp = 0, s_rhHandVolt = 0, s_rhHandDone = 0;

static void rhEmgOnRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len < (int)sizeof(RhStatusPacket) || data[0] != 'R' || data[1] != 'H' || data[2] != RH_TYPE_STATUS) return;
  portENTER_CRITICAL(&s_rhMux);
  s_rhHandDone = data[5]; s_rhHandFlags = data[6]; s_rhHandTemp = data[7]; s_rhHandVolt = data[8];
  s_rhStatusMs = millis();
  s_rhStatusSeen = true;
  if (!s_rhPeerSeen && info && info->src_addr) {
    memcpy(s_rhPeerSeenAddr, info->src_addr, 6);
    s_rhPeerSeen = true;
  }
  portEXIT_CRITICAL(&s_rhMux);
}

static bool rhEmgBegin() {
  if (!rhRadioBegin(rhEmgOnRecv)) return false;
  s_rhSession = (uint8_t)(esp_random() & 0xFF);
  if (s_rhSession == 0xFE) s_rhSession = 0x7E;     // 0xFE는 예전 패킷용으로 의수가 쓰는 값
  return true;
}

static void rhEmgTx(uint8_t cmd, uint8_t strength) {
  rhAdoptPeer();
  const RhCmdPacket p = {{'R', 'H'}, RH_TYPE_CMD, s_rhSession, s_rhSeq, cmd, strength, 0};
  esp_now_send(s_rhPeer, (const uint8_t *)&p, sizeof(p));
}

// 현재 명령 보내기 (명령이 바뀌었으면 번호를 올려 의수가 새 명령으로 실행)
static void rhEmgSend(uint8_t cmd, uint8_t strength) {
  if (!s_rhReady) return;
  if (cmd != s_rhLastCmd) { s_rhLastCmd = cmd; s_rhSeq++; }
  rhEmgTx(cmd, strength);
}

// 한 번만 실행돼야 하는 명령 (호밍 등): 같은 번호로 여러 번 보내 손실에 대비, 의수는 한 번만 실행
static void rhEmgSendEvent(uint8_t cmd, int repeat) {
  if (!s_rhReady) return;
  s_rhLastCmd = cmd;
  s_rhSeq++;
  for (int i = 0; i < repeat; ++i) {
    rhEmgTx(cmd, 0);
    vTaskDelay(pdMS_TO_TICKS(30));
  }
}

static inline bool    rhEmgHandAlive() { return s_rhStatusSeen && (millis() - s_rhStatusMs < RH_HAND_TIMEOUT_MS); }
static inline uint8_t rhEmgHandFlags() { return s_rhHandFlags; }
static inline uint8_t rhEmgHandTemp()  { return s_rhHandTemp; }
static inline uint8_t rhEmgHandVolt()  { return s_rhHandVolt; }

#endif  // REHAND_ROLE_EMG
