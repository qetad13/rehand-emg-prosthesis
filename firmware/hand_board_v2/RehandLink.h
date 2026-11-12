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
// - 연결 상대 고정: 한 번 연결되면 그 보드에서 온 패킷만 받고, 다른 보드에서 온 것은 버립니다.
//     의수      처음 명령을 보낸 근전도 보드의 주소를 보드에 저장 (전원을 껐다 켜도 유지).
//               다른 근전도 보드로 바꾸려면 rhHandUnpair() (의수 시리얼 'p' 또는 BOOT 버튼 3초)
//     근전도    처음 상태를 보낸 의수와 연결 (저장하지 않음). 의수가 1.5초 넘게 조용하면 다시 찾음
//   주소 확인만 하고 암호화는 하지 않으므로, 주소를 일부러 흉내 내는 송신기까지 막지는 못합니다.
// - 의수는 예전 근전도 펌웨어의 1~2바이트 패킷 {명령, 힘 세기}도 그대로 받습니다 (연결된 보드에서 온 것만).
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
static uint8_t       s_rhPeer[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};   // 보내는 주소 (상대를 알기 전에는 방송)

// 연결 상대. 수신 콜백(와이파이 태스크)과 poll이 함께 쓰므로 s_rhMux 안에서만 읽고 씁니다.
static uint8_t       s_rhMate[6];              // 연결된 상대 보드 주소
static bool          s_rhMateSet = false;      // true면 s_rhMate가 아닌 주소에서 온 패킷은 버림
static volatile bool s_rhMateDirty = false;    // 상대가 (새로) 정해짐 -> poll에서 1:1 전송 대상으로 등록
static volatile uint32_t s_rhRejected = 0;     // 연결되지 않은 보드에서 와서 버린 패킷 수

static inline bool rhIsBroadcast(const uint8_t *m) { return memcmp(m, RH_BROADCAST, 6) == 0; }

static bool rhAddPeer(const uint8_t *mac) {
  if (esp_now_is_peer_exist(mac)) return true;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = 0;                 // 현재 채널
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  return esp_now_add_peer(&peer) == ESP_OK;
}

// 보내는 주소를 다시 방송으로 (콜백 밖에서만 호출)
static void rhSendToBroadcast() {
  if (!rhIsBroadcast(s_rhPeer)) esp_now_del_peer(s_rhPeer);
  memcpy(s_rhPeer, RH_BROADCAST, 6);
}

// 수신 콜백이 정한 상대 주소를 1:1 전송 대상으로 등록 (콜백 밖, 각 보드의 poll에서 호출). 새로 등록했으면 true
static bool rhAdoptPeer() {
  if (!s_rhMateDirty) return false;
  uint8_t mac[6];
  portENTER_CRITICAL(&s_rhMux);
  const bool want = s_rhMateDirty && s_rhMateSet;
  memcpy(mac, s_rhMate, 6);
  s_rhMateDirty = false;
  portEXIT_CRITICAL(&s_rhMux);
  if (!want) return false;
  if (memcmp(mac, s_rhPeer, 6) == 0) return false;      // 이미 그 주소로 보내는 중
  if (!rhAddPeer(mac)) {
    s_rhMateDirty = true;                               // 다음 poll에서 다시 시도
    return false;
  }
  rhSendToBroadcast();                                  // 예전 상대가 있었으면 지움
  memcpy(s_rhPeer, mac, 6);
  Serial.printf("[LINK] 상대 보드 연결: %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  return true;
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
#include <Preferences.h>

#define RH_LINK_TIMEOUT_MS 1000     // 이만큼 명령이 없으면 연결 끊김 (현재 자세 유지)
#define RH_STATUS_MS       200      // 상태 보내는 주기
#define RH_PAIR_NS         "rhlink" // 연결 상대 주소를 저장하는 곳 (NVS)
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
static uint32_t s_rhRejLogged = 0, s_rhRejLogMs = 0;

static bool rhHandLoadMate(uint8_t *mac) {
  Preferences p;
  if (!p.begin(RH_PAIR_NS, false)) return false;
  const bool ok = p.isKey("mate") && p.getBytes("mate", mac, 6) == 6 && !rhIsBroadcast(mac);
  p.end();
  return ok;
}

static void rhHandSaveMate(const uint8_t *mac) {
  uint8_t old[6];
  if (rhHandLoadMate(old) && memcmp(old, mac, 6) == 0) return;   // 이미 저장된 값이면 다시 쓰지 않음
  Preferences p;
  if (!p.begin(RH_PAIR_NS, false)) return;
  p.putBytes("mate", mac, 6);
  p.end();
  Serial.println("[LINK] 이 근전도 보드를 연결 상대로 저장 (다른 보드의 명령은 무시). 바꾸려면 시리얼 p 또는 BOOT 3초");
}

// 수신 콜백은 와이파이 태스크에서 실행됨: 여기서는 값만 적어 두고, 서보는 loop()에서 움직임
static void rhHandOnRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  uint8_t session = 0, seq = 0, cmd, strength;
  bool legacy = false;
  if (len >= (int)sizeof(RhCmdPacket) && data[0] == 'R' && data[1] == 'H' && data[2] == RH_TYPE_CMD) {
    session = data[3]; seq = data[4]; cmd = data[5]; strength = data[6];
  } else if (len >= 1 && len <= 2) {                 // 예전 펌웨어: {명령} 또는 {명령, 힘 세기}
    cmd = data[0];
    strength = (len >= 2) ? data[1] : 0;
    legacy = true;
  } else {
    return;
  }
  if (!rhValidCmd(cmd)) return;
  if (!info || !info->src_addr) return;
  if (strength > 100) strength = 100;
  portENTER_CRITICAL(&s_rhMux);
  if (!s_rhMateSet) {                                // 아직 연결 상대가 없음: 처음 명령을 보낸 보드와 연결
    memcpy(s_rhMate, info->src_addr, 6);
    s_rhMateSet = true;
    s_rhMateDirty = true;
  } else if (memcmp(info->src_addr, s_rhMate, 6) != 0) {   // 연결되지 않은 보드의 명령은 버림
    s_rhRejected = s_rhRejected + 1;
    portEXIT_CRITICAL(&s_rhMux);
    return;
  }
  if (legacy) {
    if (cmd != s_rhLegacyLast) { s_rhLegacyLast = cmd; s_rhLegacySeq++; }
    session = 0xFE; seq = s_rhLegacySeq;
  }
  s_rhRxSession = session; s_rhRxSeq = seq; s_rhRxCmd = cmd; s_rhRxStrength = strength;
  s_rhLastRxMs = millis();
  s_rhHaveCmd = true;
  portEXIT_CRITICAL(&s_rhMux);
}

static void rhHandBegin() {
  if (RH_LINK_LED_PIN >= 0) {
    pinMode(RH_LINK_LED_PIN, OUTPUT);
    digitalWrite(RH_LINK_LED_PIN, LOW);            // 연결 전에는 켜 둠
  }
  uint8_t mac[6];
  if (rhHandLoadMate(mac)) {                       // 저장된 연결 상대가 있으면 그 보드의 명령만 받음
    memcpy(s_rhMate, mac, 6);
    s_rhMateSet = true;
    s_rhMateDirty = true;
    Serial.printf("[LINK] 저장된 연결 상대 %02X:%02X:%02X:%02X:%02X:%02X 의 명령만 받습니다 (바꾸려면 시리얼 p 또는 BOOT 3초)\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  } else {
    Serial.println("[LINK] 저장된 연결 상대 없음 -> 처음 명령을 보내는 근전도 보드와 연결합니다");
  }
  rhRadioBegin(rhHandOnRecv);
}

// 연결 상대를 지움 (다른 근전도 보드로 바꿀 때). 그다음 처음 명령을 보내는 보드와 다시 연결됩니다.
// 예전 근전도 보드가 켜져 있으면 바로 그 보드와 다시 연결되므로, 예전 보드는 꺼 두고 실행하세요. loop()에서 호출
static void rhHandUnpair() {
  Preferences p;
  if (p.begin(RH_PAIR_NS, false)) {
    if (p.isKey("mate")) p.remove("mate");
    p.end();
  }
  portENTER_CRITICAL(&s_rhMux);
  s_rhMateSet = false;
  s_rhMateDirty = false;
  s_rhHaveCmd = false;
  portEXIT_CRITICAL(&s_rhMux);
  rhSendToBroadcast();
  Serial.println("[LINK] 연결 상대를 지웠습니다 -> 처음 명령을 보내는 근전도 보드와 다시 연결합니다");
}

// 매 loop()마다 호출 (호밍 중에도). 끊김 판단과 상태 보내기를 합니다.
static void rhHandPoll(uint8_t flags, uint8_t maxTempC, uint8_t voltage_dV) {
  if (!s_rhReady) return;
  if (rhAdoptPeer()) rhHandSaveMate(s_rhPeer);
  const uint32_t now = millis();
  const bool lost = !s_rhHaveCmd || (now - s_rhLastRxMs > RH_LINK_TIMEOUT_MS);
  if (lost != s_rhLinkLost) {
    s_rhLinkLost = lost;
    Serial.println(lost ? "[LINK] 명령 끊김 - 현재 자세 유지" : "[LINK] 명령 수신 중");
    if (RH_LINK_LED_PIN >= 0) digitalWrite(RH_LINK_LED_PIN, lost ? LOW : HIGH);
  }
  const uint32_t rej = s_rhRejected;
  if (rej != s_rhRejLogged && now - s_rhRejLogMs >= 5000) {     // 5초에 한 번만 알림
    s_rhRejLogged = rej;
    s_rhRejLogMs = now;
    Serial.printf("[LINK] 연결되지 않은 보드에서 온 명령을 무시했습니다 (누적 %lu개)\n", (unsigned long)rej);
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
static inline uint32_t rhHandRejected() { return s_rhRejected; }    // 연결되지 않은 보드에서 와서 버린 명령 수

// 연결 상대 주소를 mac에 적고 true (없으면 false)
static bool rhHandMate(uint8_t *mac) {
  portENTER_CRITICAL(&s_rhMux);
  const bool set = s_rhMateSet;
  memcpy(mac, s_rhMate, 6);
  portEXIT_CRITICAL(&s_rhMux);
  return set;
}

#endif  // REHAND_ROLE_HAND

// =====================================================================
// 근전도 보드 쪽
// =====================================================================
#ifdef REHAND_ROLE_EMG

#define RH_HAND_TIMEOUT_MS 1500     // 이만큼 의수 상태가 안 오면 의수 꺼짐/멀어짐으로 보고 다시 찾음

static uint8_t  s_rhSession = 0, s_rhSeq = 0, s_rhLastCmd = 0;
static volatile uint32_t s_rhStatusMs = 0;
static volatile bool     s_rhStatusSeen = false;
static volatile uint8_t  s_rhHandFlags = 0, s_rhHandTemp = 0, s_rhHandVolt = 0, s_rhHandDone = 0;
// 의수가 이 보드 주소로 직접(1:1) 보낸 상태를 받았음 = 그 의수가 이 보드를 연결 상대로 삼고 있음 (s_rhMux 안에서 사용)
static bool     s_rhMateDirect = false;

static void rhEmgOnRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len < (int)sizeof(RhStatusPacket) || data[0] != 'R' || data[1] != 'H' || data[2] != RH_TYPE_STATUS) return;
  if (!info || !info->src_addr) return;
  const bool direct = info->des_addr && !rhIsBroadcast(info->des_addr);
  portENTER_CRITICAL(&s_rhMux);
  if (!s_rhMateSet) {                                // 아직 연결된 의수가 없음: 처음 상태를 보낸 의수와 연결
    memcpy(s_rhMate, info->src_addr, 6);
    s_rhMateSet = true;
    s_rhMateDirty = true;
    s_rhMateDirect = direct;
  } else if (memcmp(info->src_addr, s_rhMate, 6) == 0) {
    if (direct) s_rhMateDirect = true;
  } else if (direct && !s_rhMateDirect) {
    // 방송만 듣고 잡았던 의수보다, 이 보드를 연결 상대로 저장해 둔 의수가 우선
    memcpy(s_rhMate, info->src_addr, 6);
    s_rhMateDirty = true;
    s_rhMateDirect = true;
  } else {                                           // 연결되지 않은 의수의 상태는 버림
    s_rhRejected = s_rhRejected + 1;
    portEXIT_CRITICAL(&s_rhMux);
    return;
  }
  s_rhHandDone = data[5]; s_rhHandFlags = data[6]; s_rhHandTemp = data[7]; s_rhHandVolt = data[8];
  s_rhStatusMs = millis();
  s_rhStatusSeen = true;
  portEXIT_CRITICAL(&s_rhMux);
}

static bool rhEmgBegin() {
  if (!rhRadioBegin(rhEmgOnRecv)) return false;
  s_rhSession = (uint8_t)(esp_random() & 0xFF);
  if (s_rhSession == 0xFE) s_rhSession = 0x7E;     // 0xFE는 예전 패킷용으로 의수가 쓰는 값
  return true;
}

static void rhEmgTx(uint8_t cmd, uint8_t strength) {
  // 연결된 의수가 한동안 조용하면(꺼짐/교체) 연결을 풀고 다시 방송으로 찾음.
  // 같은 의수가 다시 켜지면 방송으로 보낸 명령을 받고 응답하므로 바로 다시 연결됩니다.
  bool released = false;
  portENTER_CRITICAL(&s_rhMux);
  if (s_rhMateSet && (millis() - s_rhStatusMs > RH_HAND_TIMEOUT_MS)) {
    s_rhMateSet = false;
    s_rhMateDirty = false;
    s_rhMateDirect = false;
    released = true;
  }
  portEXIT_CRITICAL(&s_rhMux);
  if (released) {
    rhSendToBroadcast();
    Serial.println("[LINK] 의수 응답 없음 -> 다시 찾는 중");
  }
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
