# 의수 보드 코드 수정 (ESP-NOW 명령 실행)

현재 의수 코드는 ESP-NOW로 명령을 **받기만 하고 실행하지 않습니다.**
`onEspNowReceive()`가 `g_newEspNowCommand`를 true로 바꾸지만, `loop()`에서 이걸 확인하는 곳이 없습니다.

`loop()`를 아래처럼 바꾸면 됩니다. (추가된 부분은 가운데 블록 하나)

```cpp
void loop() {
  static uint32_t last_cmd_ms = 0;

  if (HOMING_isBusy()) {
    while (Serial.available()) { Serial.read(); }
    vTaskDelay(pdMS_TO_TICKS(5));
    return;
  }

  // ---- ESP-NOW로 받은 명령 실행 (근전도 보드가 0.5초마다 보냄) ----
  if (g_newEspNowCommand) {
    g_newEspNowCommand = false;
    static uint8_t lastEspNowCmd = 0;
    uint8_t cmd = g_receivedCommand;
    if (cmd != lastEspNowCmd) {      // 같은 명령이 반복해서 오면 무시
      lastEspNowCmd = cmd;
      Serial.print("[ESP-NOW] Received command: ");
      Serial.println(cmd);
      executeHandCommand(cmd);
    }
  }

  checkAndEnforceSoftLimits();
  checkSerialHandCommand();
  vTaskDelay(pdMS_TO_TICKS(5));
}
```

참고: 명령 실행을 수신 콜백(`onEspNowReceive`) 안에서 하지 않고 `loop()`에서 하는 이유는,
콜백은 WiFi 태스크에서 돌기 때문에 거기서 서보 버스(시리얼, 뮤텍스)를 쓰면 WiFi가 멈출 수 있어서입니다.

## 근전도 보드가 보내는 명령 (9/24 기준, 8가지)

| 근전도 판단 | 명령 번호 |
|---|---|
| 엄지 | 1 |
| 검지 | 2 |
| 중지 | 3 |
| 약지 | 4 |
| 소지 | 5 |
| 휴식 | 6 |
| 주먹 | 7 |
| 집게 | 8 |

**주의:** 의수 코드의 수신 콜백 `onEspNowReceive()`에 `if (cmd >= 1 && cmd <= 6)`가 있으면
7, 8이 무시됩니다. `cmd <= 8`로 바꿔주세요. 시리얼 테스트용 `checkSerialHandCommand()`도 `'8'`까지 받게 바꾸면 편합니다.

번호를 바꾸려면 근전도 보드의 `esp32_emg_realtime.ino`에서 `handCmdFor()`만 수정하면 됩니다.

## 힘 세기 받기 (세게 = 빠르게)

근전도 보드가 이제 **2바이트 {명령 번호, 힘 세기}** 를 보냅니다. (판단할 때마다, 빠른 모드 기준 50ms마다)

| 힘 세기 값 | 의미 |
|---|---|
| 1~100 | 손가락 동작 중. 클수록 세게 힘을 준 것 |
| 0 | 세기 정보 없음 (휴식, 매크로 모드) -> 기본 속도로 움직이면 됨 |

의수 코드에서 바꿀 곳은 세 군데입니다. (변수/함수 이름은 실제 의수 코드에 맞춰 주세요)

```cpp
volatile uint8_t g_receivedStrength = 0;

// 1) 수신 콜백: 길이가 1이어도 2여도 받도록
//    (len == 1 인지 검사하는 줄이 있으면 len >= 1 로 바꾸기)
g_receivedCommand  = data[0];
g_receivedStrength = (len >= 2) ? data[1] : 0;
g_newEspNowCommand = true;

// 2) 세기 -> 서보 속도
//    SPEED_MIN/MAX/DEFAULT는 지금 서보 위치 명령에 넣고 있는 속도 값 기준으로 정하기
uint16_t speedFromStrength(uint8_t s) {
  if (s == 0) return SPEED_DEFAULT;
  return SPEED_MIN + (uint32_t)(SPEED_MAX - SPEED_MIN) * s / 100;
}

// 3) loop(): 같은 명령이 반복해서 와도 "속도"는 계속 갱신
if (g_newEspNowCommand) {
  g_newEspNowCommand = false;
  static uint8_t lastEspNowCmd = 0;
  uint8_t cmd = g_receivedCommand;
  g_handSpeed = speedFromStrength(g_receivedStrength);   // 매번 갱신
  if (cmd != lastEspNowCmd) {
    lastEspNowCmd = cmd;
    executeHandCommand(cmd);          // 안에서 서보 위치 명령을 보낼 때 g_handSpeed 사용
  } else {
    updateHandSpeed();                // 움직이는 중인 서보에 같은 목표 위치 + 새 속도를 다시 보냄
  }
}
```

**주의: 명령이 처음 바뀐 순간의 세기 값 하나로 속도를 고정하지 마세요.**
동작이 인식되는 순간에는 근전도가 아직 커지는 중이라, 같은 힘으로 줘도 첫 값이 10~100까지 들쭉날쭉합니다
(녹화 데이터로 확인). 0.1~0.2초 뒤에 제 값이 되므로, 움직이는 동안 들어오는 값으로 속도를 계속 바꿔야 합니다.

근전도 보드 쪽 설정은 `esp32_emg_realtime.ino`의 "힘 세기" 부분에 있습니다.
(`STRENGTH_LEVELS`를 2나 3으로 바꾸면 연속값 대신 약/강, 약/중/강 단계 값으로 보냄)

## 매크로 모드 명령 (11~19)

근전도 보드에서 SW1로 매크로 모드에 들어가면, 근전도 판단 대신 SW2로 고른 동작 번호(11~19)를 판단 주기마다 보냅니다.
의수 보드의 수신 조건이 `cmd <= 8`이면 이 명령들이 무시되므로, `11 <= cmd <= 19`도 받도록 넓히고 번호별 동작을 `executeHandCommand()`에 추가해야 합니다.
매크로 이름과 번호는 근전도 보드 `esp32_emg_realtime.ino`의 `MACROS[]`에서 바꿀 수 있습니다.
