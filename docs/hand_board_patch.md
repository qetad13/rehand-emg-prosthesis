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
