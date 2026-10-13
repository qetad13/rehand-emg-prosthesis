// 이 파일은 고치지 마세요. 모드는 emg_mode.h에서 고릅니다.
#ifndef EMG_MODEL_SELECT_H
#define EMG_MODEL_SELECT_H

#include "emg_mode.h"

#if EMG_FAST_MODE
  #include "emg_model_fast.h"
  #define EMG_MODE_NAME "빠른 모드 (0.5초 창, 50ms마다 판단)"
#else
  #include "emg_model_accurate.h"
  #define EMG_MODE_NAME "정확 모드 (1초 창, 0.5초마다 판단)"
#endif

#endif  // EMG_MODEL_SELECT_H
