// 이 파일은 고치지 마세요. 모드는 emg_mode.h에서 고릅니다.
#ifndef EMG_TEST_VECTORS_SELECT_H
#define EMG_TEST_VECTORS_SELECT_H

#include "emg_mode.h"

#if EMG_FAST_MODE
  #include "emg_test_vectors_fast.h"
#else
  #include "emg_test_vectors_accurate.h"
#endif

#endif  // EMG_TEST_VECTORS_SELECT_H
