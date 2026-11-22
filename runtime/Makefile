NDK_BIN ?= /opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/bin
ifeq ($(origin CC),default)
  CC := $(NDK_BIN)/aarch64-linux-android34-clang
endif
ifeq ($(origin CXX),default)
  CXX := $(NDK_BIN)/aarch64-linux-android34-clang++
endif
READELF ?= $(NDK_BIN)/llvm-readelf
NM      ?= $(NDK_BIN)/llvm-nm

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT     := $(abspath $(THIS_DIR)/..)

INCS := -I$(ROOT)/docs/baremetal-vii -I$(ROOT)/docs/kernel-abi -I$(THIS_DIR)/src
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -fPIE $(INCS)
BAREMETAL_LDFLAGS := -pie -static-libstdc++ -Wl,--as-needed

BM_OUT := $(THIS_DIR)/build/tpu_baremetal_runner
BM_SRC := $(THIS_DIR)/src/tpu_baremetal_runner.cpp
BM_HDRS := $(THIS_DIR)/src/json_util.hpp \
           $(ROOT)/docs/baremetal-vii/litebuf_vii_wire.h \
           $(ROOT)/docs/kernel-abi/edgetpu_buenos_uapi.h \
           $(ROOT)/docs/kernel-abi/gcip_uapi.h \
           $(ROOT)/docs/kernel-abi/gxp_laguna_uapi.h

TR_OUT := $(THIS_DIR)/build/freetpu_tile_runner
TR_HOST_OUT := $(THIS_DIR)/build/freetpu_tile_runner_host
TR_SRC := $(THIS_DIR)/src/freetpu_tile_runner.cpp
OB_ANDROID_LIB := $(ROOT)/libopenbuenos/build/android-arm64/libopenbuenos.a
OB_HOST_LIB := $(ROOT)/libopenbuenos/build/host/libopenbuenos.a

.PHONY: all clean verify-headers verify-baremetal tile-runner tile-runner-host verify-tile-runner FORCE

FORCE:

all: $(BM_OUT)

$(BM_OUT): $(BM_SRC) $(BM_HDRS)
	@mkdir -p $(dir $(BM_OUT))
	$(CXX) $(CXXFLAGS) $(BM_SRC) -o $(BM_OUT) $(BAREMETAL_LDFLAGS)

$(OB_ANDROID_LIB): FORCE
	$(MAKE) -C $(ROOT)/libopenbuenos android

$(OB_HOST_LIB): FORCE
	$(MAKE) -C $(ROOT)/libopenbuenos host

tile-runner: $(TR_OUT)

$(TR_OUT): $(TR_SRC) $(ROOT)/libopenbuenos/include/openbuenos.h $(OB_ANDROID_LIB)
	@mkdir -p $(dir $(TR_OUT))
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -fPIE -I$(ROOT)/libopenbuenos/include $(TR_SRC) $(OB_ANDROID_LIB) -o $(TR_OUT) $(BAREMETAL_LDFLAGS) -lm

tile-runner-host: $(TR_HOST_OUT)

$(TR_HOST_OUT): $(TR_SRC) $(ROOT)/libopenbuenos/include/openbuenos.h $(OB_HOST_LIB)
	@mkdir -p $(dir $(TR_HOST_OUT))
	clang++ -std=c++17 -O2 -Wall -Wextra -Werror -I$(ROOT)/libopenbuenos/include $(TR_SRC) $(OB_HOST_LIB) -o $(TR_HOST_OUT) -lm

verify-tile-runner: $(TR_OUT)
	file $(TR_OUT)
	$(READELF) -d $(TR_OUT)
	! $(READELF) -d $(TR_OUT) | grep -iE 'lib(log|tachyon|edgetpu|darwinn|gxp|tflite|tensorflow)|/vendor/'
	! $(NM) -u $(TR_OUT) | grep -E '\bdlopen\b|\bdlsym\b|\bdlclose\b|\bdlerror\b'

verify-headers:
	$(CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $(INCS) $(ROOT)/docs/baremetal-vii/litebuf_vii_wire.h
	$(CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror $(INCS) $(ROOT)/docs/baremetal-vii/litebuf_vii_wire.h
	$(CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $(INCS) $(ROOT)/docs/kernel-abi/gcip_uapi.h
	$(CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $(INCS) $(ROOT)/docs/kernel-abi/edgetpu_buenos_uapi.h
	$(CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $(INCS) $(ROOT)/docs/kernel-abi/gxp_laguna_uapi.h

verify-baremetal: $(BM_OUT) verify-headers
	file $(BM_OUT)
	$(READELF) -d $(BM_OUT)
	! $(READELF) -d $(BM_OUT) | grep -iE 'lib(log|tachyon|edgetpu|darwinn|gxp|tflite|tensorflow)|/vendor/'
	! $(NM) -u $(BM_OUT) | grep -E '\bdlopen\b|\bdlsym\b|\bdlclose\b|\bdlerror\b'

clean:
	rm -rf $(THIS_DIR)/build
