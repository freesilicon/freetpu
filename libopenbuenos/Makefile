NDK_BIN     ?= /opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/bin
ANDROID_CC  ?= $(NDK_BIN)/aarch64-linux-android34-clang
ANDROID_CXX ?= $(NDK_BIN)/aarch64-linux-android34-clang++
ANDROID_AR  ?= $(NDK_BIN)/llvm-ar
READELF     ?= $(NDK_BIN)/llvm-readelf
NM          ?= $(NDK_BIN)/llvm-nm

HOST_CC  ?= clang
HOST_CXX ?= clang++
HOST_AR  ?= ar

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT     := $(abspath $(THIS_DIR)/..)

INCS := -I$(THIS_DIR)/include -I$(THIS_DIR)/src -I$(ROOT)/freetile/include -I$(ROOT)/docs/baremetal-vii -I$(ROOT)/docs/kernel-abi
COMMON_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -DOPENBUENOS_WITH_FREETILE=1 -DOPENBUENOS_FREETILE_DEFAULT_ON_VAL=0 $(INCS)
ANDROID_CXXFLAGS := $(COMMON_CXXFLAGS) -fPIE
ANDROID_LDFLAGS  := -pie -static-libstdc++ -Wl,--as-needed -lm
HOST_CXXFLAGS    := $(COMMON_CXXFLAGS)
HOST_LDFLAGS     := -lm -lpthread

SRCS := \
	$(THIS_DIR)/src/ob_sha256.cpp \
	$(THIS_DIR)/src/ob_dgc0.cpp \
	$(THIS_DIR)/src/ob_buffer_pool.cpp \
	$(THIS_DIR)/src/ob_worker_pool.cpp \
	$(THIS_DIR)/src/ob_device.cpp \
	$(THIS_DIR)/src/ob_graph.cpp \
	$(THIS_DIR)/src/ob_freetile_bridge.cpp \
	$(THIS_DIR)/src/ob_quant_unpack.cpp \
	$(THIS_DIR)/src/ob_gemm_emitter.cpp \
	$(THIS_DIR)/src/ob_gemm.cpp

FREETILE_SRCS := \
	$(ROOT)/freetile/src/tile_ir.cpp \
	$(ROOT)/freetile/src/buenos_emitter.cpp

HDRS := \
	$(THIS_DIR)/include/openbuenos.h \
	$(THIS_DIR)/src/ob_internal.h \
	$(THIS_DIR)/src/ob_quant_unpack.hpp \
	$(ROOT)/freetile/include/freetile/tile_ir.hpp \
	$(ROOT)/freetile/include/freetile/buenos_emitter.hpp \
	$(ROOT)/docs/baremetal-vii/litebuf_vii_wire.h \
	$(ROOT)/docs/kernel-abi/edgetpu_buenos_uapi.h

HOST_DIR    := $(THIS_DIR)/build/host
ANDROID_DIR := $(THIS_DIR)/build/android-arm64

HOST_OBJS    := $(patsubst $(THIS_DIR)/src/%.cpp,$(HOST_DIR)/obj/%.o,$(SRCS)) \
                $(patsubst $(ROOT)/freetile/src/%.cpp,$(HOST_DIR)/obj/freetile_%.o,$(FREETILE_SRCS))
ANDROID_OBJS := $(patsubst $(THIS_DIR)/src/%.cpp,$(ANDROID_DIR)/obj/%.o,$(SRCS)) \
                $(patsubst $(ROOT)/freetile/src/%.cpp,$(ANDROID_DIR)/obj/freetile_%.o,$(FREETILE_SRCS))

HOST_LIB     := $(HOST_DIR)/libopenbuenos.a
HOST_TEST    := $(HOST_DIR)/openbuenos_host_test
QUANT_TEST   := $(HOST_DIR)/ob_quant_unpack_test
ANDROID_LIB  := $(ANDROID_DIR)/libopenbuenos.a
ANDROID_TEST := $(ANDROID_DIR)/openbuenos_selftest

.PHONY: all host android test verify-header verify-header-host verify-android verify clean

all: host android

host: $(HOST_LIB) $(HOST_TEST) $(QUANT_TEST)

android: $(ANDROID_LIB) $(ANDROID_TEST)

$(HOST_DIR)/obj/%.o: $(THIS_DIR)/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) -c $< -o $@

$(HOST_DIR)/obj/freetile_%.o: $(ROOT)/freetile/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) -c $< -o $@

$(HOST_LIB): $(HOST_OBJS)
	@mkdir -p $(dir $@)
	$(HOST_AR) rcs $@ $(HOST_OBJS)

$(HOST_TEST): $(THIS_DIR)/tests/openbuenos_host_test.cpp $(HOST_LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< $(HOST_LIB) -o $@ $(HOST_LDFLAGS)

$(QUANT_TEST): $(THIS_DIR)/tests/ob_quant_unpack_test.cpp $(HOST_LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $< $(HOST_LIB) -o $@ $(HOST_LDFLAGS)

$(ANDROID_DIR)/obj/%.o: $(THIS_DIR)/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(ANDROID_CXX) $(ANDROID_CXXFLAGS) -c $< -o $@

$(ANDROID_DIR)/obj/freetile_%.o: $(ROOT)/freetile/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(ANDROID_CXX) $(ANDROID_CXXFLAGS) -c $< -o $@

$(ANDROID_LIB): $(ANDROID_OBJS)
	@mkdir -p $(dir $@)
	$(ANDROID_AR) rcs $@ $(ANDROID_OBJS)

$(ANDROID_TEST): $(THIS_DIR)/tests/openbuenos_selftest.cpp $(ANDROID_LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(ANDROID_CXX) $(ANDROID_CXXFLAGS) $< $(ANDROID_LIB) -o $@ $(ANDROID_LDFLAGS)

test: $(HOST_TEST) $(QUANT_TEST)
	$(HOST_TEST)
	$(QUANT_TEST)

verify-header-host: $(THIS_DIR)/include/openbuenos.h
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $<
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror $<

verify-header: $(THIS_DIR)/include/openbuenos.h verify-header-host
	$(ANDROID_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $<
	$(ANDROID_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror $<

verify-android: $(ANDROID_TEST) $(ANDROID_LIB)
	file $(ANDROID_TEST)
	$(READELF) -d $(ANDROID_TEST)
	! $(READELF) -d $(ANDROID_TEST) | grep -iE 'lib(log|tachyon|edgetpu|darwinn|gxp|tflite|tensorflow)|/vendor/'
	! $(NM) -u $(ANDROID_TEST) | grep -E '\bdlopen\b|\bdlsym\b|\bdlclose\b|\bdlerror\b'

verify: verify-header test verify-android

clean:
	rm -rf $(THIS_DIR)/build
