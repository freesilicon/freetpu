HOST_CC  ?= clang
HOST_CXX ?= clang++
HOST_AR  ?= ar

STATIC_LIBSTDCPP ?= 0
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHLIB_EXT := dylib
  SHLIB_FLAGS := -dynamiclib
  EXE_LDFLAGS :=
else
  SHLIB_EXT := so
  ifeq ($(STATIC_LIBSTDCPP),1)
    SHLIB_FLAGS := -shared -static-libstdc++ -static-libgcc -Wl,--as-needed -Wl,--exclude-libs,ALL
    EXE_LDFLAGS := -static-libstdc++ -static-libgcc -Wl,--as-needed
  else
    SHLIB_FLAGS := -shared -Wl,--as-needed
    EXE_LDFLAGS := -Wl,--as-needed
  endif
endif

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
INCS     := -I$(THIS_DIR)/include -I$(THIS_DIR)/src -I$(THIS_DIR)/third_party/openxla
CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra -Werror -fno-exceptions -fno-rtti $(INCS)
LDFLAGS  := -lm
ifeq ($(UNAME_S),Linux)
  LDFLAGS += -lpthread -ldl
endif

SRCS := \
	$(THIS_DIR)/src/ot_vfio.cpp \
	$(THIS_DIR)/src/ot_device.cpp \
	$(THIS_DIR)/src/ot_pjrt.cpp

HDRS := \
	$(THIS_DIR)/include/opentpu.h \
	$(THIS_DIR)/include/opentpu_freetile.hpp \
	$(THIS_DIR)/src/ot_internal.h \
	$(THIS_DIR)/src/ot_vliw.h \
	$(THIS_DIR)/third_party/openxla/pjrt_c_api.h

BUILD_DIR := $(THIS_DIR)/build/host
OBJS      := $(patsubst $(THIS_DIR)/src/%.cpp,$(BUILD_DIR)/obj/%.o,$(SRCS))
SHLIB     := $(BUILD_DIR)/libopentpu.$(SHLIB_EXT)
STLIB     := $(BUILD_DIR)/libopentpu.a
HOST_TEST := $(BUILD_DIR)/opentpu_host_test
SMOKE_BIN := $(BUILD_DIR)/opentpu_vfio_smoke
PROOF_BIN := $(BUILD_DIR)/opentpu_matmul_proof

.PHONY: all host test verify-header-host clean

all: host

host: $(SHLIB) $(STLIB) $(HOST_TEST) $(SMOKE_BIN) $(PROOF_BIN)

$(BUILD_DIR)/obj/%.o: $(THIS_DIR)/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) -c $< -o $@

$(SHLIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(SHLIB_FLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(STLIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(HOST_AR) rcs $@ $(OBJS)

$(HOST_TEST): $(THIS_DIR)/tests/opentpu_host_test.cpp $(STLIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(STLIB) -o $@ $(LDFLAGS)

$(SMOKE_BIN): $(THIS_DIR)/tools/opentpu_vfio_smoke.cpp $(SHLIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(SHLIB) -Wl,-rpath,$(BUILD_DIR) -o $@ $(LDFLAGS)

$(PROOF_BIN): $(THIS_DIR)/tools/opentpu_matmul_proof.cpp $(SHLIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< -o $@ $(EXE_LDFLAGS) $(LDFLAGS)

test: verify-header-host $(HOST_TEST) $(SMOKE_BIN) $(PROOF_BIN)
	$(HOST_TEST)
	$(SMOKE_BIN) --mock
	$(PROOF_BIN) --mock --lib $(SHLIB)

verify-header-host: $(THIS_DIR)/include/opentpu.h
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror $<
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror $<

clean:
	rm -rf $(THIS_DIR)/build
