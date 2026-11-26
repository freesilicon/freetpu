HOST_CC  ?= clang
HOST_CXX ?= clang++
HOST_AR  ?= ar

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT     := $(abspath $(THIS_DIR)/..)
OB_DIR   := $(ROOT)/libopenbuenos
OB_LIB   := $(OB_DIR)/build/host/libopenbuenos.a

BUILD_DIR    := $(THIS_DIR)/build/host
ROOT_BIN_DIR := $(ROOT)/build

INCS := -I$(THIS_DIR)/include
OB_INCS := \
	-I$(OB_DIR)/include \
	-I$(OB_DIR)/src \
	-I$(ROOT)/docs/baremetal-vii \
	-I$(ROOT)/docs/kernel-abi

CXXFLAGS_BASE  := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror $(INCS)
CXXFLAGS_NOEXC := $(CXXFLAGS_BASE) -fno-exceptions -fno-rtti
LDFLAGS        := -lm -lpthread

CORE_SRCS := \
	$(THIS_DIR)/src/tile_ir.cpp \
	$(THIS_DIR)/src/mosaic_emitter.cpp \
	$(THIS_DIR)/src/buenos_emitter.cpp \
	$(THIS_DIR)/src/buenos_splines.cpp \
	$(THIS_DIR)/src/flash_attn_planner.cpp \
	$(THIS_DIR)/src/malibu_profile.cpp

CORE_HDRS := \
	$(THIS_DIR)/include/freetile/tile_ir.hpp \
	$(THIS_DIR)/include/freetile/mosaic_emitter.hpp \
	$(THIS_DIR)/include/freetile/buenos_emitter.hpp \
	$(THIS_DIR)/include/freetile/buenos_splines.hpp \
	$(THIS_DIR)/include/freetile/flash_attn_planner.hpp \
	$(THIS_DIR)/include/freetile/malibu_profile.hpp

TWIN_HDR := $(THIS_DIR)/include/freetile/twin.h
TWIN_SRC := $(THIS_DIR)/src/twin.cpp

CORE_OBJS := $(patsubst $(THIS_DIR)/src/%.cpp,$(BUILD_DIR)/obj/%.o,$(CORE_SRCS))
TWIN_OBJ  := $(BUILD_DIR)/obj/twin.o

FREETILE_LIB      := $(BUILD_DIR)/libfreetile.a
FREETILE_TWIN_LIB := $(BUILD_DIR)/libfreetile_twin.a

TWIN_CLI          := $(BUILD_DIR)/freetile-twin
ROOT_TWIN_CLI     := $(ROOT_BIN_DIR)/freetile-twin

TEST_FREETILE     := $(BUILD_DIR)/test_freetile
TEST_SPLINES      := $(BUILD_DIR)/test_splines_and_flash_attn
TEST_PARITY       := $(BUILD_DIR)/test_buenos_parity
TEST_TWIN         := $(BUILD_DIR)/test_freetile_twin

.PHONY: all host twin test verify-header clean openbuenos-host

all: host

host: $(FREETILE_LIB) $(FREETILE_TWIN_LIB) $(TWIN_CLI) $(ROOT_TWIN_CLI) \
      $(TEST_FREETILE) $(TEST_SPLINES) $(TEST_PARITY) $(TEST_TWIN)

twin: $(TWIN_CLI) $(ROOT_TWIN_CLI)

openbuenos-host:
	$(MAKE) -C $(OB_DIR) host

$(OB_LIB): openbuenos-host

$(BUILD_DIR)/obj/%.o: $(THIS_DIR)/src/%.cpp $(CORE_HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_BASE) -c $< -o $@

$(TWIN_OBJ): $(TWIN_SRC) $(TWIN_HDR) $(CORE_HDRS) $(OB_LIB)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_NOEXC) $(OB_INCS) -c $< -o $@

$(FREETILE_LIB): $(CORE_OBJS)
	@mkdir -p $(dir $@)
	$(HOST_AR) rcs $@ $(CORE_OBJS)

$(FREETILE_TWIN_LIB): $(TWIN_OBJ)
	@mkdir -p $(dir $@)
	$(HOST_AR) rcs $@ $(TWIN_OBJ)

$(TWIN_CLI): $(THIS_DIR)/tools/freetile_twin_cli.cpp $(TWIN_HDR) $(FREETILE_TWIN_LIB) $(FREETILE_LIB) $(OB_LIB)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_NOEXC) $< $(FREETILE_TWIN_LIB) $(FREETILE_LIB) $(OB_LIB) -o $@ $(LDFLAGS)

$(ROOT_TWIN_CLI): $(TWIN_CLI)
	@mkdir -p $(dir $@)
	cp -f $(TWIN_CLI) $(ROOT_TWIN_CLI)
	chmod +x $(ROOT_TWIN_CLI)

$(TEST_FREETILE): $(THIS_DIR)/tests/test_freetile.cpp $(FREETILE_LIB) $(CORE_HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_BASE) $< $(FREETILE_LIB) -o $@ $(LDFLAGS)

$(TEST_SPLINES): $(THIS_DIR)/tests/test_splines_and_flash_attn.cpp $(FREETILE_LIB) $(CORE_HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_NOEXC) $< $(FREETILE_LIB) -o $@ $(LDFLAGS)

$(TEST_PARITY): $(THIS_DIR)/tests/test_buenos_parity.cpp $(FREETILE_LIB) $(OB_LIB) $(CORE_HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_BASE) $(OB_INCS) $< $(FREETILE_LIB) $(OB_LIB) -o $@ $(LDFLAGS)

$(TEST_TWIN): $(THIS_DIR)/tests/test_freetile_twin.cpp $(TWIN_HDR) $(FREETILE_TWIN_LIB) $(FREETILE_LIB) $(OB_LIB)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS_NOEXC) $< $(FREETILE_TWIN_LIB) $(FREETILE_LIB) $(OB_LIB) -o $@ $(LDFLAGS)

verify-header: $(TWIN_HDR)
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Wpedantic -Werror $<
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Wpedantic -Werror $<

test: verify-header $(TEST_FREETILE) $(TEST_SPLINES) $(TEST_PARITY) $(TEST_TWIN) $(ROOT_TWIN_CLI)
	$(TEST_FREETILE)
	$(TEST_SPLINES)
	$(TEST_PARITY)
	$(TEST_TWIN)

clean:
	rm -rf $(THIS_DIR)/build $(ROOT_TWIN_CLI)
