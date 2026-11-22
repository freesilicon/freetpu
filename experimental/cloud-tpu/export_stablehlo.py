#!/usr/bin/env python3
"""Export StableHLO MLIR bytecode/text + CompileOptions and build Zero-Python C++17 PJRT binary."""

import ctypes
import io
import os
from pathlib import Path
import re
import shutil
import subprocess

import jax
import jax.numpy as jnp

IR_DIR = os.environ.get("TPU_PJRT_IR_DIR", "/tmp/tpu_pjrt_ir")
CPP_SRC = Path(__file__).with_name("tpu_pjrt_bench.cpp")
CPP_PATH = os.environ.get("TPU_PJRT_CPP_PATH", "/tmp/tpu_pjrt_bench.cpp")
BIN_PATH = os.environ.get("TPU_PJRT_BIN_PATH", "/tmp/tpu_pjrt_bench")
PJRT_HEADER_URL = (
    "https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h"
)


def probe_header_and_live_abi(hdr_path: str) -> None:
  hdr_txt = open(hdr_path, "r", encoding="utf-8").read()
  m_maj = re.search(r"#define\s+PJRT_API_MAJOR\s+(\d+)", hdr_txt)
  m_min = re.search(r"#define\s+PJRT_API_MINOR\s+(\d+)", hdr_txt)
  hdr_maj = int(m_maj.group(1)) if m_maj else -1
  hdr_min = int(m_min.group(1)) if m_min else -1
  print(f"Downloaded pjrt_c_api.h: PJRT_API_MAJOR={hdr_maj}, PJRT_API_MINOR={hdr_min}")

  m_api = re.search(r"typedef\s+struct\s+PJRT_Api\s*\{(.*?)\}\s*PJRT_Api;", hdr_txt, re.S)
  if not m_api:
    raise RuntimeError("Could not parse typedef struct PJRT_Api in pjrt_c_api.h")
  fields = re.findall(r"_PJRT_API_STRUCT_FIELD\(\s*([A-Za-z0-9_]+)\s*\)", m_api.group(1))
  hdr_struct_size = 40 + 8 * len(fields)
  print(f"Header PJRT_Api fn_fields={len(fields)}, expected_LP64_struct_size={hdr_struct_size}")

  required_fns = [
      "PJRT_Error_Destroy",
      "PJRT_Error_Message",
      "PJRT_Plugin_Initialize",
      "PJRT_Event_Destroy",
      "PJRT_Event_Await",
      "PJRT_Client_Create",
      "PJRT_Client_Destroy",
      "PJRT_Client_PlatformName",
      "PJRT_Client_PlatformVersion",
      "PJRT_Client_AddressableDevices",
      "PJRT_Client_Compile",
      "PJRT_Client_BufferFromHostBuffer",
      "PJRT_Device_GetDescription",
      "PJRT_DeviceDescription_Id",
      "PJRT_DeviceDescription_Kind",
      "PJRT_DeviceDescription_ToString",
      "PJRT_Device_MemoryStats",
      "PJRT_LoadedExecutable_Destroy",
      "PJRT_LoadedExecutable_GetExecutable",
      "PJRT_LoadedExecutable_AddressableDevices",
      "PJRT_LoadedExecutable_Execute",
      "PJRT_Executable_Destroy",
      "PJRT_Executable_NumOutputs",
      "PJRT_Buffer_Destroy",
      "PJRT_Buffer_ToHostBuffer",
      "PJRT_Buffer_ReadyEvent",
  ]
  fn_offsets = {name: 40 + 8 * idx for idx, name in enumerate(fields)}

  import libtpu  # noqa: WPS433

  so_path = libtpu.get_library_path()
  lib = ctypes.CDLL(so_path)
  lib.GetPjrtApi.restype = ctypes.c_void_p
  api_ptr = lib.GetPjrtApi()

  class PJRT_Api_Header(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("ver_struct_size", ctypes.c_size_t),
        ("ver_ext", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]

  rt_hdr = PJRT_Api_Header.from_address(api_ptr)
  print(
      f"Live libtpu.so GetPjrtApi(): path={so_path}, "
      f"struct_size={rt_hdr.struct_size}, version={rt_hdr.major_version}.{rt_hdr.minor_version}"
  )

  missing_or_oob = []
  for name in required_fns:
    off = fn_offsets.get(name)
    if off is None or off + 8 > rt_hdr.struct_size:
      missing_or_oob.append((name, off))
      continue
    ptr_val = ctypes.c_void_p.from_address(api_ptr + off).value
    if not ptr_val:
      missing_or_oob.append((name, off))
  if missing_or_oob:
    raise RuntimeError(f"ABI mismatch for required PJRT_Api pointers: {missing_or_oob}")

  critical_structs = [
      "PJRT_Plugin_Initialize_Args",
      "PJRT_Error_Destroy_Args",
      "PJRT_Error_Message_Args",
      "PJRT_Event_Destroy_Args",
      "PJRT_Event_Await_Args",
      "PJRT_Client_Create_Args",
      "PJRT_Client_Destroy_Args",
      "PJRT_Client_AddressableDevices_Args",
      "PJRT_Client_Compile_Args",
      "PJRT_Client_BufferFromHostBuffer_Args",
      "PJRT_Program",
      "PJRT_ExecuteOptions",
      "PJRT_LoadedExecutable_Execute_Args",
      "PJRT_Buffer_ToHostBuffer_Args",
      "PJRT_Buffer_ReadyEvent_Args",
      "PJRT_Device_GetDescription_Args",
      "PJRT_DeviceDescription_Kind_Args",
      "PJRT_Device_MemoryStats_Args",
  ]
  for sname in critical_structs:
    m_tr = re.search(
        rf"PJRT_DEFINE_STRUCT_TRAITS\(\s*{sname}\s*,\s*([A-Za-z0-9_]+)\s*\)",
        hdr_txt,
    )
    if not m_tr:
      raise RuntimeError(f"Missing PJRT_DEFINE_STRUCT_TRAITS for {sname}")


def serialize_compile_options(backend) -> tuple[bytes, bytes]:
  from jax._src.lib import xla_client

  def _serialize(o) -> bytes:
    if hasattr(o, "SerializeAsString"):
      return o.SerializeAsString()
    return o.SerializeToString()

  opts_jax = None
  try:
    from jax._src import compiler as jax_compiler

    opts_jax = jax_compiler.get_compile_options(
        num_replicas=1,
        num_partitions=1,
        device_assignment=jax.devices()[:1],
        backend=backend,
    )
  except Exception as e:
    print("Warning: jax_compiler.get_compile_options fallback:", repr(e))

  opts_min = xla_client.CompileOptions()
  opts_min.num_replicas = 1
  opts_min.num_partitions = 1
  if hasattr(opts_min, "executable_build_options"):
    eb = opts_min.executable_build_options
    eb.use_spmd_partitioning = True
    if hasattr(eb, "use_shardy_partitioner"):
      eb.use_shardy_partitioner = bool(
          getattr(jax.config, "jax_use_shardy_partitioner", True)
      )

  if opts_jax is None:
    opts_jax = opts_min
  return _serialize(opts_jax), _serialize(opts_min)


def export_gemm_ir(m: int, k: int, n: int, out_dtype_tag: str, tag: str) -> bytes:
  out_dtype = jnp.float32 if out_dtype_tag == "f32" else jnp.bfloat16

  def gemm(a, b):
    return jnp.matmul(a, b, preferred_element_type=out_dtype)

  a_spec = jax.ShapeDtypeStruct((m, k), jnp.bfloat16)
  b_spec = jax.ShapeDtypeStruct((k, n), jnp.bfloat16)
  lowered = jax.jit(gemm).lower(a_spec, b_spec)

  ir_mod = lowered.compiler_ir(dialect="stablehlo")
  mlir_text = str(ir_mod)
  if "func.func" not in mlir_text:
    mlir_text = lowered.as_text("stablehlo")

  with open(os.path.join(IR_DIR, f"gemm_{tag}.mlir"), "w", encoding="utf-8") as f:
    f.write(mlir_text)

  bio = io.BytesIO()
  if hasattr(ir_mod, "operation") and hasattr(ir_mod.operation, "write_bytecode"):
    ir_mod.operation.write_bytecode(file=bio)
    mlir_bc = bio.getvalue()
  else:
    from jax._src.interpreters import mlir as jax_mlir

    mlir_bc = jax_mlir.module_to_bytecode(ir_mod)

  with open(os.path.join(IR_DIR, f"gemm_{tag}.mlirbc"), "wb") as f:
    f.write(mlir_bc)

  print(f"Exported {tag}: .mlir={len(mlir_text)} B, .mlirbc={len(mlir_bc)} B")
  return mlir_bc


def main() -> None:
  os.makedirs(IR_DIR, exist_ok=True)
  hdr_path = os.path.join(IR_DIR, "pjrt_c_api.h")
  subprocess.run(["curl", "-fsSL", PJRT_HEADER_URL, "-o", hdr_path], check=True)
  probe_header_and_live_abi(hdr_path)

  try:
    backend = jax.extend.backend.get_backend()
  except Exception:
    from jax._src import xla_bridge

    backend = xla_bridge.get_backend()

  pb_jax, pb_min = serialize_compile_options(backend)
  with open(os.path.join(IR_DIR, "compile_options.pb"), "wb") as f:
    f.write(pb_jax)
  with open(os.path.join(IR_DIR, "compile_options_minimal.pb"), "wb") as f:
    f.write(pb_min)

  specs = [
      (1, 64, 32, "f32", "1x64x32_bf16_f32"),
      (1024, 1024, 1024, "f32", "1024x1024x1024_bf16_f32"),
      (4096, 4096, 4096, "bf16", "4096x4096x4096_bf16_bf16"),
      (8192, 8192, 8192, "bf16", "8192x8192x8192_bf16_bf16"),
  ]
  for m, k, n, dt, tag in specs:
    export_gemm_ir(m, k, n, dt, tag)

  if os.path.abspath(str(CPP_SRC)) != os.path.abspath(CPP_PATH):
    shutil.copyfile(CPP_SRC, CPP_PATH)

  compile_cmd = [
      "g++",
      "-O3",
      "-std=c++17",
      "-Wall",
      "-Wextra",
      f"-I{IR_DIR}",
      CPP_PATH,
      "-ldl",
      "-pthread",
      "-o",
      BIN_PATH,
  ]
  subprocess.run(compile_cmd, check=True)
  print(f"Built binary: {BIN_PATH} ({os.path.getsize(BIN_PATH)} bytes)")
  print("=== EXPORT_AND_BUILD_COMPLETE ===")


if __name__ == "__main__":
  main()
