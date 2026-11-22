#!/usr/bin/env python3
"""Export StableHLO MLIR bytecode/text + CompileOptions and build Zero-Python C++17 PJRT binary."""

import ctypes
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import jax
import jax.numpy as jnp

DEFAULT_WORK_DIR = os.path.join(os.path.expanduser("~"), ".cache", "freetpu-pjrt")
IR_DIR = os.environ.get("TPU_PJRT_IR_DIR") or os.path.join(DEFAULT_WORK_DIR, "ir")
CPP_SRC = Path(__file__).with_name("tpu_pjrt_bench.cpp")
BIN_PATH = os.environ.get("TPU_PJRT_BIN_PATH") or os.path.join(
    DEFAULT_WORK_DIR, "tpu_pjrt_bench"
)
PJRT_HEADER_URL = (
    "https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h"
)


def _is_safe_existing_ancestor(ancestor: str) -> bool:
  if not os.path.lexists(ancestor):
    return True
  st = os.lstat(ancestor)
  euid = os.geteuid() if hasattr(os, "geteuid") else st.st_uid
  if stat.S_ISLNK(st.st_mode):
    if st.st_uid != 0 and st.st_uid != euid:
      return False
    try:
      st = os.stat(ancestor)
    except OSError:
      return False
  if not stat.S_ISDIR(st.st_mode):
    return False
  if st.st_uid != euid and st.st_uid != 0:
    return False
  if (st.st_mode & 0o022) != 0 and not (st.st_mode & stat.S_ISVTX):
    return False
  return True


def _resolve_and_validate_dir_path(raw_path: str) -> str:
  """Resolve symlinks hop-by-hop from `/` and validate every existing ancestor."""
  expanded = os.path.expanduser(raw_path)
  full = expanded if os.path.isabs(expanded) else os.path.join(os.getcwd(), expanded)
  if not _is_safe_existing_ancestor("/"):
    raise RuntimeError("Refusing directory under untrusted ancestor: /")
  resolved = "/"
  parts = [p for p in full.split("/") if p]
  hops = 0
  idx = 0
  while idx < len(parts):
    part = parts[idx]
    idx += 1
    if part == ".":
      continue
    if part == "..":
      if resolved != "/":
        resolved = os.path.dirname(resolved)
      continue
    candidate = "/" + part if resolved == "/" else resolved + "/" + part
    if os.path.lexists(candidate):
      if not _is_safe_existing_ancestor(candidate):
        raise RuntimeError(f"Refusing directory under untrusted ancestor: {candidate}")
      if os.path.islink(candidate):
        hops += 1
        if hops > 40:
          raise RuntimeError(f"Too many symlink hops while validating path: {raw_path}")
        target = os.readlink(candidate)
        t_parts = [p for p in target.split("/") if p]
        if target.startswith("/"):
          resolved = "/"
        parts = t_parts + parts[idx:]
        idx = 0
        continue
      resolved = candidate
    else:
      if any(p == ".." for p in parts[idx:]):
        raise RuntimeError(f"Refusing '..' after missing path component in: {raw_path}")
      resolved = candidate
      for rem in parts[idx:]:
        if rem == ".":
          continue
        resolved = "/" + rem if resolved == "/" else resolved + "/" + rem
      break
  return resolved


def ensure_private_dir(raw_dir: str) -> str:
  """Ensure `raw_dir` resolves to a safe non-symlink directory owned by the user with mode 0700."""
  expanded = os.path.expanduser(raw_dir)
  if os.path.islink(expanded):
    raise RuntimeError(f"Refusing symlinked directory path: {raw_dir}")
  abs_dir = _resolve_and_validate_dir_path(expanded)
  if not os.path.lexists(abs_dir):
    missing = []
    cur = abs_dir
    while not os.path.lexists(cur):
      missing.append(cur)
      parent = os.path.dirname(cur)
      if parent == cur:
        break
      cur = parent
    for d in reversed(missing):
      try:
        os.mkdir(d, 0o700)
      except FileExistsError:
        pass
  st = os.lstat(abs_dir)
  if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
    raise RuntimeError(f"Expected non-symlink directory, got: {abs_dir}")
  if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
    raise RuntimeError(f"Directory {abs_dir} is not owned by current user")
  if (st.st_mode & 0o077) != 0:
    raise RuntimeError(
        f"Directory {abs_dir} must have private mode 0700 (got {oct(stat.S_IMODE(st.st_mode))})"
    )
  return abs_dir


def write_bytes_nofollow(path: str, data: bytes, mode: int = 0o600) -> None:
  if os.path.islink(path):
    raise RuntimeError(f"Refusing symlinked output path: {path}")
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
  fd = os.open(path, flags, mode)
  with os.fdopen(fd, "wb") as f:
    f.write(data)


def resolve_tpu_library_path() -> str:
  """Resolve the TPU library using JAX's precedence (`TPU_LIBRARY_PATH` before `libtpu`)."""
  env_path = os.environ.get("TPU_LIBRARY_PATH")
  if env_path and os.path.exists(env_path):
    return env_path
  import libtpu  # noqa: WPS433

  return libtpu.get_library_path()


def probe_header_and_live_abi(hdr_path: str) -> str:
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

  so_path = resolve_tpu_library_path()
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
  return so_path


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


def export_gemm_ir(
    ir_dir: str, m: int, k: int, n: int, out_dtype_tag: str, tag: str
) -> bytes:
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

  write_bytes_nofollow(
      os.path.join(ir_dir, f"gemm_{tag}.mlir"), mlir_text.encode("utf-8")
  )

  bio = io.BytesIO()
  if hasattr(ir_mod, "operation") and hasattr(ir_mod.operation, "write_bytecode"):
    ir_mod.operation.write_bytecode(file=bio)
    mlir_bc = bio.getvalue()
  else:
    from jax._src.interpreters import mlir as jax_mlir

    mlir_bc = jax_mlir.module_to_bytecode(ir_mod)

  write_bytes_nofollow(os.path.join(ir_dir, f"gemm_{tag}.mlirbc"), mlir_bc)

  print(f"Exported {tag}: .mlir={len(mlir_text)} B, .mlirbc={len(mlir_bc)} B")
  return mlir_bc


def main() -> None:
  if CPP_SRC.is_symlink() or not CPP_SRC.is_file():
    raise RuntimeError(f"Invalid C++ source path: {CPP_SRC}")

  ir_dir = ensure_private_dir(IR_DIR)
  bin_expanded = os.path.expanduser(BIN_PATH)
  if os.path.islink(bin_expanded):
    raise RuntimeError(f"Refusing symlinked output binary path: {bin_expanded}")
  bin_name = os.path.basename(bin_expanded)
  if bin_name in ("", ".", ".."):
    raise RuntimeError(f"Invalid output binary path: {BIN_PATH}")
  safe_bin_dir = ensure_private_dir(os.path.dirname(bin_expanded) or ".")
  bin_abs = os.path.join(safe_bin_dir, bin_name)
  if os.path.islink(bin_abs):
    raise RuntimeError(f"Refusing symlinked output binary path: {bin_abs}")

  hdr_path = os.path.join(ir_dir, "pjrt_c_api.h")
  if os.path.islink(hdr_path):
    raise RuntimeError(f"Refusing symlinked header target: {hdr_path}")
  subprocess.run(["curl", "-fsSL", PJRT_HEADER_URL, "-o", hdr_path], check=True)
  so_path = probe_header_and_live_abi(hdr_path)

  try:
    from jax.extend import backend as jax_backend

    backend = jax_backend.get_backend()
  except Exception:
    from jax._src import xla_bridge

    backend = xla_bridge.get_backend()

  pb_jax, pb_min = serialize_compile_options(backend)
  write_bytes_nofollow(os.path.join(ir_dir, "compile_options.pb"), pb_jax)
  write_bytes_nofollow(os.path.join(ir_dir, "compile_options_minimal.pb"), pb_min)

  specs = [
      (1, 64, 32, "f32", "1x64x32_bf16_f32"),
      (1024, 1024, 1024, "f32", "1024x1024x1024_bf16_f32"),
      (4096, 4096, 4096, "bf16", "4096x4096x4096_bf16_bf16"),
      (8192, 8192, 8192, "bf16", "8192x8192x8192_bf16_bf16"),
  ]
  for m, k, n, dt, tag in specs:
    export_gemm_ir(ir_dir, m, k, n, dt, tag)

  compile_cmd = [
      "g++",
      "-O3",
      "-std=c++17",
      "-Wall",
      "-Wextra",
      f"-I{ir_dir}",
      f"-DFREETPU_DEFAULT_LIBTPU_PATH={json.dumps(so_path)}",
      str(CPP_SRC),
      "-ldl",
      "-pthread",
      "-o",
      bin_abs,
  ]
  subprocess.run(compile_cmd, check=True)
  print(f"Built binary: {bin_abs} ({os.path.getsize(bin_abs)} bytes)")
  print(f"Run: LIBTPU_PATH={so_path} TPU_PJRT_IR_DIR={ir_dir} {bin_abs}")
  print("=== EXPORT_AND_BUILD_COMPLETE ===")


if __name__ == "__main__":
  main()
