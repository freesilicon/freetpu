# Third-Party Notices (`libopentpu`)

## OpenXLA PJRT C API Header (`third_party/openxla/pjrt_c_api.h`)

- **Upstream Project**: OpenXLA (`https://github.com/openxla/xla`)
- **Pinned Commit**: `0bfeeb476952fc0ae13dd38e27a9053cb1baa3ce`
- **Source Path**: `xla/pjrt/c/pjrt_c_api.h`
- **Upstream SHA-256**: `174b44736ccb200cb402c2466822220971ba578e1e8dfd690c56f822a9e71582`
- **Local Modifications**: Added an `#ifdef __cplusplus` guard on `_PJRT_API_STRUCT_FIELD(fn_type)` to qualify `::fn_type* fn_type` for strict GCC C++ builds (`-fpermissive` / `-Wchanges-meaning`) with zero ABI change.
- **Vendored SHA-256**: `cf7ac9f3a42c03f4c477a4c79e2b1816b1d52f7a73e0911fa0cd5b8b133067b5`
- **License**: Apache License, Version 2.0 (`Copyright 2022 The OpenXLA Authors`)

Zero proprietary vendor binaries or `.so` libraries are shipped or linked by `libopentpu`.
