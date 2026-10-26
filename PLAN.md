# Station Tycho: WebGPU Engine & 3D Lunar Map Plan

**Target:** `uat.kings.fyi` — Cloudflare Workers + R2 (`lunar-assets`)
**Stack:** Rust (`wasm32-unknown-unknown`), `wgpu` 30 (`webgpu` + `wgsl` features), `web-sys`, TypeScript Worker, WGSL

---

## 1. System Overview

The Rust/WASM engine renders on the client GPU and streams multi-resolution lunar elevation and albedo tiles from R2 through the Worker.

```
[NASA LOLA / SLDEM2015 / LROC WAC source rasters]
        │  offline pipeline (GDAL reprojection → cube faces → tiling → quantization)
        ▼
[R2: lunar-assets  /v{N}/dem, /v{N}/albedo, /v{N}/meta]
        │  Worker `/assets/*` (src/index.ts) + Cache API
        ▼
[WASM engine (crate/src/lib.rs)]
   ├── Camera (f64) + camera-relative rendering
   ├── Quadtree LOD (SSE + frustum + horizon culling)
   ├── Tile fetcher (bounded concurrency, cancellation, LRU GPU cache)
   └── Render pipeline (WGSL displacement, Lommel-Seeliger shading, reversed-Z)
        ▼
[<canvas id="viewport">]
```

---

## 2. Phase 1 — Engine Foundation

### 2.1 Browser compatibility — DONE
- Upgraded `wgpu` 0.20 → 30. wgpu 0.20 always sent the removed `maxInterStageShaderComponents` limit, which Chrome rejects.
- Request only the limits the engine needs (start from `wgpu::Limits::default()`), not `adapter.limits()`, so behaviour is the same on every device.

### 2.2 Frame loop
- Current: JS `requestAnimationFrame` calls `Renderer.render(t)` (a `#[wasm_bindgen]` struct). Keep this model; it keeps lifetimes simple and the HUD in JS.
- A surface that is Lost or Outdated triggers a reconfigure. Timeout or Occluded skips the frame.

### 2.3 Resize & DPI
- Remove the fixed `width`/`height` attributes from the canvas.
- Watch the canvas with a `ResizeObserver` using `devicePixelContentBoxSize`. Call `Renderer.resize(w, h)`, which reconfigures the surface and rebuilds the depth texture.

### 2.4 Depth
- `Depth32Float`, **reversed-Z**: clear to `0.0`, compare with `Greater`, infinite-far perspective projection.

### 2.5 Fallback
- If `navigator.gpu` is missing or getting the adapter fails, show a clear unsupported-browser message (no WebGL fallback for now).

---

## 3. Phase 2 — Geometry: Spherified Cube Quadtree

### 3.1 Conventions
- **Units:** metres everywhere. `R_moon = 1_737_400 m` (IAU reference sphere).
- **Frame:** IAU Moon Mean-Earth (ME) body-fixed frame. +Z = north pole, +X = 0° lon / 0° lat.
- Define `latlon_to_cube(face, u, v)` and its inverse once in Rust and use them for streaming, picking, and the HUD.

### 3.2 Cube → sphere mapping
For $\mathbf{p} \in [-1,1]^3$ on a cube face:

$$
\mathbf{s} = \begin{bmatrix}
x\sqrt{1 - \tfrac{y^2}{2} - \tfrac{z^2}{2} + \tfrac{y^2 z^2}{3}} \\
y\sqrt{1 - \tfrac{z^2}{2} - \tfrac{x^2}{2} + \tfrac{z^2 x^2}{3}} \\
z\sqrt{1 - \tfrac{x^2}{2} - \tfrac{y^2}{2} + \tfrac{x^2 y^2}{3}}
\end{bmatrix}
$$

(More even than plain normalization, but not equal-area.)

### 3.3 Precision-safe vertex positions
Absolute positions (~1.7e6 m) in `f32` resolve only to about 0.2 m, which causes jitter. Therefore:
- **CPU (f64, `glam::DVec3`):** for each visible tile, compute its centre $\mathbf{T}$ and its 4 corner positions on the sphere.
- **GPU uniform (f32):** `tile_origin_rel = T − C_camera`, plus the corner offsets relative to `T`.
- **Vertex shader:** blend the corner offsets bilinearly (or use a small local cube-to-sphere evaluation relative to `T`), add `normal * h * exaggeration`, add `tile_origin_rel`, then multiply by the view-projection matrix.
- The view matrix contains **rotation only** (the camera sits at the origin).

### 3.4 Shared grid mesh
- One static 65×65 vertex grid plus a **skirt** ring (pushed inward along the normal by a per-level depth) to hide cracks between LOD levels.
- Index buffer for triangles, and a separate line-list index buffer for wireframe. WebGPU has no polygon-mode line.

### 3.5 LOD selection
- Screen-space error: $\rho = \dfrac{\epsilon_{\text{level}} \cdot H_{\text{px}}}{2\,d\,\tan(\text{fov}/2)}$. Split when $\rho > \tau$ (for example 2 px).
- $\epsilon_{\text{level}}$ = geometric error per level, taken from the manifest (or halved per level).
- Cull with the view frustum and the **horizon** (the occluding sphere test).
- Keep a parent visible until all 4 children have loaded, so the terrain never shows holes.

---

## 4. Phase 3 — Asset Pipeline & Edge Delivery

### 4.1 Source datasets
| Use | Dataset | Coverage / resolution |
|---|---|---|
| DEM (mid-latitudes) | SLDEM2015 (LOLA + Kaguya TC) | ±60° lat, ~59 m/px |
| DEM (poles) | LOLA polar stereographic DEMs | >60° lat, 5–20 m/px |
| DEM (Tycho detail) | LROC NAC DTMs | local, ~2–5 m/px |
| Albedo | LROC WAC **Hapke-normalized** reflectance mosaic | global, ~100 m/px |

Do **not** use the WAC *Morphology* mosaic for albedo: its shading is baked in and would be lit twice.

### 4.2 Offline pipeline
1. Reproject each source to the 6 cube faces (GDAL warp with a custom cube-face projection, or a Python/Rust resampler).
2. Blend the DEM sources with feathered seams (SLDEM ↔ polar at about 60°).
3. Tile each face into a quadtree. Tiles are **257×257** (one shared border pixel) so neighbouring tiles line up exactly.
4. Quantize the DEM to `u16` with a global `scale` and `offset` (range about −9.2 km to +10.8 km, 0.5 m/LSB).
5. Write `manifest.json`: per-level tile availability, min/max height per tile, geometric error per level, quantization parameters, dataset version.

### 4.3 R2 layout (versioned, immutable)
```text
lunar-assets/
└── v1/
    ├── meta/manifest.json
    ├── dem/{face}/{level}/{x}/{y}.bin       # u16 LE, 257×257
    └── albedo/{face}/{level}/{x}/{y}.rgba   # RGBA8 (KTX2 later)
```
`face ∈ {px, nx, py, ny, pz, nz}`. Never overwrite a tile; publish a new `vN` instead.

### 4.4 Worker (`src/index.ts`)
- Already done: `Cache-Control: public, max-age=31536000, immutable`.
- Add **Cache API** (`caches.default`) lookups and stores. Worker responses are not cached at the edge automatically.
- Add `ETag` / `If-None-Match` support using R2 `httpEtag` and `onlyIf`.
- Add `Range` support (R2 `get(key, { range })`) for a packed bundle of levels 0–3.
- Restrict CORS to the app origin (or drop it, since requests are same-origin).

---

## 5. Phase 4 — Shading & Camera

### 5.1 Textures (WebGPU-core safe)
| Data | Format | Sampling |
|---|---|---|
| DEM | `R16Uint` (or `R32Float`) | `textureLoad` + manual bilinear in the vertex shader |
| Albedo | `Rgba8UnormSrgb` | filtered sampler |
| Albedo (later) | KTX2/BasisU → BC7 / ASTC / ETC2 by adapter features | needs a WASM transcoder |

### 5.2 Bind groups (`crate/src/shaders/lunar_terrain.wgsl`)
- **Group 0 (per frame):** `view_proj` (rotation-only view), `sun_dir: vec4<f32>` (padded), `exaggeration`, `debug_flags`.
- **Group 1 (per tile):** `tile_origin_rel: vec4`, corner offsets, `uv_scale_offset: vec4`, DEM texture, albedo texture.
- Later: move to texture arrays or an atlas to cut bind-group churn.

### 5.3 Lighting
- Start with **Lommel-Seeliger**, blended with Lambert (lunar-Lambert / McEwen). Add the Hapke opposition surge later.
- No atmosphere: black sky, hard terminator.
- **Shadows** (crater rims, central peaks) need a separate technique: per-tile horizon maps (precomputed offline) or a cascaded shadow map. Vertex displacement alone only gives shading, not cast shadows.

### 5.4 Camera & input
- Orbit camera stored in f64 (target, distance, azimuth, elevation). Mouse and touch drag, wheel zoom (scaled logarithmically with altitude).
- Near plane scales with altitude; far plane is infinite (reversed-Z).

---

## 6. Runtime Systems
- **Tile cache:** LRU with a GPU memory budget. Evict tiles that haven't been visible for N frames.
- **Fetching:** at most 6–8 requests in flight, priority = SSE × visibility, cancel requests that are no longer needed (`AbortController`).
- **Errors:** a 404 or a failed decode marks the tile unavailable and keeps the parent visible.
- **Picking:** keep downsampled height data on the CPU per loaded tile. Cast a ray from the camera against the sphere plus heights, convert the hit to lat/lon with `cube_to_latlon`.

---

## 7. Roadmap

| Milestone | Deliverable | Verification |
|---|---|---|
| **M1: Heartbeat** ✅ | wgpu 30, animated test pipeline, FPS readout | Rotating triangle at refresh rate; HUD shows `WebGPU Engine Online` |
| **M2: 3D foundation** | f64 orbit camera, camera-relative rendering, reversed-Z depth, resize/DPI, procedural sphere (6 faces, line-list wireframe) | 60 FPS sphere, mouse orbit/zoom, no jitter at 1 m altitude |
| **M3: R2 textures** | Fetch a raw RGBA8 + `u16` DEM test tile from `/assets/v1/...`, upload with `write_texture` | Sphere textured from R2; second load served from the edge cache (`cf-cache-status: HIT`) |
| **M3.5: Asset pipeline** | Cube-face reprojection, 257² tiling, quantization, manifest, upload script | Manifest validates; neighbouring tiles share identical edge heights |
| **M4: Quadtree terrain** | SSE + horizon culling, skirts, tile-relative displacement, LRU cache, prioritized fetching | Tycho rim and central peak visible; no cracks or holes while zooming |
| **M4.5: Lighting** | Lommel-Seeliger shading, sun controls, horizon-map shadows | Terminator crossing Tycho shows correct relief shadows |
| **M5: HUD & telemetry** | CPU-side picking, lat/lon/altitude readout, streaming stats (in-flight, cached, GPU MB) | Clicking Tycho's centre reads about 43.3°S, 11.2°W |

---

## 8. Open Decisions
- KTX2 transcoder choice (`basis_universal` via C++ → WASM vs. a pure-Rust alternative) and when to adopt it.
- Whether to support adapters without WebGPU at all (WebGL2 backend in wgpu) or show an unsupported message only.
- Hosting R2 tiles on a custom domain instead of serving them through the Worker.