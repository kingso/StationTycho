# Station Tycho: WebGPU Engine & 3D Lunar Map Architecture Plan

**Target Environment:** `uat.kings.fyi` / Cloudflare Workers & R2 Edge  
**Core Stack:** Rust (`wasm32-unknown-unknown`), `wgpu`, `web-sys`, TypeScript (Cloudflare Worker), WGSL  

---

## 1. System Overview & Data Flow

Station Tycho is an edge-streamed 3D planetary rendering engine. Rather than bundling massive lunar datasets into the client payload, the Rust/WASM engine runs on the client GPU and streams multi-resolution NASA Lunar Reconnaissance Orbiter (LRO) elevation and albedo tiles directly from a private Cloudflare R2 bucket (`lunar-assets`).

```
[NASA LRO Raw GeoTIFFs] 
       │ (Offline Preprocessing Pipeline)
       ▼
[Cloudflare R2: `lunar-assets` (KTX2 / Quantized Heightmaps)]
       │
       ▼ (Edge Route: `/assets/*` via `src/index.ts`)
[Rust WASM Engine (`crate/src/lib.rs`)]
       ├── Quadtree LOD Manager (Frustum & Screen-Space Error Culling)
       ├── Async Tile Fetcher (`web-sys` Fetch -> `wgpu::Queue::write_texture`)
       └── WebGPU Render Pipeline (WGSL Vertex Displacement + Hapke BRDF Shading)
              │
              ▼
   [`<canvas id="viewport">`]
```

---

## 2. Phase 1: WebGPU Context & Modern Browser Compatibility (Current Focus)

### 2.1 Fixing the `maxInterStageShaderComponents` Browser Error
**Root Cause:** In mid-2024/2025, Chromium and WebGPU standards removed the deprecated `maxInterStageShaderComponents` limit from `GPUAdapter.requestDevice()`. However, `wgpu v0.20.x` unconditionally serializes every struct field in `wgpu::Limits` into the JavaScript `GPUDeviceDescriptor` dictionary whenever any limit differs from the spec default—or when `wgpu`'s internal default includes `max_inter_stage_shader_components: 60`.

**Permanent Fixes (Choose Option A or Option B):**
* **Option A (Recommended): Upgrade `wgpu` in `crate/Cargo.toml` to `v22.0+` or `v23.0+`**  
  `wgpu` 22+ removed `max_inter_stage_shader_components` from the `webgpu` backend serialization to match modern Chrome/Edge WebGPU specs.
* **Option B (If staying on `wgpu 0.20`): Strip `maxInterStageShaderComponents` in JS or upgrade `web-sys`**  
  If locked to `0.20`, patch the `Cargo.toml` dependencies or use a tiny JS wrapper on `GPUAdapter.prototype.requestDevice` in `index.html` before calling `init()`:
  ```javascript
  // Temporary polyfill in index.html if using wgpu 0.20 on modern Chrome:
  if (navigator.gpu) {
    const origRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = function(descriptor) {
      if (descriptor && descriptor.requiredLimits) {
        delete descriptor.requiredLimits.maxInterStageShaderComponents;
      }
      return origRequestDevice.call(this, descriptor);
    };
  }
  ```

### 2.2 Frame Loop & Resize Handling
* Transition `init_render_engine` from a single-frame clear pass into a persistent ` EngineState` struct held inside a `Rc<RefCell<EngineState>>`.
* Hook into `window.requestAnimationFrame` via `wasm-bindgen` closures to drive continuous 60–120 FPS rendering.
* Bind a `ResizeObserver` on `<canvas id="viewport">` to reconfigure the `wgpu::SurfaceConfiguration` and rebuild the depth texture (`wgpu::TextureFormat::Depth32Float`) whenever the browser window changes size.

---

## 3. Phase 2: 3D Lunar Geometry & Spherified Cube Quadtree

Standard UV spheres suffer from severe polar pinching at the North and South lunar poles (such as Shackleton Crater). Station Tycho will use a **Chunked LOD Spherified Cube (Quadtree Sphere)**.

### 3.1 Base Geometry
1. **Six Root Faces:** The Moon is represented as a cube with 6 root faces ($+X, -X, +Y, -Y, +Z, -Z$).
2. **Shared Grid Mesh:** A single static $65 \times 65$ vertex grid buffer is uploaded to the GPU once at startup. Every quadtree tile instanced on screen reuses this exact vertex buffer.
3. **Spherification Math (In Vertex Shader):**
   For a cube coordinate $\mathbf{p} \in [-1, 1]^3$, project onto the unit sphere using the uniform distribution mapping:
   $$\mathbf{s} = \begin{bmatrix} x\sqrt{1 - \frac{y^2}{2} - \frac{z^2}{2} + \frac{y^2 z^2}{3}} \\ y\sqrt{1 - \frac{z^2}{2} - \frac{x^2}{2} + \frac{z^2 x^2}{3}} \\ z\sqrt{1 - \frac{x^2}{2} - \frac{y^2}{2} + \frac{x^2 y^2}{3}} \end{bmatrix}$$
4. **Real-World Lunar Scaling:**
   * Mean Lunar Radius: $R_{\text{moon}} = 1737.4\text{ km}$.
   * Vertex Final Position:
     $$\mathbf{P}_{\text{world}} = \mathbf{s} \cdot \left(R_{\text{moon}} + h(u, v) \cdot S_{\text{vertical}}\right)$$
     where $h(u, v)$ is sampled from the R2 elevation tile and $S_{\text{vertical}}$ is an interactive vertical exaggeration uniform.

### 3.2 Quadtree LOD & Skirt Geometry
* **Screen-Space Error (SSE) Subdivision:** As the camera orbits closer to Tycho Crater or the lunar poles, nodes closer than threshold distance $d < k \cdot 2^{-\text{level}}$ split into 4 child quadrants.
* **Crack Prevention:** Include a 1-vertex outer "skirt" around the $65 \times 65$ grid mesh pushed slightly toward the lunar center to hide T-junction cracks between adjacent LOD levels.

---

## 4. Phase 3: R2 3D Map & Terrain Asset Pipeline

### 4.1 Source Datasets (NASA LRO)
* **Elevation (DEM):** LOLA (Lunar Orbiter Laser Altimeter) + Kaguya Merged DEM (59m/pixel global, up to 5m/pixel at polar/Tycho regions).
* **Color / Albedo:** LROC WAC (Wide Angle Camera) Global Morphology Mosaic (100m/pixel) + NAC (Narrow Angle Camera) high-res overlays for Station Tycho.

### 4.2 Tile Pyramid Structure in R2 (`lunar-assets`)
Store assets in Cloudflare R2 using a deterministic slippy-tile/quadtree key hierarchy served via `/assets/*`:
```text
lunar-assets/
├── meta/
│   └── manifest.json                  # Bounding boxes, min/max elevation per tile
├── dem/
│   └── {face}/{z}/{x}/{y}.bin         # 16-bit unsigned integer heightmap (256x256)
└── albedo/
    └── {face}/{z}/{x}/{y}.ktx2        # BasisU / UASTC compressed color + normal map
```

### 4.3 Edge Worker Optimization (`src/index.ts`)
* Attach aggressive edge caching headers (`Cache-Control: public, max-age=31536000, immutable`) to `/assets/dem/*` and `/assets/albedo/*` responses so Cloudflare's CDN caches tiles globally after the first fetch.
* Support HTTP `Range` requests if packing low-LOD levels (Levels 0–3) into a single packed archive file for single-request boot.

---

## 5. Phase 4: WGSL Shader Pipeline & Orbital Camera

### 5.1 Floating-Origin / Relative-to-Center Rendering (RTC)
Standard 32-bit floats (`f32`) lose precision at planetary scales ($1,737,400\text{ meters}$), causing violent vertex jitter when zoomed into surface stations.
* **CPU Side (`f64`):** Store camera position $\mathbf{C}_{\text{world}}$ and tile center $\mathbf{T}_{\text{world}}$ in double-precision `glam::DVec3` in Rust.
* **GPU Uniform (`f32`):** Pass the relative offset $\Delta \mathbf{T} = \text{vec3<f32>}(\mathbf{T}_{\text{world}} - \mathbf{C}_{\text{world}})$ to the WGSL vertex shader so all high-precision transforms happen relative to $(0, 0, 0)$ at the camera eye.

### 5.2 WGSL Shader Architecture (`crate/src/shaders/lunar_terrain.wgsl`)
* **Bind Group 0 (Global Frame Uniforms):**
  * View-Projection Matrix (`mat4x4<f32>`)
  * Sun Direction Vector (`vec3<f32>`)
  * Vertical Exaggeration & Debug Wireframe toggles
* **Bind Group 1 (Per-Tile Uniforms & Textures):**
  * Tile Offset & UV Scale (`vec4<f32>`)
  * `texture_2d<f32>` for Elevation DEM (`R16Unorm` or `R32Float`)
  * `texture_2d<f32>` for Albedo (`Bc7RgbaUnorm` / `Rgba8UnormSrgb`)
* **Lighting Model:**
  * Implement **Lommel-Seeliger / Hapke Lunar BRDF** rather than standard Lambertian diffuse. Lunar regolith exhibits strong retro-reflection (opposition surge) and zero atmospheric scattering, giving the Moon its high-contrast, dusty appearance.

---

## 6. Step-by-Step Implementation Roadmap

| Milestone | Deliverable | Verification Criteria |
| :--- | :--- | :--- |
| **M1: Engine Heartbeat** | Bypass `maxInterStageShaderComponents` error; verify lime-green/deep-space clear color on `<canvas id="viewport">`. | HUD displays `WebGPU Engine Online` and canvas updates color on rebuild. |
| **M2: First WGSL Pipeline** | Add `shader.wgsl`, Depth32Float buffer, and render a rotating 3D procedural wireframe sphere in the viewport. | 60 FPS procedural sphere visible in browser with mouse orbit controls. |
| **M3: R2 Texture Binding** | Fetch `test_tile.ktx2` (or raw RGBA/R16 binary tile) from R2 into Rust `Vec<u8>` and upload to `wgpu::Texture`. | Sphere renders with live texture fetched from Cloudflare R2 edge. |
| **M4: Quadtree & Displacement** | Replace static sphere with 6-face quadtree chunks and displace vertices in WGSL using LOLA DEM data. | Real 3D lunar craters (Tycho central peak & rim) cast accurate relief shadows. |
| **M5: HUD & Telemetry** | Connect mouse raycasting to display Lunar Latitude/Longitude, Altitude, and active R2 tile streaming stats. | Clicking a crater displays exact selenographic coordinates on the HUD. |