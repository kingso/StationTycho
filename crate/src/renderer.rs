use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::camera::OrbitCamera;
use crate::geometry::{
    enu_frame, grid_attributes, grid_indices, grid_vertex_count, texel_spacing, tile_vertices, Face, MOON_RADIUS,
};
use crate::lighting::{sun_direction, wrap_longitude_deg, DEFAULT_SUBSOLAR_LON_DEG};
use crate::lod::{select_tiles, skirt_depths, LodConfig, TileId, MAX_VISIBLE_TILES, TILE_SEGMENTS};
use crate::manifest::{DemQuantization, Manifest};
use crate::terrain::{grid_heights, ray_terrain, DrawnTerrain};
use crate::streaming::{
    tile_gpu_bytes, Completion, FetchResult, TileCache, GPU_TILE_BUDGET_BYTES, MAX_IN_FLIGHT_TILES,
};
use crate::tiles::{manifest_path, tile_path, Layer, TILE_SIZE};
use glam::DVec3;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::console;
use wgpu::util::DeviceExt;

const TERRAIN_SHADER: &str = include_str!("shaders/terrain.wgsl");
const WIREFRAME_SHADER: &str = include_str!("shaders/wireframe.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const FRAME_UNIFORM_SIZE: u64 = 96;
const OBJECT_UNIFORM_SIZE: u64 = 80;
/// Marker vertices: `[offset.xyz, normal.xyz, uv.xy]`.
const MARKER_VERTEX_STRIDE: u64 = 32;
/// Tile vertices: an f32 offset from the tile centre (buffer 0), plus shared `[u, v, skirt]` (buffer 1).
const TILE_VERTEX_STRIDE: u64 = 12;

const DEBUG_WIREFRAME: u32 = 1;
/// Clears to magenta instead of black, so any crack in the terrain stands out.
const DEBUG_CRACKS: u32 = 4;

const ALBEDO_BYTES: usize = (TILE_SIZE * TILE_SIZE * 4) as usize;
const DEM_BYTES: usize = (TILE_SIZE * TILE_SIZE * 2) as usize;

const TYCHO_LAT_DEG: f64 = -43.31;
const TYCHO_LON_DEG: f64 = -11.36;

fn face_color(face: Face) -> [f32; 4] {
    match face {
        Face::Px => [0.43, 0.66, 1.0, 1.0],
        Face::Nx => [0.30, 0.48, 0.85, 1.0],
        Face::Py => [0.45, 0.85, 0.90, 1.0],
        Face::Ny => [0.30, 0.62, 0.70, 1.0],
        Face::Pz => [0.75, 0.72, 1.0, 1.0],
        Face::Nz => [0.52, 0.48, 0.82, 1.0],
    }
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn u32_bytes(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|i| i.to_le_bytes()).collect()
}

fn index_buffer(device: &wgpu::Device, label: &str, indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &u32_bytes(indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

/// Mesh whose vertices are f32 offsets from an f64 `origin` in the body-fixed frame.
struct Mesh {
    vertex_buffer: wgpu::Buffer,
    line_buffer: wgpu::Buffer,
    line_count: u32,
    origin: DVec3,
    color: [f32; 4],
}

impl Mesh {
    /// `vertices` holds interleaved `[offset.xyz, normal.xyz, uv.xy]`; `lines` are line-list pairs.
    fn new(device: &wgpu::Device, label: &str, origin: DVec3, vertices: &[f32], lines: &[u32], color: [f32; 4]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &f32_bytes(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self { vertex_buffer, line_buffer: index_buffer(device, label, lines), line_count: lines.len() as u32, origin, color }
    }
}

/// Index buffers shared by every tile, which all use the same `TILE_SEGMENTS` grid.
struct TileGrid {
    triangles: wgpu::Buffer,
    triangle_count: u32,
    lines: wgpu::Buffer,
    line_count: u32,
    /// `[u, v, skirt]` per vertex, the second vertex buffer of every tile.
    attributes: wgpu::Buffer,
}

impl TileGrid {
    fn new(device: &wgpu::Device) -> Self {
        let (triangles, lines) = grid_indices(TILE_SEGMENTS);
        Self {
            triangles: index_buffer(device, "Tile Grid Triangles", &triangles),
            triangle_count: triangles.len() as u32,
            lines: index_buffer(device, "Tile Grid Lines", &lines),
            line_count: lines.len() as u32,
            attributes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tile Grid Attributes"),
                contents: &f32_bytes(&grid_attributes(TILE_SEGMENTS)),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        }
    }
}

/// GPU resources of one tile-cache slot, reused by whichever tile the cache assigns to it.
struct TileSlot {
    albedo: wgpu::Texture,
    dem: wgpu::Texture,
    vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Centre of the tile now in the slot; its vertices are offsets from it.
    origin: DVec3,
    /// The face's u and v axes with the metres per DEM texel along them, for terrain normals.
    u_axis: [f32; 4],
    v_axis: [f32; 4],
    /// Quantized heights under the grid vertices, for picking and keeping the camera above ground.
    grid: Vec<u16>,
}

impl TileSlot {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler) -> Self {
        let albedo = create_tile_texture(device, "Tile Albedo", wgpu::TextureFormat::Rgba8UnormSrgb);
        let dem = create_tile_texture(device, "Tile DEM", wgpu::TextureFormat::R16Uint);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Vertices"),
            size: TILE_VERTEX_STRIDE * u64::from(grid_vertex_count(TILE_SEGMENTS)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile Textures"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dem.create_view(&Default::default())),
                },
            ],
        });
        Self { albedo, dem, vertex_buffer, bind_group, origin: DVec3::ZERO, u_axis: [0.0; 4], v_axis: [0.0; 4], grid: Vec::new() }
    }

    fn upload(&mut self, queue: &wgpu::Queue, tile: TileId, albedo: &[u8], dem: &[u8]) {
        let (origin, vertices) =
            tile_vertices(tile.face, tile.level, tile.x, tile.y, TILE_SEGMENTS).expect("cached tiles are inside the quadtree");
        write_tile(queue, &self.albedo, albedo, 4);
        write_tile(queue, &self.dem, dem, 2);
        queue.write_buffer(&self.vertex_buffer, 0, &f32_bytes(&vertices));
        self.origin = origin;
        let (_, u_axis, v_axis) = tile.face.axes();
        let (du, dv) = texel_spacing(tile.face, tile.level, tile.x, tile.y).expect("cached tiles are inside the quadtree");
        self.u_axis = u_axis.as_vec3().extend(du as f32).to_array();
        self.v_axis = v_axis.as_vec3().extend(dv as f32).to_array();
        self.grid = grid_heights(dem);
    }
}

/// Tile cache plus the GPU slots and open requests it manages; exists once the manifest has loaded.
struct Streaming {
    cache: TileCache,
    slots: Vec<TileSlot>,
    requests: HashMap<u64, web_sys::AbortController>,
}

/// A finished tile fetch, processed on the next frame.
struct Fetched {
    tile: TileId,
    request: u64,
    result: Result<(Vec<u8>, Vec<u8>), FetchResult>,
}

type Inbox = Rc<RefCell<Vec<Fetched>>>;

/// 20 m wireframe cube standing on the reference sphere at Tycho's centre.
fn tycho_marker(device: &wgpu::Device) -> Mesh {
    const HALF: f64 = 10.0;
    let (east, north, up) = enu_frame(TYCHO_LAT_DEG.to_radians(), TYCHO_LON_DEG.to_radians());
    let mut vertices = Vec::with_capacity(8 * 8);
    for corner in 0..8u32 {
        let x = if corner & 1 == 0 { -HALF } else { HALF };
        let y = if corner & 2 == 0 { -HALF } else { HALF };
        let z = if corner & 4 == 0 { 0.0 } else { 2.0 * HALF };
        let offset = (east * x + north * y + up * z).as_vec3();
        let normal = up.as_vec3();
        vertices.extend_from_slice(&[offset.x, offset.y, offset.z, normal.x, normal.y, normal.z, 0.0, 0.0]);
    }
    let mut indices = Vec::with_capacity(24);
    for corner in 0..8u32 {
        for bit in [1, 2, 4] {
            if corner & bit == 0 {
                indices.extend_from_slice(&[corner, corner | bit]);
            }
        }
    }
    Mesh::new(device, "Tycho Marker", up * MOON_RADIUS, &vertices, &indices, [1.0, 0.45, 0.2, 1.0])
}

/// Heights of the drawn `visible` tiles.
fn drawn_terrain<'a>(streaming: Option<&'a Streaming>, visible: &'a [TileId], dem: DemQuantization) -> Option<DrawnTerrain<'a>> {
    let streaming = streaming?;
    let tiles = visible.iter().filter_map(|&tile| Some((tile, streaming.slots[streaming.cache.slot(tile)?].grid.as_slice())));
    Some(DrawnTerrain::new(tiles, dem))
}

/// `[latitude°, longitude°, height above the reference sphere m]` of a body-fixed point.
fn location(point: DVec3) -> Vec<f64> {
    let r = point.length();
    vec![(point.z / r).asin().to_degrees(), point.y.atan2(point.x).to_degrees(), r - MOON_RADIUS]
}

/// Per-object uniform: origin relative to the camera with the skirt depth, the wireframe colour, the origin itself
/// (only used for directions), then the face axes with their texel spacing (zero for non-terrain objects).
fn object_uniform(origin: DVec3, skirt_depth: f64, color: [f32; 4], axes: ([f32; 4], [f32; 4]), camera_position: DVec3) -> Vec<u8> {
    let rel = (origin - camera_position).as_vec3();
    let center = origin.as_vec3();
    let mut data = vec![rel.x, rel.y, rel.z, skirt_depth as f32];
    data.extend_from_slice(&color);
    data.extend_from_slice(&[center.x, center.y, center.z, 0.0]);
    data.extend_from_slice(&axes.0);
    data.extend_from_slice(&axes.1);
    f32_bytes(&data)
}

fn create_tile_texture(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn write_tile(queue: &wgpu::Queue, texture: &wgpu::Texture, data: &[u8], bytes_per_pixel: u32) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TILE_SIZE * bytes_per_pixel),
            rows_per_image: Some(TILE_SIZE),
        },
        wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
    );
}

async fn read_body(response: JsFuture) -> Result<Vec<u8>, FetchResult> {
    let response: web_sys::Response = response.await.and_then(|r| r.dyn_into()).map_err(|_| FetchResult::Failed)?;
    match response.status() {
        404 => return Err(FetchResult::NotFound),
        _ if !response.ok() => return Err(FetchResult::Failed),
        _ => {}
    }
    let body = response.array_buffer().map_err(|_| FetchResult::Failed)?;
    let buffer = JsFuture::from(body).await.map_err(|_| FetchResult::Failed)?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Fetches a tile's albedo and DEM in parallel; `signal` aborts both.
async fn fetch_tile_data(tile: TileId, signal: &web_sys::AbortSignal) -> Result<(Vec<u8>, Vec<u8>), FetchResult> {
    let window = web_sys::window().ok_or(FetchResult::Failed)?;
    let init = web_sys::RequestInit::new();
    init.set_signal(Some(signal));
    let path = |layer| tile_path(layer, tile.face, u32::from(tile.level), tile.x, tile.y);
    // Wrapping both promises now attaches their handlers, and both are always awaited, so aborting never leaves
    // an unhandled rejection.
    let albedo = JsFuture::from(window.fetch_with_str_and_init(&path(Layer::Albedo), &init));
    let dem = JsFuture::from(window.fetch_with_str_and_init(&path(Layer::Dem), &init));
    let (albedo, dem) = (read_body(albedo).await, read_body(dem).await);
    let (albedo, dem) = (albedo?, dem?);
    if albedo.len() != ALBEDO_BYTES || dem.len() != DEM_BYTES {
        return Err(FetchResult::Corrupt);
    }
    Ok((albedo, dem))
}

/// Starts fetching `tile`; the result lands in `inbox` unless the returned controller aborts it first.
fn spawn_fetch(inbox: &Inbox, tile: TileId, request: u64) -> Option<web_sys::AbortController> {
    let controller = web_sys::AbortController::new().ok();
    let signal = controller.as_ref().map(|c| c.signal());
    let inbox = inbox.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let result = match &signal {
            Some(signal) => fetch_tile_data(tile, signal).await,
            None => Err(FetchResult::Failed),
        };
        if !signal.is_some_and(|s| s.aborted()) {
            inbox.borrow_mut().push(Fetched { tile, request, result });
        }
    });
    controller
}

/// The dataset manifest, from [`fetch_manifest`] for [`Renderer::set_manifest`].
#[wasm_bindgen]
pub struct TileManifest(Manifest);

/// Fetches and parses the dataset manifest.
#[wasm_bindgen]
pub async fn fetch_manifest() -> Result<TileManifest, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No global window"))?;
    let path = manifest_path();
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(&path)).await?.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!("{path}: HTTP {}", response.status())));
    }
    let text = JsFuture::from(response.text()?).await?.as_string().unwrap_or_default();
    Manifest::parse(&text).map(TileManifest).map_err(|e| JsValue::from_str(&e))
}

fn create_depth_target(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[wasm_bindgen]
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    terrain_pipeline: wgpu::RenderPipeline,
    terrain_under_wire_pipeline: wgpu::RenderPipeline,
    tile_wire_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    object_stride: u64,
    lod_config: LodConfig,
    dem: DemQuantization,
    tile_grid: TileGrid,
    tile_layout: wgpu::BindGroupLayout,
    albedo_sampler: wgpu::Sampler,
    streaming: Option<Streaming>,
    inbox: Inbox,
    /// Resident tiles drawn this frame; the LOD selector only returns resident tiles.
    visible_tiles: Vec<TileId>,
    marker: Mesh,
    frame_index: u64,
    debug_flags: u32,
    camera: OrbitCamera,
    sun_lon_deg: f64,
    /// sRGB view of the surface texture, so linear shader output is encoded for display.
    view_format: wgpu::TextureFormat,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

#[wasm_bindgen]
impl Renderer {
    /// Resizes the drawing surface and depth buffer to `width` x `height` device pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        let width = width.clamp(1, max);
        let height = height.clamp(1, max);
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let (texture, view) = create_depth_target(&self.device, width, height);
        self._depth_texture = texture;
        self.depth_view = view;
    }

    /// Pans the camera target by a drag of `dx`/`dy` device pixels.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.camera.pan(dx, dy, self.config.height as f64);
    }

    /// Changes camera heading and tilt by a drag of `dx`/`dy` device pixels.
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.camera.rotate(dx, dy);
    }

    /// Multiplies the camera's distance to its target by `factor`.
    pub fn zoom(&mut self, factor: f64) {
        self.camera.zoom(factor);
    }

    /// Camera height above the terrain under it, in metres.
    pub fn altitude(&self) -> f64 {
        self.camera.altitude()
    }

    /// Bit 0 draws the wireframe over the terrain; bit 1 shows heights instead of albedo; bit 2 clears to magenta
    /// so cracks stand out; bit 3 turns skirts off.
    pub fn set_debug_flags(&mut self, flags: u32) {
        self.debug_flags = flags;
    }

    /// Latitude and longitude in degrees and terrain height in metres of the point the camera orbits.
    pub fn target_location(&self) -> Vec<f64> {
        let target = self.camera.target();
        location(target)
    }

    /// The terrain point under device pixel `(x, y)` of the canvas as `[latitude°, longitude°, height m]`, or
    /// nothing if the pixel shows sky. Only drawn tiles count, so it matches what is on screen.
    pub fn pick(&self, x: f64, y: f64) -> Option<Vec<f64>> {
        let (width, height) = (self.config.width as f64, self.config.height as f64);
        let ray = self.camera.ray(2.0 * x / width - 1.0, 1.0 - 2.0 * y / height, width / height);
        let terrain = drawn_terrain(self.streaming.as_ref(), &self.visible_tiles, self.dem)?;
        ray_terrain(self.camera.position(), ray, |dir| terrain.height_at(dir)).map(location)
    }

    /// Longitude of the sub-solar point in degrees (east positive).
    pub fn sun_longitude(&self) -> f64 {
        self.sun_lon_deg
    }

    /// Moves the sun to sub-solar longitude `lon_deg`, wrapped to [-180, 180).
    pub fn set_sun_longitude(&mut self, lon_deg: f64) {
        if lon_deg.is_finite() {
            self.sun_lon_deg = wrap_longitude_deg(lon_deg);
        }
    }

    /// Number of quadtree tiles drawn in the last frame.
    pub fn visible_tile_count(&self) -> u32 {
        self.visible_tiles.len() as u32
    }

    /// Deepest quadtree level drawn in the last frame.
    pub fn finest_visible_level(&self) -> u32 {
        self.visible_tiles.iter().map(|t| u32::from(t.level)).max().unwrap_or(0)
    }

    /// Tiles resident on the GPU.
    pub fn resident_tile_count(&self) -> u32 {
        self.streaming.as_ref().map_or(0, |s| s.cache.resident() as u32)
    }

    /// Most tiles the GPU budget allows.
    pub fn tile_capacity(&self) -> u32 {
        self.streaming.as_ref().map_or(0, |s| s.cache.capacity() as u32)
    }

    /// GPU memory allocated for tile slots, in MiB.
    pub fn tile_gpu_megabytes(&self) -> f64 {
        self.streaming.as_ref().map_or(0.0, |s| (s.slots.len() as u64 * tile_gpu_bytes()) as f64 / (1u64 << 20) as f64)
    }

    /// Tiles being fetched.
    pub fn loading_tile_count(&self) -> u32 {
        self.streaming.as_ref().map_or(0, |s| s.cache.in_flight() as u32)
    }

    /// Tile fetches cancelled so far because the view moved on.
    pub fn cancelled_tile_count(&self) -> f64 {
        self.streaming.as_ref().map_or(0.0, |s| s.cache.stats().cancelled as f64)
    }

    /// Tile fetches that failed so far (network or server errors, retried with backoff).
    pub fn failed_tile_count(&self) -> f64 {
        self.streaming.as_ref().map_or(0.0, |s| s.cache.stats().failed as f64)
    }

    /// Starts streaming tiles listed in `manifest`; nothing is drawn before this.
    pub fn set_manifest(&mut self, manifest: TileManifest) {
        let manifest = manifest.0;
        if let Some(old) = self.streaming.take() {
            old.requests.values().for_each(web_sys::AbortController::abort);
        }
        self.dem = manifest.dem;
        self.lod_config = LodConfig {
            max_level: manifest.max_level(),
            terrain_error_m: manifest.geometric_error_m.clone(),
            ..LodConfig::default()
        };
        let capacity = (GPU_TILE_BUDGET_BYTES / tile_gpu_bytes()) as usize;
        self.streaming = Some(Streaming {
            cache: TileCache::new(manifest, capacity, MAX_IN_FLIGHT_TILES),
            slots: Vec::new(),
            requests: HashMap::new(),
        });
    }

    /// Renders one frame. `time_ms` is the requestAnimationFrame timestamp.
    pub fn render(&mut self, time_ms: f64) -> Result<(), JsValue> {
        let aspect = self.config.width as f64 / self.config.height as f64;
        self.frame_index += 1;
        self.update_tiles(aspect, time_ms);
        self.follow_terrain();
        if self.visible_tiles.len() > MAX_VISIBLE_TILES {
            return Err(JsValue::from_str(&format!("{} tiles exceed the {MAX_VISIBLE_TILES}-tile buffer", self.visible_tiles.len())));
        }
        let skirts = skirt_depths(&self.visible_tiles, &self.lod_config);
        let drawn: Vec<(&TileSlot, f64, [f32; 4])> = match &self.streaming {
            Some(streaming) => self
                .visible_tiles
                .iter()
                .zip(skirts)
                .map(|(tile, skirt)| {
                    let slot = streaming.cache.slot(*tile).expect("selected tiles are resident");
                    (&streaming.slots[slot], skirt, face_color(tile.face))
                })
                .collect(),
            None => Vec::new(),
        };

        let view_proj = self.camera.view_proj(aspect);
        let mut frame_data = f32_bytes(&view_proj.to_cols_array());
        frame_data.extend_from_slice(&u32_bytes(&[self.debug_flags]));
        frame_data.extend_from_slice(&f32_bytes(&[self.dem.scale as f32, self.dem.offset as f32, 0.0]));
        let sun = sun_direction(self.sun_lon_deg).as_vec3();
        frame_data.extend_from_slice(&f32_bytes(&[sun.x, sun.y, sun.z, 0.0]));
        self.queue.write_buffer(&self.frame_buffer, 0, &frame_data);

        // Visible tiles take the first uniform slots and the marker the last.
        let camera_pos = self.camera.position();
        let stride = self.object_stride as usize;
        let visible_count = drawn.len();
        let mut objects = vec![0u8; stride * (visible_count + 1)];
        for (i, (slot, skirt, color)) in drawn.iter().enumerate() {
            objects[i * stride..i * stride + OBJECT_UNIFORM_SIZE as usize]
                .copy_from_slice(&object_uniform(slot.origin, *skirt, *color, (slot.u_axis, slot.v_axis), camera_pos));
        }
        objects[visible_count * stride..visible_count * stride + OBJECT_UNIFORM_SIZE as usize]
            .copy_from_slice(&object_uniform(self.marker.origin, 0.0, self.marker.color, ([0.0; 4], [0.0; 4]), camera_pos));
        self.queue.write_buffer(&self.object_buffer, 0, &objects);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            other => return Err(JsValue::from_str(&format!("Surface error: {:?}", other))),
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(match self.debug_flags & DEBUG_CRACKS {
                            0 => wgpu::Color::BLACK,
                            _ => wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 },
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);

            let show_wireframe = self.debug_flags & DEBUG_WIREFRAME != 0;
            pass.set_pipeline(if show_wireframe { &self.terrain_under_wire_pipeline } else { &self.terrain_pipeline });
            pass.set_vertex_buffer(1, self.tile_grid.attributes.slice(..));
            pass.set_index_buffer(self.tile_grid.triangles.slice(..), wgpu::IndexFormat::Uint32);
            for (i, (slot, _, _)) in drawn.iter().enumerate() {
                pass.set_bind_group(1, &self.object_bind_group, &[(i as u64 * self.object_stride) as u32]);
                pass.set_bind_group(2, &slot.bind_group, &[]);
                pass.set_vertex_buffer(0, slot.vertex_buffer.slice(..));
                pass.draw_indexed(0..self.tile_grid.triangle_count, 0, 0..1);
            }

            if show_wireframe {
                pass.set_pipeline(&self.tile_wire_pipeline);
                pass.set_index_buffer(self.tile_grid.lines.slice(..), wgpu::IndexFormat::Uint32);
                for (i, (slot, _, _)) in drawn.iter().enumerate() {
                    pass.set_bind_group(1, &self.object_bind_group, &[(i as u64 * self.object_stride) as u32]);
                    pass.set_bind_group(2, &slot.bind_group, &[]);
                    pass.set_vertex_buffer(0, slot.vertex_buffer.slice(..));
                    pass.draw_indexed(0..self.tile_grid.line_count, 0, 0..1);
                }
            }
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(1, &self.object_bind_group, &[(visible_count as u64 * self.object_stride) as u32]);
            pass.set_vertex_buffer(0, self.marker.vertex_buffer.slice(..));
            pass.set_index_buffer(self.marker.line_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.marker.line_count, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        Ok(())
    }

    /// Moves the orbit target onto the drawn terrain and keeps the camera above it. The target height eases
    /// towards the terrain, so a tile refining to slightly different heights does not make the view jump.
    fn follow_terrain(&mut self) {
        const EASE: f64 = 0.2;
        const CLEARANCE_M: f64 = 1.0;
        let Some(terrain) = drawn_terrain(self.streaming.as_ref(), &self.visible_tiles, self.dem) else { return };
        let camera = &mut self.camera;
        if let Some(height) = terrain.height_at(camera.target().normalize()) {
            camera.target_height += (height - camera.target_height) * EASE;
        }
        let fallback = camera.target_height;
        camera.lift_above_terrain(|dir| terrain.height_at(dir).unwrap_or(fallback), CLEARANCE_M);
        if let Some(height) = terrain.height_at(camera.position().normalize()) {
            camera.ground_height = height;
        }
    }

    /// Selects this frame's tiles, then uploads finished fetches and starts or cancels fetches for the selection.
    fn update_tiles(&mut self, aspect: f64, now_ms: f64) {
        let frame = self.frame_index;
        let Some(streaming) = &mut self.streaming else {
            self.visible_tiles.clear();
            return;
        };
        let cache = &streaming.cache;
        let selection = select_tiles(&self.camera, self.config.height, aspect, &self.lod_config, |t| cache.status(t));
        // Visible tiles and their ancestors (all resident, since selection only splits ready tiles) are marked
        // used, so the uploads below cannot evict them and zooming out can merge back without reloading.
        for &tile in &selection.tiles {
            let mut current = Some(tile);
            while let Some(t) = current.filter(|&t| streaming.cache.touch(t, frame)) {
                current = t.parent();
            }
        }
        self.visible_tiles = selection.tiles;

        let fetched = std::mem::take(&mut *self.inbox.borrow_mut());
        for Fetched { tile, request, result } in fetched {
            streaming.requests.remove(&request);
            let outcome = result.as_ref().err().copied().unwrap_or(FetchResult::Loaded);
            if let (Completion::Upload { slot, .. }, Ok((albedo, dem))) =
                (streaming.cache.complete(tile, request, outcome, frame, now_ms), &result)
            {
                while streaming.slots.len() <= slot {
                    streaming.slots.push(TileSlot::new(&self.device, &self.tile_layout, &self.albedo_sampler));
                }
                streaming.slots[slot].upload(&self.queue, tile, albedo, dem);
            }
        }

        let (start, cancel) = streaming.cache.schedule(&selection.requests, frame, now_ms);
        for (_, request) in cancel {
            if let Some(controller) = streaming.requests.remove(&request) {
                controller.abort();
            }
        }
        for (tile, request) in start {
            if let Some(controller) = spawn_fetch(&self.inbox, tile, request) {
                streaming.requests.insert(request, controller);
            }
        }
    }
}

#[wasm_bindgen]
pub async fn init_render_engine(canvas_id: &str, width: u32, height: u32) -> Result<Renderer, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No global window"))?;
    let document = window.document().ok_or_else(|| JsValue::from_str("No document"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str("Canvas not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    console::log_1(&"Initializing wgpu instance...".into());

    // 1. Create wgpu instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    // 2. Create surface from canvas element (pass by value, not by reference)
    let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
    let surface = instance
        .create_surface(surface_target)
        .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

    // 3. Request WebGPU adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to find a WebGPU adapter: {:?}", e)))?;

    console::log_1(&"WebGPU adapter acquired successfully.".into());

    // 4. Request device and queue with spec-default limits so behaviour matches across GPUs
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("StationTycho Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to request device: {:?}", e)))?;

    // 5. Configure surface dimensions and format
    let max_dim = device.limits().max_texture_dimension_2d;
    let width = width.clamp(1, max_dim);
    let height = height.clamp(1, max_dim);
    let caps = surface.get_capabilities(&adapter);
    let surface_format = caps.formats[0];
    // Browsers prefer a non-sRGB canvas (bgra8unorm); render through its sRGB view so shading stays linear.
    let view_format = surface_format.add_srgb_suffix();

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: if view_format == surface_format { vec![] } else { vec![view_format] },
        desired_maximum_frame_latency: 2,
        color_space: Default::default(),
    };
    surface.configure(&device, &config);

    console::log_1(&"wgpu surface configured. Building pipelines...".into());

    // 6. Build the terrain and line pipelines
    let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Terrain Shader"),
        source: wgpu::ShaderSource::Wgsl(TERRAIN_SHADER.into()),
    });
    let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Wireframe Shader"),
        source: wgpu::ShaderSource::Wgsl(WIREFRAME_SHADER.into()),
    });

    let marker = tycho_marker(&device);

    let object_stride = OBJECT_UNIFORM_SIZE.max(device.limits().min_uniform_buffer_offset_alignment as u64);
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frame Uniforms"),
        size: FRAME_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Object Uniforms"),
        size: object_stride * (MAX_VISIBLE_TILES + 1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_layout_entry = |dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    };
    let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Frame Layout"),
        entries: &[uniform_layout_entry(false)],
    });
    let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Object Layout"),
        entries: &[uniform_layout_entry(true)],
    });

    let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Frame Bind Group"),
        layout: &frame_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: frame_buffer.as_entire_binding(),
        }],
    });
    let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Object Bind Group"),
        layout: &object_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &object_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(OBJECT_UNIFORM_SIZE),
            }),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Wireframe Pipeline Layout"),
        bind_group_layouts: &[Some(&frame_layout), Some(&object_layout)],
        immediate_size: 0,
    });

    let texture_entry = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let tile_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Tile Texture Layout"),
        entries: &[
            texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            texture_entry(2, wgpu::TextureSampleType::Uint),
        ],
    });
    let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Terrain Pipeline Layout"),
        bind_group_layouts: &[Some(&frame_layout), Some(&object_layout), Some(&tile_layout)],
        immediate_size: 0,
    });

    let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Albedo Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    // With the wireframe on, the surface is pushed back by one pixel's worth of its own depth slope (negative is
    // farther with reversed-Z), so lines along its edges still win at grazing angles, where the surface half a pixel
    // nearer would otherwise beat the line shader's constant 0.1% pull. WebGPU applies depth bias to triangles only.
    // It is not used otherwise: on near-vertical skirts the bias can exceed the depth itself, which clamps to the
    // far plane and leaves pinholes.
    let wireframe_bias = wgpu::DepthBiasState { constant: 0, slope_scale: -1.0, clamp: 0.0 };
    let tile_buffers = [
        Some(wgpu::VertexBufferLayout {
            array_stride: TILE_VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: TILE_VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![1 => Float32x3],
        }),
    ];
    let marker_buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: MARKER_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
    })];
    let line_list = wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() };
    let terrain = |label, bias| {
        create_pipeline(
            &device,
            label,
            (&terrain_shader, "vs_main", "fs_main"),
            &tile_buffers,
            &terrain_layout,
            view_format,
            wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            bias,
        )
    };
    let terrain_pipeline = terrain("Terrain Pipeline", wgpu::DepthBiasState::default());
    let terrain_under_wire_pipeline = terrain("Terrain Under Wireframe Pipeline", wireframe_bias);
    let tile_wire_pipeline = create_pipeline(
        &device,
        "Tile Wireframe Pipeline",
        (&terrain_shader, "vs_wire", "fs_wire"),
        &tile_buffers,
        &terrain_layout,
        view_format,
        line_list,
        wgpu::DepthBiasState::default(),
    );
    let line_pipeline = create_pipeline(
        &device,
        "Marker Pipeline",
        (&line_shader, "vs_main", "fs_main"),
        &marker_buffers,
        &pipeline_layout,
        view_format,
        line_list,
        wgpu::DepthBiasState::default(),
    );

    let (depth_texture, depth_view) = create_depth_target(&device, width, height);
    let tile_grid = TileGrid::new(&device);

    console::log_1(&"StationTycho WebGPU pipelines ready.".into());
    Ok(Renderer {
        surface,
        device,
        queue,
        config,
        terrain_pipeline,
        terrain_under_wire_pipeline,
        tile_wire_pipeline,
        line_pipeline,
        frame_buffer,
        frame_bind_group,
        object_buffer,
        object_bind_group,
        object_stride,
        lod_config: LodConfig::default(),
        dem: DemQuantization { scale: 1.0, offset: 0.0 },
        tile_grid,
        tile_layout,
        albedo_sampler,
        streaming: None,
        inbox: Rc::new(RefCell::new(Vec::new())),
        visible_tiles: Vec::new(),
        marker,
        frame_index: 0,
        debug_flags: 0,
        camera: OrbitCamera::new(TYCHO_LAT_DEG, TYCHO_LON_DEG, 3.0 * MOON_RADIUS),
        sun_lon_deg: DEFAULT_SUBSOLAR_LON_DEG,
        view_format,
        _depth_texture: depth_texture,
        depth_view,
    })
}

/// Depth-tested pipeline (reversed-Z: nearer is greater) drawing into the surface format.
#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    (shader, vertex_entry, fragment_entry): (&wgpu::ShaderModule, &str, &str),
    buffers: &[Option<wgpu::VertexBufferLayout>],
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    primitive: wgpu::PrimitiveState,
    bias: wgpu::DepthBiasState,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive,
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias,
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}