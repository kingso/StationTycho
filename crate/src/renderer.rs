use crate::camera::OrbitCamera;
use crate::geometry::{enu_frame, face_chunks, Face, MOON_RADIUS};
use crate::tiles::{tile_path, Layer, TILE_SIZE};
use glam::DVec3;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;
use wgpu::util::DeviceExt;

const TERRAIN_SHADER: &str = include_str!("shaders/terrain.wgsl");
const WIREFRAME_SHADER: &str = include_str!("shaders/wireframe.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const FRAME_UNIFORM_SIZE: u64 = 80;
const OBJECT_UNIFORM_SIZE: u64 = 32;
const VERTEX_STRIDE: u64 = 32;

const DEBUG_WIREFRAME: u32 = 1;

const ALBEDO_BYTES: usize = (TILE_SIZE * TILE_SIZE * 4) as usize;
const DEM_BYTES: usize = (TILE_SIZE * TILE_SIZE * 2) as usize;

const TYCHO_LAT_DEG: f64 = -43.31;
const TYCHO_LON_DEG: f64 = -11.36;

// 4×4 chunks of 8×8 cells per face, chunked so f32 offsets stay small.
const SPHERE_CHUNKS_PER_FACE: u32 = 4;
const SPHERE_SEGMENTS_PER_CHUNK: u32 = 8;

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

fn sphere_meshes(device: &wgpu::Device) -> Vec<Mesh> {
    Face::ALL
        .iter()
        .enumerate()
        .flat_map(|(face_index, &face)| {
            face_chunks(face, SPHERE_CHUNKS_PER_FACE, SPHERE_SEGMENTS_PER_CHUNK)
                .into_iter()
                .map(move |chunk| (face_index, face, chunk))
        })
        .map(|(face_index, face, chunk)| {
            let mut mesh = Mesh::new(device, "Sphere Chunk", chunk.origin, &chunk.vertices, &chunk.lines, face_color(face));
            mesh.surface = Some(Surface {
                face: face_index,
                index_buffer: index_buffer(device, "Sphere Chunk Triangles", &chunk.triangles),
                index_count: chunk.triangles.len() as u32,
            });
            mesh
        })
        .collect()
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

/// Triangles of a mesh textured from one cube face.
struct Surface {
    face: usize,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// Mesh whose vertices are f32 offsets from an f64 `origin` in the body-fixed frame.
struct Mesh {
    vertex_buffer: wgpu::Buffer,
    line_buffer: wgpu::Buffer,
    line_count: u32,
    surface: Option<Surface>,
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
        Self {
            vertex_buffer,
            line_buffer: index_buffer(device, label, lines),
            line_count: lines.len() as u32,
            surface: None,
            origin,
            color,
        }
    }
}

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

/// Albedo and height textures for one cube face, starting as flat grey placeholders.
struct FaceTextures {
    albedo: wgpu::Texture,
    dem: wgpu::Texture,
    bind_group: wgpu::BindGroup,
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

async fn fetch_bytes(path: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No global window"))?;
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(path)).await?.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!("{path}: HTTP {}", response.status())));
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Level-0 albedo and height tiles for one cube face.
#[wasm_bindgen]
pub struct FaceTile {
    face: usize,
    albedo: Vec<u8>,
    dem: Vec<u8>,
}

/// Fetches the level-0 tiles for face `face` (0..6, in `Face::ALL` order) from R2.
#[wasm_bindgen]
pub async fn fetch_face_tile(face: usize) -> Result<FaceTile, JsValue> {
    let key = *Face::ALL.get(face).ok_or_else(|| JsValue::from_str(&format!("No face {face}")))?;
    let albedo_path = tile_path(Layer::Albedo, key, 0, 0, 0);
    let dem_path = tile_path(Layer::Dem, key, 0, 0, 0);
    let albedo = fetch_bytes(&albedo_path).await?;
    let dem = fetch_bytes(&dem_path).await?;
    if albedo.len() != ALBEDO_BYTES || dem.len() != DEM_BYTES {
        return Err(JsValue::from_str(&format!(
            "{albedo_path} / {dem_path}: expected {ALBEDO_BYTES} / {DEM_BYTES} bytes, got {} / {}",
            albedo.len(),
            dem.len()
        )));
    }
    Ok(FaceTile { face, albedo, dem })
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
    line_pipeline: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    object_stride: u64,
    meshes: Vec<Mesh>,
    faces: Vec<FaceTextures>,
    debug_flags: u32,
    camera: OrbitCamera,
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

    /// Camera height above the reference sphere, in metres.
    pub fn altitude(&self) -> f64 {
        self.camera.altitude()
    }

    /// Bit 0 draws the wireframe over the terrain; bit 1 shows heights instead of albedo.
    pub fn set_debug_flags(&mut self, flags: u32) {
        self.debug_flags = flags;
    }

    /// Uploads a face's tiles from [`fetch_face_tile`] into its GPU textures.
    pub fn upload_face_tile(&mut self, tile: FaceTile) {
        let face = &self.faces[tile.face];
        write_tile(&self.queue, &face.albedo, &tile.albedo, 4);
        write_tile(&self.queue, &face.dem, &tile.dem, 2);
    }

    /// Renders one frame. `_time_ms` is the requestAnimationFrame timestamp.
    pub fn render(&mut self, _time_ms: f64) -> Result<(), JsValue> {
        let aspect = self.config.width as f64 / self.config.height as f64;
        let view_proj = self.camera.view_proj(aspect);
        let mut frame_data = f32_bytes(&view_proj.to_cols_array());
        frame_data.extend_from_slice(&u32_bytes(&[self.debug_flags, 0, 0, 0]));
        self.queue.write_buffer(&self.frame_buffer, 0, &frame_data);

        let camera_pos = self.camera.position();
        let mut objects = vec![0u8; self.object_stride as usize * self.meshes.len()];
        for (i, mesh) in self.meshes.iter().enumerate() {
            let rel = (mesh.origin - camera_pos).as_vec3();
            let [r, g, b, a] = mesh.color;
            let start = i * self.object_stride as usize;
            objects[start..start + OBJECT_UNIFORM_SIZE as usize]
                .copy_from_slice(&f32_bytes(&[rel.x, rel.y, rel.z, 0.0, r, g, b, a]));
        }
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
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

            pass.set_pipeline(&self.terrain_pipeline);
            for (i, mesh) in self.meshes.iter().enumerate() {
                let Some(surface) = &mesh.surface else { continue };
                pass.set_bind_group(1, &self.object_bind_group, &[(i as u64 * self.object_stride) as u32]);
                pass.set_bind_group(2, &self.faces[surface.face].bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.index_count, 0, 0..1);
            }

            pass.set_pipeline(&self.line_pipeline);
            let show_wireframe = self.debug_flags & DEBUG_WIREFRAME != 0;
            for (i, mesh) in self.meshes.iter().enumerate() {
                if mesh.surface.is_some() && !show_wireframe {
                    continue;
                }
                pass.set_bind_group(1, &self.object_bind_group, &[(i as u64 * self.object_stride) as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.line_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.line_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        Ok(())
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

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
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

    let mut meshes = sphere_meshes(&device);
    meshes.push(tycho_marker(&device));

    let object_stride = OBJECT_UNIFORM_SIZE.max(device.limits().min_uniform_buffer_offset_alignment as u64);
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frame Uniforms"),
        size: FRAME_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Object Uniforms"),
        size: object_stride * meshes.len().max(1) as u64,
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
    let face_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Face Texture Layout"),
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
        bind_group_layouts: &[Some(&frame_layout), Some(&object_layout), Some(&face_layout)],
        immediate_size: 0,
    });

    let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Albedo Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let placeholder_albedo = [0x50u8, 0x50, 0x50, 0xff].repeat((TILE_SIZE * TILE_SIZE) as usize);
    let faces = Face::ALL
        .iter()
        .map(|face| {
            let albedo = create_tile_texture(&device, &format!("Albedo {}", face.key()), wgpu::TextureFormat::Rgba8UnormSrgb);
            let dem = create_tile_texture(&device, &format!("DEM {}", face.key()), wgpu::TextureFormat::R16Uint);
            write_tile(&queue, &albedo, &placeholder_albedo, 4);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Face Textures"),
                layout: &face_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&albedo.create_view(&Default::default())),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&albedo_sampler) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&dem.create_view(&Default::default())),
                    },
                ],
            });
            FaceTextures { albedo, dem, bind_group }
        })
        .collect();

    let terrain_pipeline = create_pipeline(
        &device,
        "Terrain Pipeline",
        &terrain_shader,
        &terrain_layout,
        surface_format,
        wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
        wgpu::CompareFunction::Greater,
    );
    let line_pipeline = create_pipeline(
        &device,
        "Wireframe Pipeline",
        &line_shader,
        &pipeline_layout,
        surface_format,
        wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
        wgpu::CompareFunction::Greater,
    );

    let (depth_texture, depth_view) = create_depth_target(&device, width, height);

    console::log_1(&"StationTycho WebGPU pipelines ready.".into());
    Ok(Renderer {
        surface,
        device,
        queue,
        config,
        terrain_pipeline,
        line_pipeline,
        frame_buffer,
        frame_bind_group,
        object_buffer,
        object_bind_group,
        object_stride,
        meshes,
        faces,
        debug_flags: 0,
        camera: OrbitCamera::new(TYCHO_LAT_DEG, TYCHO_LON_DEG, 3.0 * MOON_RADIUS),
        _depth_texture: depth_texture,
        depth_view,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    primitive: wgpu::PrimitiveState,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
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
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}