use crate::camera::OrbitCamera;
use crate::geometry::{enu_frame, face_wireframe_chunks, Face, MOON_RADIUS};
use glam::DVec3;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;
use wgpu::util::DeviceExt;

const WIREFRAME_SHADER: &str = include_str!("shaders/wireframe.wgsl");

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const FRAME_UNIFORM_SIZE: u64 = 64;
const OBJECT_UNIFORM_SIZE: u64 = 32;
const VERTEX_STRIDE: u64 = 24;

const TYCHO_LAT_DEG: f64 = -43.31;
const TYCHO_LON_DEG: f64 = -11.36;

// 4×4 chunks of 8×8 cells: a 32-line grid per face, chunked so f32 offsets stay small.
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

fn sphere_wireframe(device: &wgpu::Device) -> Vec<LineMesh> {
    Face::ALL
        .iter()
        .flat_map(|&face| {
            face_wireframe_chunks(face, SPHERE_CHUNKS_PER_FACE, SPHERE_SEGMENTS_PER_CHUNK)
                .into_iter()
                .map(move |chunk| (face, chunk))
        })
        .map(|(face, chunk)| {
            LineMesh::new(device, "Sphere Chunk", chunk.origin, &chunk.vertices, &chunk.indices, face_color(face))
        })
        .collect()
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn u32_bytes(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|i| i.to_le_bytes()).collect()
}

/// Line-list mesh whose vertices are f32 offsets from an f64 `origin` in the body-fixed frame.
struct LineMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    origin: DVec3,
    color: [f32; 4],
}

impl LineMesh {
    /// `vertices` holds interleaved `[offset.xyz, normal.xyz]`; `indices` are line-list pairs.
    fn new(device: &wgpu::Device, label: &str, origin: DVec3, vertices: &[f32], indices: &[u32], color: [f32; 4]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &f32_bytes(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &u32_bytes(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self { vertex_buffer, index_buffer, index_count: indices.len() as u32, origin, color }
    }
}

/// 20 m wireframe cube standing on the reference sphere at Tycho's centre.
fn tycho_marker(device: &wgpu::Device) -> LineMesh {
    const HALF: f64 = 10.0;
    let (east, north, up) = enu_frame(TYCHO_LAT_DEG.to_radians(), TYCHO_LON_DEG.to_radians());
    let mut vertices = Vec::with_capacity(8 * 6);
    for corner in 0..8u32 {
        let x = if corner & 1 == 0 { -HALF } else { HALF };
        let y = if corner & 2 == 0 { -HALF } else { HALF };
        let z = if corner & 4 == 0 { 0.0 } else { 2.0 * HALF };
        let offset = (east * x + north * y + up * z).as_vec3();
        let normal = up.as_vec3();
        vertices.extend_from_slice(&[offset.x, offset.y, offset.z, normal.x, normal.y, normal.z]);
    }
    let mut indices = Vec::with_capacity(24);
    for corner in 0..8u32 {
        for bit in [1, 2, 4] {
            if corner & bit == 0 {
                indices.extend_from_slice(&[corner, corner | bit]);
            }
        }
    }
    LineMesh::new(device, "Tycho Marker", up * MOON_RADIUS, &vertices, &indices, [1.0, 0.45, 0.2, 1.0])
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
    pipeline: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    object_stride: u64,
    meshes: Vec<LineMesh>,
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

    /// Renders one frame. `_time_ms` is the requestAnimationFrame timestamp.
    pub fn render(&mut self, _time_ms: f64) -> Result<(), JsValue> {
        let aspect = self.config.width as f64 / self.config.height as f64;
        let view_proj = self.camera.view_proj(aspect);
        self.queue.write_buffer(&self.frame_buffer, 0, &f32_bytes(&view_proj.to_cols_array()));

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
                label: Some("Wireframe Pass"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            for (i, mesh) in self.meshes.iter().enumerate() {
                pass.set_bind_group(1, &self.object_bind_group, &[(i as u64 * self.object_stride) as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
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

    console::log_1(&"wgpu surface configured. Building wireframe pipeline...".into());

    // 6. Build the wireframe pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Wireframe Shader"),
        source: wgpu::ShaderSource::Wgsl(WIREFRAME_SHADER.into()),
    });

    let mut meshes = sphere_wireframe(&device);
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

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Wireframe Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let (depth_texture, depth_view) = create_depth_target(&device, width, height);

    console::log_1(&"StationTycho WebGPU wireframe pipeline ready.".into());
    Ok(Renderer {
        surface,
        device,
        queue,
        config,
        pipeline,
        frame_buffer,
        frame_bind_group,
        object_buffer,
        object_bind_group,
        object_stride,
        meshes,
        camera: OrbitCamera::new(TYCHO_LAT_DEG, TYCHO_LON_DEG, 3.0 * MOON_RADIUS),
        _depth_texture: depth_texture,
        depth_view,
    })
}