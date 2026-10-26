use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // Set up panic hook so Rust panics show up nicely in the browser developer console
    console_error_panic_hook::set_once();
    console::log_1(&"StationTycho engine initialized.".into());
    Ok(())
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Welcome to Station Tycho, {}.", name)
}

#[wasm_bindgen]
pub async fn ping_r2_asset(asset_name: &str) -> Result<JsValue, JsValue> {
    let window = web_sys::window().expect("no global `window` exists");
    let url = format!("/assets/{}", asset_name);
    console::log_1(&format!("Requesting lunar asset from R2 edge: {}", url).into());

    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&url)).await?;
    let resp: web_sys::Response = resp_value.dyn_into()?;

    console::log_1(&format!("R2 Response Status: {}", resp.status()).into());
    Ok(JsValue::from_f64(resp.status() as f64))
}

const TEST_SHADER: &str = r#"
struct Uniforms { time: f32, aspect: f32, _pad: vec2<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2(0.0, 0.6), vec2(-0.52, -0.3), vec2(0.52, -0.3));
    var c = array<vec3<f32>, 3>(vec3(1.0, 0.2, 0.3), vec3(0.2, 1.0, 0.4), vec3(0.3, 0.5, 1.0));
    let s = sin(u.time);
    let co = cos(u.time);
    let v = p[i];
    let r = vec2(v.x * co - v.y * s, v.x * s + v.y * co);
    var out: VsOut;
    out.pos = vec4(r.x / u.aspect, r.y, 0.0, 1.0);
    out.color = c[i] * (0.75 + 0.25 * sin(u.time * 3.0 + f32(i) * 2.094));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
"#;

#[wasm_bindgen]
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[wasm_bindgen]
impl Renderer {
    /// Renders one frame of the test animation. `time_ms` is the requestAnimationFrame timestamp.
    pub fn render(&mut self, time_ms: f64) -> Result<(), JsValue> {
        let t = (time_ms / 1000.0) as f32;
        let aspect = self.config.width as f32 / self.config.height as f32;
        let uniforms: Vec<u8> = [t, aspect, 0.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.queue.write_buffer(&self.uniform_buffer, 0, &uniforms);

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
            let pulse = 0.5 + 0.5 * (t as f64 * 0.8).sin();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Test Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.03 + 0.04 * pulse,
                            b: 0.08 + 0.10 * pulse,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

#[wasm_bindgen]
pub async fn init_render_engine(canvas_id: &str) -> Result<Renderer, JsValue> {
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

// 4. Request device and queue using the adapter's native reported limits
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("StationTycho Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to request device: {:?}", e)))?;

    // 5. Configure surface dimensions and format
    let width = canvas.width();
    let height = canvas.height();
    let caps = surface.get_capabilities(&adapter);
    let surface_format = caps.formats[0];

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: if width > 0 { width } else { 1 },
        height: if height > 0 { height } else { 1 },
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: Default::default(),
    };
    surface.configure(&device, &config);

    console::log_1(&"wgpu surface configured. Building test pipeline...".into());

    // 6. Build the test animation pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Test Shader"),
        source: wgpu::ShaderSource::Wgsl(TEST_SHADER.into()),
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniforms"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Uniform Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Uniform Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Test Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Test Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
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
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    console::log_1(&"StationTycho WebGPU test pipeline ready.".into());
    Ok(Renderer {
        surface,
        device,
        queue,
        config,
        pipeline,
        uniform_buffer,
        bind_group,
    })
}