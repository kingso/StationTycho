struct Frame { view_proj: mat4x4<f32>, debug_flags: u32 };
// origin_rel is the object's f64 origin minus the camera position, rounded to f32 on the CPU.
struct Object { origin_rel: vec4<f32>, color: vec4<f32> };

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;
@group(2) @binding(2) var dem_tex: texture_2d<u32>;

const DEBUG_HEIGHT: u32 = 2u;
// Quantisation from tools/cube.mjs (also in manifest.json) until the engine reads the manifest.
const DEM_SCALE: f32 = 0.5;
const DEM_OFFSET: f32 = -9200.0;

struct VsIn {
    @location(0) offset: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = frame.view_proj * vec4(object.origin_rel.xyz + v.offset, 1.0);
    out.uv = v.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if ((frame.debug_flags & DEBUG_HEIGHT) != 0u) {
        let dims = textureDimensions(dem_tex);
        let texel = min(vec2<u32>(in.uv * vec2<f32>(dims)), dims - 1u);
        let height = f32(textureLoad(dem_tex, texel, 0).r) * DEM_SCALE + DEM_OFFSET;
        return vec4(vec3(clamp((height + 5000.0) / 10000.0, 0.0, 1.0)), 1.0);
    }
    return textureSample(albedo_tex, albedo_sampler, in.uv);
}
