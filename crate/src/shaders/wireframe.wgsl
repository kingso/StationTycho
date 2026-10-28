struct Frame { view_proj: mat4x4<f32> };
// origin_rel is the object's f64 origin minus the camera position, rounded to f32 on the CPU.
struct Object { origin_rel: vec4<f32>, color: vec4<f32> };

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;

struct VsIn {
    @location(0) offset: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let rel = object.origin_rel.xyz + v.offset;
    var out: VsOut;
    out.pos = frame.view_proj * vec4(rel, 1.0);
    out.rel = rel;
    out.normal = v.normal;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Hide lines below the camera's horizon (the far side of the sphere).
    if (dot(in.normal, -in.rel) < 0.0) {
        discard;
    }
    return object.color;
}
