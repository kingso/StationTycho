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
    // Reversed-Z: depth 0 is the far plane, so the triangle sits at 0.5.
    out.pos = vec4(r.x / u.aspect, r.y, 0.5, 1.0);
    out.color = c[i] * (0.75 + 0.25 * sin(u.time * 3.0 + f32(i) * 2.094));
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
