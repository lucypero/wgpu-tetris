struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

@group(1) @binding(0)
var t_character: texture_2d<f32>;
@group(1) @binding(1)
var s_character: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var t = textureSample(t_character, s_character, in.tex_coords).r; // testing font
    // var the_tint = vec3(0.94,0.45,0.82); // pink
    var the_tint = vec3(1.0, 1.0, 1.0);
    return vec4(the_tint, t);
}
