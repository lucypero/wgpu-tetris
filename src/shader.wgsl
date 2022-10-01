struct ModelUniform {
    model: array<mat4x4<f32>>,
};
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;
@group(1) @binding(0)
var<storage, read> model_mat: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @builtin(instance_index) inst_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) inst_index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * model_mat.model[model.inst_index] * vec4<f32>(model.position, 1.0);

    var outUV = vec2<f32>(f32((model.inst_index << u32(1)) & u32(2)), f32(model.inst_index & u32(2)));
    out.tex_coords = model.tex_coords;
    out.inst_index = model.inst_index;
    return out;
}

struct ColorUniform {
    color: array<vec4<f32>>,
};
@group(2) @binding(0)
var<storage, read> color_buf: ColorUniform;

@group(3) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(3) @binding(1)
var s_diffuse: sampler;

//testing font rendering (this is temporary)
@group(4) @binding(0)
var t_character: texture_2d<f32>;
@group(4) @binding(1)
var s_character: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // rendering the block (this is the real thing)

    /*

    var t = textureSample(t_diffuse, s_diffuse, in.tex_coords).x;
    var the_tint = color_buf.color[in.inst_index].xyz;

    var res = mix(the_tint, vec3(t), abs(t - 0.5) * 2.0);
    return vec4(res, color_buf.color[in.inst_index].w);

    */

    // rendering font (testing)
    var t = textureSample(t_character, s_character, in.tex_coords).r; // testing font
    var the_tint = color_buf.color[in.inst_index].xyz;
    return vec4(the_tint, t);

}
