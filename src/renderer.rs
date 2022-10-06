extern crate core;

use crate::{game, Game};
use libs::bytemuck;
use libs::cgmath;
use libs::cgmath::{Matrix4, Vector2, Vector3};
use libs::freetype_sys as ft;
use libs::image;
use libs::wgpu;
use libs::wgpu::util::DeviceExt;
use libs::wgpu::RenderPass;
use libs::wgpu::Sampler;
use libs::wgpu::{
    BindGroup, BindGroupLayout, BindingResource, Device, Queue, ShaderStages, TextureView,
};
use libs::winit;
use std::collections::HashMap;
use std::mem;

use crate::game::Camera;

pub const WINDOW_INNER_WIDTH: u32 = 1000;
pub const WINDOW_INNER_HEIGHT: u32 = 900;
const MAX_CHARACTERS_ON_SCREEN: usize = 5000;
// Fixed number of block instances in the instance renderer
//  In the game, there will always be less than this.
const BLOCK_COUNT: usize = 1024;

// const SAMPLE_STRING: &'static str = r#"test string"#;
const SAMPLE_STRING: &'static str = r#"test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string "#;

#[derive(Debug)]
struct Character {
    bind_group: BindGroup,
    size: Vector2<u32>,
    bearing: Vector2<i32>,
    advance: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vec4([f32; 4]);

unsafe impl bytemuck::Zeroable for Vec4 {}
unsafe impl bytemuck::Pod for Vec4 {}

impl From<[f32; 4]> for Vec4 {
    fn from(the_vec: [f32; 4]) -> Self {
        Vec4(the_vec)
    }
}

impl From<cgmath::Vector4<f32>> for Vec4 {
    fn from(the_vec: cgmath::Vector4<f32>) -> Self {
        Vec4(the_vec.into())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Mat4([[f32; 4]; 4]);

unsafe impl bytemuck::Zeroable for Mat4 {}
unsafe impl bytemuck::Pod for Mat4 {}

impl From<[[f32; 4]; 4]> for Mat4 {
    fn from(the_mat: [[f32; 4]; 4]) -> Self {
        Mat4(the_mat)
    }
}

impl From<Matrix4<f32>> for Mat4 {
    fn from(the_mat: Matrix4<f32>) -> Self {
        Mat4(the_mat.into())
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// Renders all ASCII characters of `font_name` font into textures into a hashmap with a size of `size`
unsafe fn render_font(
    device: &Device,
    queue: &Queue,
    sampler: &Sampler,
    font_name: String,
    font_size: u32,
) -> HashMap<char, Character> {
    let mut ft: ft::FT_Library = std::mem::zeroed();
    let res = ft::FT_Init_FreeType(&mut ft);
    if res != 0 {
        panic!("freetype could not init");
    }

    let mut face: ft::FT_Face = std::mem::zeroed();

    // add null terminator to font name to make it a cstr
    let mut font_name = font_name.clone();
    font_name.push('\0');

    let res = ft::FT_New_Face(ft, font_name.as_ptr() as *const i8, 0, &mut face);
    if res != 0 {
        panic!("could not load font");
    }

    ft::FT_Set_Pixel_Sizes(face, 0, font_size);

    let char_count = (0x20..0x7Fu8).len();

    // create the sampler
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // render all ascii
    // 128 code points (0x20..0x7F).
    let mut char_map = HashMap::with_capacity(char_count);
    for c in 0x20u8..0x7Fu8 {
        let c = c as char;
        if c == ' ' {
            continue;
        }

        let res = ft::FT_Load_Char(face, c as u32, ft::FT_LOAD_RENDER);
        if res != 0 {
            panic!("could not load {}", c);
        }

        let g = &(*(*face).glyph);
        let char_w = g.bitmap.width as u32;
        let char_h = g.bitmap.rows as u32;
        let bearing = Vector2::new(g.bitmap_left, g.bitmap_top);
        let advance = g.advance.x as u32;

        let char_buffer =
            std::slice::from_raw_parts_mut(g.bitmap.buffer, (char_w * char_h) as usize);

        // creating and rendering texture
        let texture_size = wgpu::Extent3d {
            width: char_w,
            height: char_h,
            depth_or_array_layers: 1,
        };
        let character_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: None,
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &character_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &char_buffer,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(char_w),
                rows_per_image: std::num::NonZeroU32::new(char_h),
            },
            texture_size,
        );

        // create the view
        let character_texture_view =
            character_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let character_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &get_normal_texture_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&character_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        char_map.insert(
            c,
            Character {
                bind_group: character_texture_bind_group,
                size: Vector2::new(char_w, char_h),
                bearing,
                advance,
            },
        );
    }

    ft::FT_Done_Face(face);
    ft::FT_Done_FreeType(ft);

    char_map
}

fn init_text_renderer(
    device: &Device,
    queue: &Queue,
    surface_config: &wgpu::SurfaceConfiguration,
) -> TextRenderer {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let char_map = unsafe { render_font(&device, &queue, &sampler, "fonts/Roboto.ttf".into(), 22) };

    // make the text vertex buffer
    let zeroed_verts = vec![0_u8; 6 * std::mem::size_of::<Vertex>() * MAX_CHARACTERS_ON_SCREEN];

    //vertex buffer for text
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glyph vertex buffer"),
        contents: &zeroed_verts,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    // text pipeline
    let text_shader = device.create_shader_module(wgpu::include_wgsl!("text.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            &get_camera_texture_bind_group_layout(&device),
            &get_normal_texture_bind_group_layout(&device),
        ],
        push_constant_ranges: &[],
    });

    let pipeline = create_pipeline(
        &device,
        &surface_config,
        &text_shader,
        &pipeline_layout,
        "text pipeline",
    );

    // writing all vertices

    //updating text vertex buffer

    // making all the vertices

    const STRING_START_X: f32 = 20.0;
    const STRING_START_Y: f32 = 20.0;

    let mut string_x = STRING_START_X; // the start x pos
    let mut string_y = STRING_START_Y; // the start y pos


    let characters_to_draw_len = SAMPLE_STRING.chars().filter(|c| *c != ' ').count();
    let mut vertices: Vec<Vertex> = Vec::with_capacity(6 * characters_to_draw_len);

    for c in SAMPLE_STRING.chars() {
        // inserting line breaks
        if c == ' ' {
            string_x += 10.0;
            continue;
        }

        // soft wrap
        if string_x > WINDOW_INNER_HEIGHT as f32 {
            string_x = STRING_START_X;
            //adding line_height
            string_y += 22.0;
        }

        let char_info = char_map.get(&c).unwrap();

        let xpos: f32 = string_x + char_info.bearing.x as f32;
        let ypos: f32 = string_y + char_info.size.y as f32 - char_info.bearing.y as f32;
        let w: f32 = char_info.size.x as f32;
        let h: f32 = char_info.size.y as f32;

        vertices.push(Vertex {
            position: [xpos, ypos - h, 0.0],
            tex_coords: [0.0, 0.0],
        });
        vertices.push(Vertex {
            position: [xpos, ypos, 0.0],
            tex_coords: [0.0, 1.0],
        });
        vertices.push(Vertex {
            position: [xpos + w, ypos, 0.0],
            tex_coords: [1.0, 1.0],
        });
        vertices.push(Vertex {
            position: [xpos, ypos - h, 0.0],
            tex_coords: [0.0, 0.0],
        });
        vertices.push(Vertex {
            position: [xpos + w, ypos, 0.0],
            tex_coords: [1.0, 1.0],
        });
        vertices.push(Vertex {
            position: [xpos + w, ypos - h, 0.0],
            tex_coords: [1.0, 0.0],
        });
        string_x += (char_info.advance >> 6) as f32;
    }

    queue.write_buffer(
        &vertex_buffer,
        0,
        bytemuck::cast_slice(vertices.as_slice()),
    );

    TextRenderer {
        pipeline,
        vertex_buffer,
        sampler,
        char_map,
    }
}

// @TODO(lucypero): text renderer needs to store the bind groups probs. or something.
fn render_text<'a>(
    text_renderer: &'a TextRenderer,
    render_pass: &mut RenderPass<'a>,
    camera_bg: &'a BindGroup,
) {
    render_pass.set_pipeline(&text_renderer.pipeline);
    render_pass.set_bind_group(0, camera_bg, &[]);
    render_pass.set_vertex_buffer(0, text_renderer.vertex_buffer.slice(..));

    for (i, c) in SAMPLE_STRING.chars().filter(|c| *c != ' ').enumerate() {
        let char_info = text_renderer.char_map.get(&c).unwrap();
        render_pass.set_bind_group(1, &char_info.bind_group, &[]);
        render_pass.draw(i as u32 * 6..i as u32 * 6 + 6, 0..1);
    }
}

fn create_pipeline(
    device: &Device,
    config: &wgpu::SurfaceConfiguration,
    shader: &wgpu::ShaderModule,
    render_pipeline_layout: &wgpu::PipelineLayout,
    name: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[Vertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

// helpers
fn build_storage_buffer_layout(device: &Device, stages: ShaderStages) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: None,
    })
}

fn get_normal_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: None,
    })
}

fn get_camera_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("camera_bind_group_layout"),
    })
}

fn build_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    resource: BindingResource,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource,
        }],
        label: None,
    })
}

// Camera andy
fn update_cam_buffer(
    cam: &Camera,
    cam_bind_group: &mut BindGroupSetThing<CameraUniform>,
    queue: &wgpu::Queue,
) {
    // write new matrix in the uniform

    // transformation
    let t_mat = Matrix4::from_translation(Vector3::new(cam.position.x, cam.position.y, 0.0));
    // idk if this should be here. if it works delete this comment:
    //cam.bind_group.the_data.view_proj = Mat4::from(t_mat);

    // zoom
    let view_proj = {
        let new_vec = cam.initial_size * cam.zoom_amount;
        let proj = cgmath::ortho(0.0, new_vec.x, new_vec.y, 0.0, -1.0, 1.0);
        OPENGL_TO_WGPU_MATRIX * proj
    };

    let new_mat: [[f32; 4]; 4] = (view_proj * t_mat).into();
    cam_bind_group.the_data.view_proj = new_mat.into();

    // write new data to buffer
    queue.write_buffer(
        &cam_bind_group.buffer,
        0,
        bytemuck::cast_slice(&[cam_bind_group.the_data]),
    );
}

// we are gonna use an ortho camera
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CameraUniform {
    view_proj: Mat4,
}

unsafe impl bytemuck::Zeroable for CameraUniform {}
unsafe impl bytemuck::Pod for CameraUniform {}

// TODO: update camera uniform on resize
impl CameraUniform {
    fn new(size: Vector2<f32>) -> Self {
        let view_proj: [[f32; 4]; 4] = {
            let proj = cgmath::ortho(0.0, size.x, size.y, 0.0, -1.0, 1.0);
            (OPENGL_TO_WGPU_MATRIX * proj).into()
        };

        // tried a 3d camera. it isn't working out.

        /*

        // position the camera one unit up and 2 units back
        // +z is out of the screen
        let eye: Point3<f32> = (0.0, 0.0, -3.0).into();
        // have it look at the origin
        let target: Point3<f32> = (0.0, 0.0, 0.0).into();
        // which way is "up"
        let up: Vector3<f32> = Vector3::unit_y();
        let aspect = size.x as f32 / size.y as f32;
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;

        // 1.
        let view = Matrix4::look_at_rh(eye, target, up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(fovy), aspect, znear, zfar);

        let res = OPENGL_TO_WGPU_MATRIX * proj * view;

         */

        Self {
            view_proj: view_proj.into(),
        }
    }
}

struct ModelUniform {
    model: Vec<Mat4>,
}

impl ModelUniform {
    fn new() -> Self {
        let mut model_vec: Vec<Mat4> = Vec::with_capacity(BLOCK_COUNT);

        for _ in 0..BLOCK_COUNT {
            let model: [[f32; 4]; 4] = Matrix4::from_translation(Vector3 {
                x: -game::BLOCK_SIZE,
                y: -game::BLOCK_SIZE,
                z: 0.0,
            })
            .into();
            model_vec.push(model.into());
        }

        Self { model: model_vec }
    }
}

struct ColorUniform {
    color: Vec<Vec4>,
}

impl ColorUniform {
    fn new() -> Self {
        let mut color_vec = Vec::with_capacity(BLOCK_COUNT);

        for _ in 0..BLOCK_COUNT {
            let color: Vec4 = cgmath::Vector4 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                w: 1.0,
            }
            .into();
            color_vec.push(color);
        }

        Self { color: color_vec }
    }
}

struct BindGroupSetThing<T> {
    the_data: T,
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    /*
    (0)-----(1)
     |       |
     |       |
     |       |
    (2)-----(3)
     */
    Vertex {
        position: [0.0, 0.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [game::BLOCK_SIZE, 0.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
    Vertex {
        position: [0.0, game::BLOCK_SIZE, 0.0],
        tex_coords: [0.0, 1.0],
    },
    Vertex {
        position: [game::BLOCK_SIZE, game::BLOCK_SIZE, 0.0],
        tex_coords: [1.0, 1.0],
    },
];

const INDICES: &[u16] = &[1, 0, 2, 2, 3, 1];

struct TextRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    // @TODO(lucypero): do index buffer for text vertices later
    char_map: HashMap<char, Character>,
}

pub struct Renderer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: libs::winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera_uniform_set: BindGroupSetThing<CameraUniform>,
    model_uniform_set: BindGroupSetThing<ModelUniform>,
    color_uniform_set: BindGroupSetThing<ColorUniform>,
    diffuse_bind_group: wgpu::BindGroup,
    text_renderer: TextRenderer,
}

impl Renderer {
    pub async fn new(window: &libs::winit::window::Window, game: &Game) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        //println!("{:?}",instance.generate_report());

        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits,
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &surface_config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        // -- bind descriptors --

        // we need an ortho camera
        let camera_uniform = CameraUniform::new(game.camera.initial_size);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = get_camera_texture_bind_group_layout(&device);

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_uniform_set = BindGroupSetThing {
            the_data: camera_uniform,
            buffer: camera_buffer,
            layout: camera_bind_group_layout,
            bind_group: camera_bind_group,
        };

        // we need a model matrix and color

        // model matrix : we need to create the buffer and the bind group
        // this is the instance data for the tetris blocks

        let model_uniform = ModelUniform::new();

        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model matrix buffer"),
            contents: bytemuck::cast_slice(model_uniform.model.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let model_bind_group_layout = build_storage_buffer_layout(&device, ShaderStages::VERTEX);

        let model_bind_group = build_bind_group(
            &device,
            &model_bind_group_layout,
            model_buffer.as_entire_binding(),
        );

        let model_uniform_set = BindGroupSetThing {
            the_data: model_uniform,
            buffer: model_buffer,
            layout: model_bind_group_layout,
            bind_group: model_bind_group,
        };

        // color uniform
        // cute colors for all the bloccs

        let color_uniform = ColorUniform::new();

        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color buffer"),
            contents: bytemuck::cast_slice(color_uniform.color.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let color_layout = build_storage_buffer_layout(&device, ShaderStages::FRAGMENT);

        let color_bind_group =
            build_bind_group(&device, &color_layout, color_buffer.as_entire_binding());

        let color_uniform_set = BindGroupSetThing {
            the_data: color_uniform,
            buffer: color_buffer,
            layout: color_layout,
            bind_group: color_bind_group,
        };

        // the texture

        let diffuse_bytes = include_bytes!("block.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &diffuse_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        let diffuse_texture_view =
            diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &get_normal_texture_bind_group_layout(&device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // -- bind group end --

        // creating pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    // THE ORDER HERE MATTERS !!!!
                    &camera_uniform_set.layout,
                    &model_uniform_set.layout,
                    &color_uniform_set.layout,
                    &get_normal_texture_bind_group_layout(&device),
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = create_pipeline(
            &device,
            &surface_config,
            &shader,
            &render_pipeline_layout,
            "main pipeline",
        );

        let text_renderer = init_text_renderer(&device, &queue, &surface_config);

        Self {
            surface,
            device,
            queue,
            config: surface_config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            model_uniform_set,
            camera_uniform_set,
            color_uniform_set,
            diffuse_bind_group,
            //font stuff start
            text_renderer,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(&mut self, game: &Game) -> Result<(), wgpu::SurfaceError> {
        // bind groups for font rendering

        // update all the buffers
        {
            //updating camera
            update_cam_buffer(&game.camera, &mut self.camera_uniform_set, &self.queue);

            // update block model matrices
            for (the_mat, (_, block)) in self
                .model_uniform_set
                .the_data
                .model
                .iter_mut()
                .take(game.blocks.len())
                .zip(game.blocks.iter())
            {
                let new_mat = Matrix4::from_translation(Vector3 {
                    x: block.pos.x,
                    y: block.pos.y,
                    z: 0.0,
                });
                the_mat.0 = new_mat.into();
            }

            self.queue.write_buffer(
                &self.model_uniform_set.buffer,
                0,
                bytemuck::cast_slice(self.model_uniform_set.the_data.model.as_slice()),
            );

            //updating block colors
            for (color_mat, (_, block)) in self
                .color_uniform_set
                .the_data
                .color
                .iter_mut()
                .take(game.blocks.len())
                .zip(game.blocks.iter())
            {
                color_mat.0 = block.color.into();
            }

            self.queue.write_buffer(
                &self.color_uniform_set.buffer,
                0,
                bytemuck::cast_slice(self.color_uniform_set.the_data.color.as_slice()),
            );
        }

        // do the render
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_uniform_set.bind_group, &[]);
            render_pass.set_bind_group(1, &self.model_uniform_set.bind_group, &[]);
            render_pass.set_bind_group(2, &self.color_uniform_set.bind_group, &[]);
            render_pass.set_bind_group(3, &self.diffuse_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..self.num_indices, 0, 0..game.blocks.len() as _);

            // text rendering
            render_text(
                &self.text_renderer,
                &mut render_pass,
                &self.camera_uniform_set.bind_group,
            );
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
