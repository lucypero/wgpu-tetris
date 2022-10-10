extern crate core;

mod font_renderer;

use crate::{game, Game};
use libs::bytemuck;
use libs::cgmath;
use libs::cgmath::{Matrix4, Vector2, Vector3, Vector4};
use libs::image;
use libs::pollster::block_on;
use libs::wgpu;
use libs::wgpu::Queue;
use libs::wgpu::SurfaceConfiguration;
use libs::wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, BindingResource, Device, ShaderStages,
};
use libs::winit;
use std::mem;

pub const WINDOW_INNER_WIDTH: u32 = 1000;
pub const WINDOW_INNER_HEIGHT: u32 = 900;
const MAX_CHARACTERS_ON_SCREEN: usize = 50000;
// Fixed number of block instances in the instance renderer
//  In the game, there will always be less than this.
const BLOCK_COUNT: usize = 1024;

const SAMPLE_STRING: &'static str = r#"test string"#;
// const SAMPLE_STRING: &'static str = r#"test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string test string "#;

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

struct BorderedRect {
    pos: Vector2<f32>,
    extents: Vector2<f32>,
    border_width: f32,
    border_color: Vector4<f32>,
    fill_color: Vector4<f32>,
}

struct BorderedRectsRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    rects: Vec<BorderedRect>,
    border_color_buffer: BindGroupSetThing<Vec<Vec4>>,
    fill_color_buffer: BindGroupSetThing<Vec<Vec4>>,
    aspect_ratio_buffer: BindGroupSetThing<Vec<f32>>,
    border_width_buffer: BindGroupSetThing<Vec<f32>>,
}

pub struct WgpuContext {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: SurfaceConfiguration,
    pub size: libs::winit::dpi::PhysicalSize<u32>,
}

struct BlockRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    model_uniform_set: BindGroupSetThing<Vec<Mat4>>,
    color_uniform_set: BindGroupSetThing<Vec<Vec4>>,
    block_texture_bg: wgpu::BindGroup,
}

pub struct Renderer {
    pub ctx: WgpuContext,
    camera: BindGroupSetThing<CameraUniform>,
    block_renderer: BlockRenderer,
    text_renderer: font_renderer::TextRenderer,
    bordered_rects_renderer: BorderedRectsRenderer,
}

fn init_block_renderer(
    device: &Device,
    queue: &Queue,
    camera_layout: &BindGroupLayout,
    surface_config: &SurfaceConfiguration,
) -> BlockRenderer {
    let num_indices = INDICES.len() as u32;

    // Model matrix buffer
    let model_uniform_set = new_storage_buffer::<Mat4>(device, BLOCK_COUNT, ShaderStages::VERTEX);

    // Block Color buffer
    let color_uniform_set = new_storage_buffer::<Vec4>(device, BLOCK_COUNT, ShaderStages::FRAGMENT);

    let block_texture_bg = {
        // making block texture

        // the texture
        let diffuse_bytes = include_bytes!("../../assets/block.png");
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

        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        })
    };

    // Creating pipeline stuff

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

    let render_pipeline = {
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    // THE ORDER HERE MATTERS !!!!
                    camera_layout,
                    &model_uniform_set.layout,
                    &color_uniform_set.layout,
                    &get_normal_texture_bind_group_layout(&device),
                ],
                push_constant_ranges: &[],
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/shader.wgsl"));
        create_pipeline(
            &device,
            &surface_config,
            &shader,
            &render_pipeline_layout,
            "main pipeline",
        )
    };

    BlockRenderer {
        render_pipeline,
        vertex_buffer,
        index_buffer,
        num_indices,
        model_uniform_set,
        color_uniform_set,
        block_texture_bg,
    }
}

fn init_wgpu_context(window: &libs::winit::window::Window) -> WgpuContext {
    let size = window.inner_size();

    // The instance is a handle to our GPU
    // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    //println!("{:?}",instance.generate_report());

    let surface = unsafe { instance.create_surface(window) };
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let mut limits = wgpu::Limits::default();
    limits.max_bind_groups = 8;

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits,
            label: None,
        },
        None,
    ))
    .unwrap();

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_supported_formats(&adapter)[0],
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &surface_config);

    WgpuContext {
        surface,
        device,
        queue,
        surface_config,
        size,
    }
}

/// draws semi transparent rect with cute pinki border
fn new_bordered_rect(pos: Vector2<f32>, extents: Vector2<f32>) -> BorderedRect {
    BorderedRect {
        pos,
        extents,
        border_width: 5.0,
        border_color: Vector4::new(1.0, 0.459, 0.918, 1.0),
        fill_color: Vector4::new(1.0, 1.0, 1.0, 0.2),
    }
}

/// Builds a storage buffer and associated bind group
/// Given a size and shader stages
/// size: number of T that the buffer can hold
/// stages: the shader stages where the buffer can be accessed from
fn new_storage_buffer<T>(
    device: &Device,
    size: usize,
    stages: ShaderStages,
) -> BindGroupSetThing<Vec<T>> {
    let zeroed_vec = vec![0_u8; std::mem::size_of::<T>() * size];

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &zeroed_vec,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let layout = build_storage_buffer_layout(&device, stages);

    let bind_group = build_bind_group(&device, &layout, buffer.as_entire_binding());

    BindGroupSetThing {
        the_data: Vec::<T>::with_capacity(size),
        buffer,
        layout,
        bind_group,
    }
}

fn init_bordered_rect_renderer(
    device: &Device,
    queue: &Queue,
    camera_layout: &BindGroupLayout,
    surface_config: &SurfaceConfiguration,
) -> BorderedRectsRenderer {
    // storage buffers for fragment shader
    let border_color_buffer =
        new_storage_buffer::<Vec4>(device, BLOCK_COUNT, ShaderStages::FRAGMENT);
    let fill_color_buffer = new_storage_buffer::<Vec4>(device, BLOCK_COUNT, ShaderStages::FRAGMENT);
    let aspect_ratio_buffer =
        new_storage_buffer::<f32>(device, BLOCK_COUNT, ShaderStages::FRAGMENT);
    let border_width_buffer =
        new_storage_buffer::<f32>(device, BLOCK_COUNT, ShaderStages::FRAGMENT);

    // @TODO(lucypero): do the vertices like u did with text
    // @TODO(lucypero): mayb an index buffer

    let zeroed_verts = vec![0_u8; 6 * std::mem::size_of::<Vertex>() * BLOCK_COUNT];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &zeroed_verts,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    // @TODO(lucypero): do some test bordered rects and write the vertices

    // making pipeline
    let pipeline = {
        let render_pipeline_layout =
            // @TODO(lucypero): the layouts are not done yet
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    // THE ORDER HERE MATTERS !!!!
                    camera_layout,
                    &border_color_buffer.layout,
                    &fill_color_buffer.layout,
                    &aspect_ratio_buffer.layout,
                    &border_width_buffer.layout,
                ],
                push_constant_ranges: &[],
            });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/bordered_block.wgsl"));

        create_pipeline(
            &device,
            &surface_config,
            &shader,
            &render_pipeline_layout,
            "main pipeline",
        )
    };

    let mut rects: Vec<BorderedRect> = Vec::with_capacity(20);
    rects.push(new_bordered_rect(
        Vector2::new(20.0, 200.0),
        Vector2::new(50.0, 100.0),
    ));
    rects.push(new_bordered_rect(
        Vector2::new(200.0, 200.0),
        Vector2::new(50.0, 150.0),
    ));
    rects.push(new_bordered_rect(
        Vector2::new(20.0, 600.0),
        Vector2::new(150.0, 50.0),
    ));
    rects.push(new_bordered_rect(
        Vector2::new(600.0, 600.0),
        Vector2::new(200.0, 100.0),
    ));

    BorderedRectsRenderer {
        pipeline,
        vertex_buffer,
        rects,
        border_color_buffer,
        fill_color_buffer,
        aspect_ratio_buffer,
        border_width_buffer,
    }
}

pub fn init_renderer(window: &libs::winit::window::Window, game: &Game) -> Renderer {
    // making blocks pipeline
    let ctx = init_wgpu_context(window);
    let device = &ctx.device;
    let queue = &ctx.queue;

    // Making Camera
    let camera = {
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

        BindGroupSetThing {
            the_data: camera_uniform,
            buffer: camera_buffer,
            layout: camera_bind_group_layout,
            bind_group: camera_bind_group,
        }
    };

    let block_renderer = init_block_renderer(device, queue, &camera.layout, &ctx.surface_config);

    let text_renderer = font_renderer::init_text_renderer(device, queue, &ctx.surface_config);
    text_renderer.update_vertices(queue);

    let bordered_rects_renderer =
        init_bordered_rect_renderer(device, queue, &camera.layout, &ctx.surface_config);

    Renderer {
        ctx,
        camera,
        block_renderer,
        text_renderer,
        bordered_rects_renderer,
    }
}

pub fn render(renderer: &mut Renderer, game: &Game) -> Result<(), wgpu::SurfaceError> {
    // bind groups for font rendering

    let queue = &renderer.ctx.queue;

    // Updating  all the buffers

    // Updating Camera
    {
        // write new matrix in the uniform
        let cam = &game.camera;

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
        renderer.camera.the_data.view_proj = new_mat.into();

        // write new data to buffer
        queue.write_buffer(
            &renderer.camera.buffer,
            0,
            bytemuck::cast_slice(&[renderer.camera.the_data]),
        );
    }

    // updating block buffers: transformation matrix and color buffers
    renderer.block_renderer.model_uniform_set.the_data.clear();
    renderer.block_renderer.color_uniform_set.the_data.clear();

    for (_, block) in game.blocks.iter() {
        let new_mat = Matrix4::from_translation(Vector3 {
            x: block.pos.x,
            y: block.pos.y,
            z: 0.0,
        });
        renderer
            .block_renderer
            .model_uniform_set
            .the_data
            .push(new_mat.into());
        renderer
            .block_renderer
            .color_uniform_set
            .the_data
            .push(block.color.into());
    }

    queue.write_buffer(
        &renderer.block_renderer.model_uniform_set.buffer,
        0,
        bytemuck::cast_slice(
            renderer
                .block_renderer
                .model_uniform_set
                .the_data
                .as_slice(),
        ),
    );

    queue.write_buffer(
        &renderer.block_renderer.color_uniform_set.buffer,
        0,
        bytemuck::cast_slice(
            renderer
                .block_renderer
                .color_uniform_set
                .the_data
                .as_slice(),
        ),
    );

    let border_renderer = &mut renderer.bordered_rects_renderer;

    // writing bordered rect vertex buffer
    {
        let mut vertices: Vec<Vertex> = Vec::with_capacity(6 * border_renderer.rects.len());


        for block in border_renderer.rects.iter() {

            let xpos = block.pos.x;
            let ypos = block.pos.y;
            let w = block.extents.x;
            let h = block.extents.y;

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
        }

        queue.write_buffer(
            &border_renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices.as_slice()),
        );
    }

    // writing to all bordered rect buffers
    border_renderer.border_color_buffer.the_data.clear();
    border_renderer.fill_color_buffer.the_data.clear();
    border_renderer.aspect_ratio_buffer.the_data.clear();
    border_renderer.border_width_buffer.the_data.clear();

    for rect in border_renderer.rects.iter() {
        border_renderer
            .border_color_buffer
            .the_data
            .push(rect.border_color.into());
        border_renderer
            .fill_color_buffer
            .the_data
            .push(rect.fill_color.into());
        border_renderer
            .border_width_buffer
            .the_data
            .push(rect.border_width / rect.extents.x);
        border_renderer
            .aspect_ratio_buffer
            .the_data
            .push(rect.extents.x / rect.extents.y);
    }

    queue.write_buffer(
        &border_renderer.border_color_buffer.buffer,
        0,
        bytemuck::cast_slice(border_renderer.border_color_buffer.the_data.as_slice()),
    );
    queue.write_buffer(
        &border_renderer.fill_color_buffer.buffer,
        0,
        bytemuck::cast_slice(border_renderer.fill_color_buffer.the_data.as_slice()),
    );
    queue.write_buffer(
        &border_renderer.aspect_ratio_buffer.buffer,
        0,
        bytemuck::cast_slice(border_renderer.aspect_ratio_buffer.the_data.as_slice()),
    );
    queue.write_buffer(
        &border_renderer.border_width_buffer.buffer,
        0,
        bytemuck::cast_slice(border_renderer.border_width_buffer.the_data.as_slice()),
    );

    // Render pass
    let output = renderer.ctx.surface.get_current_texture()?;
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = renderer
        .ctx
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

        {
            // Rendering tetris blocks
            render_pass.set_pipeline(&renderer.block_renderer.render_pipeline);
            render_pass.set_bind_group(0, &renderer.camera.bind_group, &[]);
            render_pass.set_bind_group(
                1,
                &renderer.block_renderer.model_uniform_set.bind_group,
                &[],
            );
            render_pass.set_bind_group(
                2,
                &renderer.block_renderer.color_uniform_set.bind_group,
                &[],
            );
            render_pass.set_bind_group(3, &renderer.block_renderer.block_texture_bg, &[]);
            render_pass.set_vertex_buffer(0, renderer.block_renderer.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                renderer.block_renderer.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            // @NOTE(lucypero): working with text for now, the tetris was distracting
            //   uncomment this to see tetris blocks
            render_pass.draw_indexed(
                0..renderer.block_renderer.num_indices,
                0,
                0..game.blocks.len() as _,
            );
        }

        // Rendering text
        renderer
            .text_renderer
            .render(&mut render_pass, &renderer.camera.bind_group);

        // Rendering bordered rectangles
        let border_renderer = &renderer.bordered_rects_renderer;
        render_pass.set_pipeline(&border_renderer.pipeline);
        render_pass.set_bind_group(0, &renderer.camera.bind_group, &[]);
        render_pass.set_bind_group(1, &border_renderer.border_color_buffer.bind_group, &[]);
        render_pass.set_bind_group(2, &border_renderer.fill_color_buffer.bind_group, &[]);
        render_pass.set_bind_group(3, &border_renderer.aspect_ratio_buffer.bind_group, &[]);
        render_pass.set_bind_group(4, &border_renderer.border_width_buffer.bind_group, &[]);
        render_pass.set_vertex_buffer(0, border_renderer.vertex_buffer.slice(..));

        // @TODO(lucypero): fuck how do i do this..
        // so i have all verts in a buffer so should b able to draw all that in 1 draw call i think
        // but then how do index all the buffers
        // damn i am not sure
        render_pass.draw(0..border_renderer.rects.len() as u32 * 6, 0..1);
    }

    // submit will accept anything that implements IntoIter
    queue.submit(std::iter::once(encoder.finish()));
    output.present();
    Ok(())
}

pub fn resize(renderer: &mut Renderer, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
        let ctx = &mut renderer.ctx;
        ctx.size = new_size;
        ctx.surface_config.width = new_size.width;
        ctx.surface_config.height = new_size.height;
        ctx.surface.configure(&ctx.device, &ctx.surface_config);
    }
}
