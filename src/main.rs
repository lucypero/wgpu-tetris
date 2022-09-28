#![allow(dead_code)]

mod game;
mod input;
mod renderer;

// self.do_block_controls($b) ==>> do_block_controls(&mut self, $b)

// self.do_block_controls(&input);

// pub struct Game {
//     pub blocks: Arena<Block>,
//     active_block_set: Option<BlockSet>,
//     hold_block_preview: Option<StaticBlockSet>,
//     pub grid: Grid,
//     tick_timer: Duration,
//     pub camera: Camera,
//     hold_enabled: bool,
//     next_blocks: VecDeque<StaticBlockSet>,
//     next_block_types: Vec<BlockSetType>,
//     next_block_index: usize,
//     rng: rand::rngs::ThreadRng,
// }


use crate::game::Game;
use crate::input::Input;
use crate::renderer::Renderer;
use std::time::Instant;

use libs::cgmath::Vector2;
use libs::wgpu;
use libs::winit;
use libs::winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// foo($a, $b) ==>> foo($b, $a)

fn main() {
    libs::pollster::block_on(run());
}

pub async fn run() {
    let event_loop = EventLoop::new();

    let inner_size = winit::dpi::PhysicalSize {
        width: renderer::WINDOW_INNER_WIDTH,
        height: renderer::WINDOW_INNER_HEIGHT,
    };

    let window = WindowBuilder::new()
        .with_title("wgpu-tetris")
        .with_inner_size(inner_size)
        .with_position(winit::dpi::PhysicalPosition { x: 300., y: 10. })
        // .with_fullscreen(Some(Borderless(None)))
        .build(&event_loop)
        .unwrap();

    let mut input = Input::new();
    let mut game = Game::new(Vector2::new(inner_size.width, inner_size.height));
    let mut renderer = Renderer::new(&window, &game).await;

    let mut now = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::KeyboardInput {
                input: kb_input, ..
            } => {
                input.process_key_event(kb_input);
            }
            WindowEvent::Resized(physical_size) => {
                renderer.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer.resize(**new_inner_size);
            }
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            game.update(&input, now.elapsed());
            now = Instant::now();
            input.save_snapshot();
            match renderer.render(&game) {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
        }
        _ => {}
    })
}
