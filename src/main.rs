#![allow(dead_code, non_camel_case_types)]

mod game;
mod gui;
mod input;
mod renderer;

use game::*;
use crate::game::Game;
use crate::input::Input;
use std::time::Instant;

use libs::cgmath::Vector2;
use libs::wgpu;
use libs::winit;
use libs::winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub const WINDOW_INNER_WIDTH: u32 = 1000;
pub const WINDOW_INNER_HEIGHT: u32 = 900;

fn main() {
    let event_loop = EventLoop::new();

    let inner_size = winit::dpi::PhysicalSize {
        width: WINDOW_INNER_WIDTH,
        height: WINDOW_INNER_HEIGHT,
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
    let mut renderer = renderer::init_renderer(&window, &game);

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
                renderer::resize(&mut renderer, *physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer::resize(&mut renderer, **new_inner_size);
            }
            WindowEvent::CursorMoved {
                position, ..
            } => {
                input.process_mouse_position(position.x as f32, position.y as f32);
            },
            WindowEvent::MouseInput {
                state,
                button,
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                input.process_mouse_event(pressed, *button);
            }
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            game_update(&mut game, &input, now.elapsed());
            now = Instant::now();
            input.save_snapshot();
            match renderer::render(&mut renderer, &game) {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => {
                    let new_size = renderer.ctx.size;
                    renderer::resize(&mut renderer, new_size);
                }
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
