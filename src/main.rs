use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder, Window},
};
use winit::window::Fullscreen::Borderless;

use wgpu_tetris;

fn main() {
    pollster::block_on(run());
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();

    let inner_size = winit::dpi::PhysicalSize{
        width: wgpu_tetris::WINDOW_INNER_WIDTH,
        height: wgpu_tetris::WINDOW_INNER_HEIGHT,
    };

    let window = WindowBuilder::new()
        .with_title("wgpu-tetris")
        .with_inner_size(inner_size)
        // .with_fullscreen(Some(Borderless(None)))
        .build(&event_loop).unwrap();

    let mut state = wgpu_tetris::State::new(&window).await;

    let mut frames: u64 = 0;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => if !state.input(event) {
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                _ => {}
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
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
