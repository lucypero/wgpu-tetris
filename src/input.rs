// use
// use libs::winit;

use libs::winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode};

#[repr(u32)]
pub enum Keys {
    W,
    A,
    S,
    D,
    H,
    J,
    K,
    U,
    Space,
    NumpadAdd,
    NumpadSubtract,
    Right,
    Left,
    Up,
    Down,
    Mouse1,
    Mouse2,
    COUNT,
}

pub struct Input {
    keys_prev_frame: [bool; Keys::COUNT as usize],
    keys: [bool; Keys::COUNT as usize],
    pub mouse_x: f32,
    pub mouse_y: f32,
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys_prev_frame: [false; Keys::COUNT as usize],
            keys: [false; Keys::COUNT as usize],
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    pub fn process_mouse_position(&mut self, position_x: f32, position_y: f32) {
        self.mouse_x = position_x;
        self.mouse_y = position_y;
    }

    pub fn process_mouse_event(&mut self, pressed: bool, button: MouseButton) {
        match button {
            MouseButton::Left => self.keys[Keys::Mouse1 as usize] = pressed,
            MouseButton::Right => self.keys[Keys::Mouse2 as usize] = pressed,
            _ => {}
        }
    }

    pub fn process_key_event(&mut self, event: &KeyboardInput) {
        //Pressed Or Released
        let pressed = match event.state {
            ElementState::Pressed => true,
            ElementState::Released => false,
        };

        match event.virtual_keycode {
            Some(vk) => match vk {
                VirtualKeyCode::W => {
                    self.keys[Keys::W as usize] = pressed;
                }
                VirtualKeyCode::A => {
                    self.keys[Keys::A as usize] = pressed;
                }
                VirtualKeyCode::S => {
                    self.keys[Keys::S as usize] = pressed;
                }
                VirtualKeyCode::D => {
                    self.keys[Keys::D as usize] = pressed;
                }
                VirtualKeyCode::NumpadAdd => {
                    self.keys[Keys::NumpadAdd as usize] = pressed;
                }
                VirtualKeyCode::NumpadSubtract => {
                    self.keys[Keys::NumpadSubtract as usize] = pressed;
                }
                VirtualKeyCode::Up => {
                    self.keys[Keys::Up as usize] = pressed;
                }
                VirtualKeyCode::Down => {
                    self.keys[Keys::Down as usize] = pressed;
                }
                VirtualKeyCode::Left => {
                    self.keys[Keys::Left as usize] = pressed;
                }
                VirtualKeyCode::Right => {
                    self.keys[Keys::Right as usize] = pressed;
                }
                VirtualKeyCode::H => {
                    self.keys[Keys::H as usize] = pressed;
                }
                VirtualKeyCode::J => {
                    self.keys[Keys::J as usize] = pressed;
                }
                VirtualKeyCode::K => {
                    self.keys[Keys::K as usize] = pressed;
                }
                VirtualKeyCode::U => {
                    self.keys[Keys::U as usize] = pressed;
                }
                VirtualKeyCode::Space => {
                    self.keys[Keys::Space as usize] = pressed;
                }
                kc => println!("Key {:?} not implemented! add it in input.rs!", kc),
            },
            _ => {}
        }
    }

    pub fn save_snapshot(&mut self) {
        self.keys_prev_frame = self.keys;
    }

    //returns true if key is being held down
    pub fn get_key(&self, key: Keys) -> bool {
        self.keys[key as usize]
    }

    //returns true on the first frame that the key is down
    pub fn get_key_down(&self, key: Keys) -> bool {
        let ind = key as usize;
        self.keys[ind] && !self.keys_prev_frame[ind]
    }

    //returns true on the first frame that the key is up
    pub fn get_key_up(&self, key: Keys) -> bool {
        let ind = key as usize;
        !self.keys[ind] && self.keys_prev_frame[ind]
    }
}
