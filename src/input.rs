use strum::EnumCount;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

#[derive(EnumCount)]
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
}

fn key_to_index(key: Keys) -> usize {
    match key {
        Keys::W => 0,
        Keys::A => 1,
        Keys::S => 2,
        Keys::D => 3,
        Keys::H => 4,
        Keys::J => 5,
        Keys::K => 6,
        Keys::U => 7,
        Keys::Space => 8,
        Keys::NumpadAdd => 9,
        Keys::NumpadSubtract => 10,
        Keys::Right => 11,
        Keys::Left => 12,
        Keys::Up => 13,
        Keys::Down => 14,
    }
}

pub struct Input {
    keys_prev_frame: [bool; Keys::COUNT],
    keys: [bool; Keys::COUNT],
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys_prev_frame: [false; Keys::COUNT],
            keys: [false; Keys::COUNT],
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
                    self.keys[key_to_index(Keys::W)] = pressed;
                }
                VirtualKeyCode::A => {
                    self.keys[key_to_index(Keys::A)] = pressed;
                }
                VirtualKeyCode::S => {
                    self.keys[key_to_index(Keys::S)] = pressed;
                }
                VirtualKeyCode::D => {
                    self.keys[key_to_index(Keys::D)] = pressed;
                }
                VirtualKeyCode::NumpadAdd => {
                    self.keys[key_to_index(Keys::NumpadAdd)] = pressed;
                }
                VirtualKeyCode::NumpadSubtract => {
                    self.keys[key_to_index(Keys::NumpadSubtract)] = pressed;
                }
                VirtualKeyCode::Up => {
                    self.keys[key_to_index(Keys::Up)] = pressed;
                }
                VirtualKeyCode::Down => {
                    self.keys[key_to_index(Keys::Down)] = pressed;
                }
                VirtualKeyCode::Left => {
                    self.keys[key_to_index(Keys::Left)] = pressed;
                }
                VirtualKeyCode::Right => {
                    self.keys[key_to_index(Keys::Right)] = pressed;
                }
                VirtualKeyCode::H => {
                    self.keys[key_to_index(Keys::H)] = pressed;
                }
                VirtualKeyCode::J => {
                    self.keys[key_to_index(Keys::J)] = pressed;
                }
                VirtualKeyCode::K => {
                    self.keys[key_to_index(Keys::K)] = pressed;
                }
                VirtualKeyCode::U => {
                    self.keys[key_to_index(Keys::U)] = pressed;
                }
                VirtualKeyCode::Space => {
                    self.keys[key_to_index(Keys::Space)] = pressed;
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
        self.keys[key_to_index(key)]
    }

    //returns true on the first frame that the key is down
    pub fn get_key_down(&self, key: Keys) -> bool {
        let ind = key_to_index(key);
        self.keys[ind] && !self.keys_prev_frame[ind]
    }

    //returns true on the first frame that the key is up
    pub fn get_key_up(&self, key: Keys) -> bool {
        let ind = key_to_index(key);
        !self.keys[ind] && self.keys_prev_frame[ind]
    }
}
