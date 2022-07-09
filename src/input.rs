use std::ops::{Index, IndexMut};
use strum::EnumCount;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

#[derive(EnumCount)]
pub enum Keys {
    W,
    A,
    S,
    D,
    Plus,
    Minus,
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
        Keys::Plus => 4,
        Keys::Minus => 5,
        Keys::Right => 6,
        Keys::Left => 7,
        Keys::Up => 8,
        Keys::Down => 9,
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
                VirtualKeyCode::Plus => {
                    self.keys[key_to_index(Keys::Plus)] = pressed;
                }
                VirtualKeyCode::Minus => {
                    self.keys[key_to_index(Keys::Minus)] = pressed;
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
                _ => {}
            }
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