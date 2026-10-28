//! Keyboard and mouse handling for the windowed game.
//!
//! This module is deliberately kept out of the headless training binary so that
//! `doom_headless` does not need to link macroquad or the macOS windowing
//! frameworks. Only `main.rs` includes it.

use crate::game::{distance, Game};
use crate::levels::levels;
use crate::settings::{settings_for_category, DifficultyPreset};
use macroquad::prelude::*;

impl Game {
    pub(crate) fn handle_settings_input(&mut self) {
        if is_key_pressed(KeyCode::Escape) {
            self.toggle_settings();
            return;
        }
        if is_key_pressed(KeyCode::Key1) {
            self.apply_preset(DifficultyPreset::Easy);
            return;
        }
        if is_key_pressed(KeyCode::Key2) {
            self.apply_preset(DifficultyPreset::Medium);
            return;
        }
        if is_key_pressed(KeyCode::Key3) {
            self.apply_preset(DifficultyPreset::Hard);
            return;
        }

        if is_key_pressed(KeyCode::Tab) || is_key_pressed(KeyCode::PageDown) {
            self.settings_category = (self.settings_category + 1) % 4;
            self.settings_selection = 0;
        } else if is_key_pressed(KeyCode::PageUp) {
            self.settings_category = (self.settings_category + 3) % 4;
            self.settings_selection = 0;
        }

        let count = settings_for_category(self.settings_category).len();
        if is_key_pressed(KeyCode::Down) {
            self.settings_selection = (self.settings_selection + 1) % count;
        }
        if is_key_pressed(KeyCode::Up) {
            self.settings_selection = (self.settings_selection + count - 1) % count;
        }
        let direction = if is_key_pressed(KeyCode::Right) {
            1.0
        } else if is_key_pressed(KeyCode::Left) {
            -1.0
        } else {
            0.0
        };
        if direction != 0.0 {
            let definition = settings_for_category(self.settings_category)[self.settings_selection];
            let old_value = self.setting_value(definition.key);
            let value =
                (old_value + definition.step * direction).clamp(definition.min, definition.max);
            self.write_setting_value(definition.key, value);
            self.difficulty = DifficultyPreset::Custom;
        }
    }

    pub(crate) fn handle_input(&mut self, dt: f32) {
        let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        if shift_down && is_key_pressed(KeyCode::P) {
            self.toggle_settings();
            return;
        }
        if self.settings_open {
            self.handle_settings_input();
            return;
        }

        if is_key_pressed(KeyCode::F1) && !self.dead && !self.won {
            self.paused = !self.paused;
        }

        if self.paused {
            if is_key_pressed(KeyCode::F10) {
                macroquad::miniquad::window::quit();
                return;
            }
            if is_key_pressed(KeyCode::Up) {
                self.selected_level = (self.selected_level + levels().len() - 1) % levels().len();
            }
            if is_key_pressed(KeyCode::Down) {
                self.selected_level = (self.selected_level + 1) % levels().len();
            }
            if is_key_pressed(KeyCode::Q) {
                self.start_new_run(0);
            } else if is_key_pressed(KeyCode::Enter) {
                self.start_new_run(self.selected_level);
            } else if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Space) {
                self.paused = false;
            }
            return;
        }

        if self.dead || self.won {
            if is_key_pressed(KeyCode::R) || is_key_pressed(KeyCode::Enter) {
                self.restart();
            }
            return;
        }

        if is_key_pressed(KeyCode::Escape) {
            self.paused = true;
            return;
        }

        if is_key_pressed(KeyCode::Key1) {
            self.choose_weapon(0);
        }
        if is_key_pressed(KeyCode::Key2) {
            self.choose_weapon(1);
        }
        if is_key_pressed(KeyCode::Key3) {
            self.choose_weapon(2);
        }
        if is_key_pressed(KeyCode::Key4) {
            self.choose_weapon(3);
        }
        let (_, wheel_y) = mouse_wheel();
        if wheel_y > 0.0 {
            self.cycle_weapon(1);
        }
        if wheel_y < 0.0 {
            self.cycle_weapon(-1);
        }

        let turn = (is_key_down(KeyCode::Right) as i8 - is_key_down(KeyCode::Left) as i8) as f32;
        let running = shift_down;
        let turn_speed = if running {
            self.settings.run_turn_speed
        } else {
            self.settings.turn_speed
        };
        self.angle = (self.angle + turn * turn_speed * dt).rem_euclid(std::f32::consts::TAU);

        let forward = (is_key_down(KeyCode::W) as i8 - is_key_down(KeyCode::S) as i8) as f32;
        let strafe = (is_key_down(KeyCode::D) as i8 - is_key_down(KeyCode::A) as i8) as f32;
        let mut vx = self.angle.cos() * forward - self.angle.sin() * strafe;
        let mut vy = self.angle.sin() * forward + self.angle.cos() * strafe;
        let magnitude = (vx * vx + vy * vy).sqrt();
        if magnitude > 1.0 {
            vx /= magnitude;
            vy /= magnitude;
        }
        let speed = if running {
            self.settings.run_speed
        } else {
            self.settings.walk_speed
        };
        let (nx, ny) = self.try_move(self.x + vx * speed * dt, self.y + vy * speed * dt);
        self.x = nx;
        self.y = ny;

        if is_key_pressed(KeyCode::E) {
            let exit = self.current_level().exit;
            if distance(self.x, self.y, exit.0, exit.1) < 1.35 {
                if self.level_index + 1 == levels().len() {
                    self.won = true;
                } else {
                    self.load_level(self.level_index + 1);
                }
            } else {
                self.message = "FIND THE GREEN EXIT PORTAL".to_string();
                self.message_timer = 1.0;
            }
        }

        if (is_key_down(KeyCode::Space)
            || is_key_down(KeyCode::LeftControl)
            || is_key_down(KeyCode::RightControl)
            || is_mouse_button_down(MouseButton::Left))
            && self.shot_timer <= 0.0
        {
            self.fire();
        }
    }
}
