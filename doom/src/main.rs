// On Windows, launch as a GUI application so opening the game does not spawn
// a console window. macOS and Linux do not use this subsystem attribute.
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use macroquad::prelude::*;

mod game;
mod levels;
mod render;
mod settings;
mod ui;

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const VIEW_HEIGHT: usize = 170;
const HUD_TOP: usize = 172;

use game::Game;
use levels::initialize_levels;
use ui::{draw_overlay, draw_settings_overlay};

fn window_conf() -> Conf {
    Conf {
        window_title: "DOOM: Hell on Mars".to_string(),
        window_width: 960,
        window_height: 600,
        window_resizable: true,
        sample_count: 1,
        high_dpi: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    if let Err(error) = initialize_levels() {
        eprintln!("Failed to load game levels: {error}");
        return;
    }
    let mut game = Game::new();
    let mut frame = Image::empty();
    frame.width = WIDTH as u16;
    frame.height = HEIGHT as u16;
    frame.bytes = game.pixels.clone();
    let texture = Texture2D::from_image(&frame);
    texture.set_filter(FilterMode::Nearest);
    let mut previous = get_time();

    loop {
        let now = get_time();
        let dt = (now - previous).clamp(0.0, 0.05) as f32;
        previous = now;

        game.handle_input(dt);
        game.update(dt);
        game.render();
        texture.update_from_bytes(WIDTH as u32, HEIGHT as u32, &game.pixels);

        clear_background(Color::from_rgba(5, 5, 6, 255));
        let sw = screen_width();
        let sh = screen_height();
        let scale = (sw / WIDTH as f32).min(sh / HEIGHT as f32);
        let draw_w = WIDTH as f32 * scale;
        let draw_h = HEIGHT as f32 * scale;
        let draw_x = (sw - draw_w) * 0.5;
        let draw_y = (sh - draw_h) * 0.5;
        draw_texture_ex(
            &texture,
            draw_x,
            draw_y,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(draw_w, draw_h)),
                ..Default::default()
            },
        );

        if game.settings_open {
            draw_settings_overlay(&game);
        } else if game.paused {
            draw_overlay(draw_x, draw_y, draw_w, draw_h, false, false, &game);
        } else if game.dead {
            draw_overlay(draw_x, draw_y, draw_w, draw_h, true, false, &game);
        } else if game.won {
            draw_overlay(draw_x, draw_y, draw_w, draw_h, true, true, &game);
        }
        next_frame().await;
    }
}
