// On Windows, launch as a GUI application so opening the game does not spawn
// a console window. macOS and Linux do not use this subsystem attribute.
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use macroquad::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

mod game;
mod input;
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

struct AgentPolicy {
    weights: Vec<f32>,
    generation: usize,
    difficulty: String,
    modified: SystemTime,
}

fn policy_value<'a>(source: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let after_key = source.get(source.find(&marker)? + marker.len()..)?;
    let after_colon = after_key.get(after_key.find(':')? + 1..)?.trim_start();
    Some(after_colon)
}

fn load_agent_policy(path: &Path) -> Result<AgentPolicy, String> {
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let generation_text = policy_value(&source, "generation").ok_or("missing generation")?;
    let generation = generation_text
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or("invalid generation")?;
    let difficulty = policy_value(&source, "doom_difficulty")
        .and_then(|value| value.strip_prefix('"'))
        .and_then(|value| value.split('"').next())
        .unwrap_or("medium")
        .to_string();
    let weight_text = policy_value(&source, "weights")
        .and_then(|value| value.strip_prefix('['))
        .and_then(|value| value.split(']').next())
        .ok_or("missing weights")?;
    let weights = weight_text
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|_| "invalid weight".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != 430 {
        return Err(format!("expected 430 weights, found {}", weights.len()));
    }
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(|error| error.to_string())?;
    Ok(AgentPolicy {
        weights,
        generation,
        difficulty,
        modified,
    })
}

fn random_level() -> usize {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|time| (time.as_nanos() % levels::levels().len() as u128) as usize)
        .unwrap_or(0)
}

fn start_agent_episode(game: &mut Game, policy: &AgentPolicy) {
    game.agent_start(random_level(), &policy.difficulty);
}

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
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let watch_path = if arguments.first().map(String::as_str) == Some("--watch-agent") {
        Some(PathBuf::from(
            arguments
                .get(1)
                .map(String::as_str)
                .unwrap_or("training/output/best_policy.json"),
        ))
    } else {
        None
    };
    let mut policy = watch_path
        .as_ref()
        .and_then(|path| load_agent_policy(path).ok());
    if let Some(policy) = &policy {
        start_agent_episode(&mut game, policy);
    }
    let mut next_policy_check = 0.0;
    let mut completed_levels = 0usize;
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

        if let Some(path) = &watch_path {
            if now >= next_policy_check {
                next_policy_check = now + 0.5;
                if let Ok(updated) = load_agent_policy(path) {
                    let changed = policy.as_ref().map_or(true, |current| {
                        updated.modified > current.modified
                            || updated.generation != current.generation
                            || updated.difficulty != current.difficulty
                    });
                    if changed {
                        if policy.is_none()
                            || policy
                                .as_ref()
                                .is_some_and(|current| current.difficulty != updated.difficulty)
                        {
                            start_agent_episode(&mut game, &updated);
                            completed_levels = 0;
                        }
                        policy = Some(updated);
                    }
                }
            }
            if is_key_pressed(KeyCode::Q) {
                macroquad::miniquad::window::quit();
            }
            if let Some(policy) = &policy {
                if is_key_pressed(KeyCode::R)
                    || game.dead
                    || completed_levels >= levels::levels().len()
                {
                    start_agent_episode(&mut game, policy);
                    completed_levels = 0;
                } else {
                    let previous_level = game.level_index;
                    let action = game.agent_policy_action(&policy.weights);
                    game.agent_step(
                        action.0, action.1, action.2, action.3, action.4, action.5, action.6, dt,
                    );
                    if game.level_index != previous_level {
                        completed_levels += 1;
                    }
                }
            }
        } else {
            game.handle_input(dt);
            game.update(dt);
        }
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
        if watch_path.is_some() {
            draw_rectangle(
                0.0,
                0.0,
                screen_width(),
                29.0,
                Color::new(0.02, 0.02, 0.02, 0.82),
            );
            let status = policy.as_ref().map_or_else(
                || "WATCHING BEST AGENT  |  WAITING FOR training/output/best_policy.json  |  Q: QUIT".to_string(),
                |active| format!("WATCHING BEST AGENT  |  GENERATION {}  |  LEVELS {}/{}  |  R: RESTART  Q: QUIT",
                                 active.generation + 1, completed_levels, levels::levels().len()),
            );
            draw_text(&status, 12.0, 20.0, 16.0, Color::new(1.0, 0.82, 0.48, 1.0));
        }
        next_frame().await;
    }
}
