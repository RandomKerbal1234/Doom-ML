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
/// Upper bound on fixed simulation steps per rendered frame in watch mode.
const MAX_STEPS_PER_FRAME: usize = 8;

use game::Game;
use levels::initialize_levels;
use ui::{draw_overlay, draw_settings_overlay};

struct AgentPolicy {
    network: game::Policy,
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

/// Reads the leading number (integer or decimal) from a JSON value's text.
fn parse_number<T: std::str::FromStr>(raw: &str) -> Option<T> {
    let end = raw
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .map(|(index, c)| index + c.len_utf8())
        .last()
        .unwrap_or(0);
    raw.get(..end)?.parse().ok()
}

/// Mirrors `training/env.py::start_level_bounds`: a 1-based inclusive `[low,
/// high]` range, the shorthand `0` (whole campaign) or `N` (pinned level).
fn parse_level_range(source: &str, campaign: usize) -> Option<(usize, usize)> {
    let raw = policy_value(source, "start_level")?;
    if let Some(rest) = raw.strip_prefix('[') {
        let inside = rest.split(']').next()?;
        let mut parts = inside.split(',');
        let low = parse_number::<usize>(parts.next()?.trim())?;
        let high = parse_number::<usize>(parts.next()?.trim())?;
        if low == 0 || high == 0 || low > high {
            return None;
        }
        let high = high.min(campaign);
        let low = low.min(high);
        return Some((low - 1, high - 1));
    }
    let level = parse_number::<usize>(raw)?;
    if level == 0 {
        Some((0, campaign - 1))
    } else {
        let level = level.min(campaign);
        Some((level - 1, level - 1))
    }
}

/// Settings the watcher mirrors from `config.json` so a watched episode runs
/// under the same conditions as a training episode.
#[derive(Clone, Copy)]
struct WatchSettings {
    start_level_low: usize,
    start_level_high: usize,
    step_seconds: f32,
    max_steps: usize,
    max_levels: usize,
}

impl WatchSettings {
    fn from_config(path: &Path) -> Self {
        let campaign = levels::levels().len().max(1);
        let mut settings = Self {
            start_level_low: 0,
            start_level_high: campaign - 1,
            step_seconds: 0.05,
            max_steps: 15000,
            max_levels: campaign,
        };
        let Ok(source) = fs::read_to_string(path) else {
            return settings;
        };
        if let Some((low, high)) = parse_level_range(&source, campaign) {
            settings.start_level_low = low;
            settings.start_level_high = high;
        }
        if let Some(step) = policy_value(&source, "time_step_seconds").and_then(parse_number::<f32>)
        {
            if step > 0.0 && step <= 0.05 {
                settings.step_seconds = step;
            }
        }
        if let Some(steps) = policy_value(&source, "max_steps").and_then(parse_number::<usize>) {
            if steps > 0 {
                settings.max_steps = steps;
            }
        }
        if let Some(levels) =
            policy_value(&source, "max_levels_per_agent").and_then(parse_number::<usize>)
        {
            if levels > 0 {
                settings.max_levels = levels.min(campaign);
            }
        }
        settings
    }

    /// Draws a start level from the configured range (0-based, inclusive).
    fn random_start_level(&self) -> usize {
        let span = self.start_level_high.saturating_sub(self.start_level_low) + 1;
        self.start_level_low + macroquad::rand::gen_range(0, span)
    }
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
    let shape = policy_value(&source, "policy_shape")
        .and_then(|value| value.strip_prefix('"'))
        .and_then(|value| value.split('"').next())
        .ok_or("missing policy_shape")?
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|_| "invalid policy_shape".to_string())
        })
        .collect::<Result<Vec<usize>, _>>()?;
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
    let network = game::Policy::new(&shape, weights)?;
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(|error| error.to_string())?;
    Ok(AgentPolicy {
        network,
        generation,
        difficulty,
        modified,
    })
}

fn start_agent_episode(game: &mut Game, policy: &AgentPolicy, watch: &WatchSettings) {
    game.agent_start(watch.random_start_level(), &policy.difficulty);
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
    // Seed before the first episode; `SystemTime` only has microsecond
    // resolution on some platforms, which used to make the modulo pick a
    // constant level. quad-rand mixes the whole seed.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|time| time.as_nanos() as u64)
        .unwrap_or(0)
        ^ (u64::from(std::process::id()) << 32);
    macroquad::rand::srand(seed);

    let mut watch_path: Option<PathBuf> = None;
    let mut config_path = PathBuf::from("config.json");
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--watch-agent" => {
                watch_path =
                    Some(PathBuf::from(arguments.next().unwrap_or_else(|| {
                        "training/output/best_policy.json".to_string()
                    })));
            }
            "--config" => {
                if let Some(path) = arguments.next() {
                    config_path = PathBuf::from(path);
                }
            }
            _ => {}
        }
    }

    let watch = WatchSettings::from_config(&config_path);
    let mut policy = watch_path
        .as_ref()
        .and_then(|path| load_agent_policy(path).ok());
    if let Some(policy) = &policy {
        start_agent_episode(&mut game, policy, &watch);
    }
    let mut next_policy_check = 0.0;
    let mut completed_levels = 0usize;
    let mut episode_steps = 0usize;
    let mut watch_accumulator = 0.0f32;
    let mut frame = Image::empty();
    frame.width = WIDTH as u16;
    frame.height = HEIGHT as u16;
    frame.bytes = game.pixels.clone();
    let texture = Texture2D::from_image(&frame);
    texture.set_filter(FilterMode::Nearest);
    let mut previous = get_time();

    loop {
        let now = get_time();
        let elapsed = (now - previous).clamp(0.0, 0.25) as f32;
        let dt = elapsed.min(0.05);
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
                            start_agent_episode(&mut game, &updated, &watch);
                            completed_levels = 0;
                            episode_steps = 0;
                            watch_accumulator = 0.0;
                        }
                        policy = Some(updated);
                    }
                }
            }
            if is_key_pressed(KeyCode::Q) {
                macroquad::miniquad::window::quit();
            }
            if let Some(policy) = &policy {
                let episode_over = game.dead
                    || completed_levels >= watch.max_levels
                    || episode_steps >= watch.max_steps;
                if is_key_pressed(KeyCode::R) || episode_over {
                    start_agent_episode(&mut game, policy, &watch);
                    completed_levels = 0;
                    episode_steps = 0;
                    watch_accumulator = 0.0;
                } else {
                    // Training steps the simulation at a fixed timestep and lets
                    // the policy act once per step. Stepping at the frame rate
                    // changes the control frequency and how fast the recurrent
                    // memory decays, so match the trainer instead.
                    watch_accumulator += elapsed;
                    let mut applied = 0;
                    while watch_accumulator >= watch.step_seconds && applied < MAX_STEPS_PER_FRAME {
                        let previous_level = game.level_index;
                        let action = game.agent_policy_action(&policy.network);
                        game.agent_step(
                            action.0,
                            action.1,
                            action.2,
                            action.3,
                            action.4,
                            action.5,
                            action.6,
                            watch.step_seconds,
                        );
                        watch_accumulator -= watch.step_seconds;
                        applied += 1;
                        episode_steps += 1;
                        if game.level_index != previous_level {
                            completed_levels += 1;
                        }
                        if game.dead
                            || completed_levels >= watch.max_levels
                            || episode_steps >= watch.max_steps
                        {
                            break;
                        }
                    }
                    // A long stall (window drag, breakpoint) must not trigger a
                    // huge catch-up burst of simulation.
                    if applied == MAX_STEPS_PER_FRAME {
                        watch_accumulator = watch_accumulator.min(watch.step_seconds);
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
                                 active.generation + 1, completed_levels, watch.max_levels),
            );
            draw_text(&status, 12.0, 20.0, 16.0, Color::new(1.0, 0.82, 0.48, 1.0));
        }
        next_frame().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy_file(name: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        fs::write(&path, text).unwrap();
        path
    }

    /// The watcher parses `best_policy.json` by string search, so the shape and
    /// weight plumbing still needs a direct check.
    #[test]
    fn loads_shape_and_weights() {
        let weights = (0..160)
            .map(|index| format!("{:.3}", index as f32 * 0.01))
            .collect::<Vec<_>>()
            .join(",");
        let text = format!(
            "{{\n  \"architecture\": {{\"inputs\": 64, \"hidden_layers\": [2], \"outputs\": 10}},\n  \
             \"doom_difficulty\": \"easy\",\n  \"generation\": 7,\n  \
             \"policy_shape\": \"64,2,10\",\n  \"result\": {{}},\n  \"weights\": [{weights}]\n}}"
        );
        let path = policy_file("doom-policy-ok.json", &text);
        let policy = load_agent_policy(&path).expect("policy should load");
        assert_eq!(policy.generation, 7);
        assert_eq!(policy.difficulty, "easy");
        assert_eq!(policy.network.weight_count(), 160);
    }

    #[test]
    fn rejects_missing_shape() {
        let text = "{\"generation\": 1, \"weights\": [0,0,0]}";
        let path = policy_file("doom-policy-bad.json", text);
        assert!(load_agent_policy(&path).is_err());
    }

    /// Watched episodes must start inside the configured 1-based range.
    #[test]
    fn parses_start_level_range() {
        let source = r#"{ "start_level": [1, 8], "campaign_levels": 50 }"#;
        assert_eq!(parse_level_range(source, 50), Some((0, 7)));
        assert_eq!(
            parse_level_range(r#"{ "start_level": [3, 3] }"#, 50),
            Some((2, 2))
        );
        assert_eq!(parse_level_range(r#"{ "start_level": [0, 4] }"#, 50), None);
        assert_eq!(parse_level_range(r#"{ "start_level": [9, 4] }"#, 50), None);
    }

    #[test]
    fn parses_start_level_shorthand() {
        assert_eq!(
            parse_level_range(r#"{ "start_level": 0 }"#, 50),
            Some((0, 49))
        );
        assert_eq!(
            parse_level_range(r#"{ "start_level": 7 }"#, 50),
            Some((6, 6))
        );
        assert_eq!(parse_level_range(r#"{ "fitness": {} }"#, 50), None);
    }

    /// The watcher now draws from the configured range instead of `nanos %
    /// levels`, which was always 0 on microsecond-resolution clocks.
    #[test]
    fn start_level_stays_in_range() {
        let watch = WatchSettings {
            start_level_low: 3,
            start_level_high: 7,
            step_seconds: 0.05,
            max_steps: 15000,
            max_levels: 50,
        };
        for _ in 0..200 {
            let level = watch.random_start_level();
            assert!((3..=7).contains(&level), "unexpected level {level}");
        }
    }

    /// Watched episodes must mirror the trainer's start levels, timestep, and
    /// episode caps from `config.json`.
    #[test]
    fn reads_watch_settings_from_config() {
        initialize_levels().ok();
        let path = policy_file(
            "doom-watch-config.json",
            r#"{
              "start_level": [1, 8],
              "time_step_seconds": 0.05,
              "max_steps": 15000,
              "max_levels_per_agent": 50
            }"#,
        );
        let watch = WatchSettings::from_config(&path);
        assert_eq!((watch.start_level_low, watch.start_level_high), (0, 7));
        assert!((watch.step_seconds - 0.05).abs() < 1e-6);
        assert_eq!(watch.max_steps, 15000);
        assert_eq!(watch.max_levels, 50);
    }
}
