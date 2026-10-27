#![allow(dead_code)]

use std::io::{self, BufRead, Read};

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const VIEW_HEIGHT: usize = 170;
const HUD_TOP: usize = 172;

#[path = "../game.rs"]
mod game;
#[path = "../levels.rs"]
mod levels;
#[path = "../render.rs"]
mod render;
#[path = "../settings.rs"]
mod settings;

fn main() {
    if let Err(error) = levels::initialize_levels() {
        eprintln!("level initialization failed: {error}");
        std::process::exit(1);
    }
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|arg| arg == "--evaluate") {
        if let Err(error) = evaluate_episode(&arguments[1..]) {
            eprintln!("evaluation failed: {error}");
            std::process::exit(2);
        }
        return;
    }
    let stdin = io::stdin();
    let mut game = game::Game::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let fields = line.trim().split(',').collect::<Vec<_>>();
        match fields.first().copied() {
            Some("reset") => {
                let level = fields.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                game.agent_start(level, fields.get(2).copied().unwrap_or("medium"));
                println!("{}", game.agent_observation());
            }
            Some("step") if fields.len() >= 9 => {
                let n = |i: usize| fields[i].parse::<f32>().unwrap_or(0.0);
                game.agent_step(
                    n(1),
                    n(2),
                    n(3),
                    n(4) != 0.0,
                    n(5) != 0.0,
                    n(6).max(0.0) as usize,
                    n(7) != 0.0,
                    n(8),
                );
                println!("{}", game.agent_observation());
            }
            Some("quit") => break,
            _ => println!("{{\"error\":\"invalid command\"}}"),
        }
    }
}

fn evaluate_episode(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 10 {
        return Err(
            "expected start_level, difficulty, dt, max_steps, max_levels, and five fitness weights"
                .into(),
        );
    }
    let parse = |index: usize| {
        arguments[index]
            .parse::<f32>()
            .map_err(|error| format!("invalid argument {}: {error}", index + 1))
    };
    let start_level = arguments[0]
        .parse::<usize>()
        .map_err(|error| format!("invalid start level: {error}"))?;
    let difficulty = &arguments[1];
    let dt = parse(2)?;
    let max_steps = arguments[3]
        .parse::<usize>()
        .map_err(|error| format!("invalid max steps: {error}"))?;
    let max_levels = arguments[4]
        .parse::<usize>()
        .map_err(|error| format!("invalid max levels: {error}"))?;
    if !(1..=50).contains(&max_levels) {
        return Err("max_levels must be between 1 and 50".into());
    }
    let level_completion = parse(5)?;
    let partial_level = parse(6)?;
    let health_per_point = parse(7)?;
    let kill_reward = parse(8)?;
    let seconds_penalty = arguments[9]
        .parse::<f32>()
        .map_err(|error| format!("invalid seconds penalty: {error}"))?;

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read policy weights: {error}"))?;
    let weights = input
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != 430 {
        return Err(format!(
            "expected 430 policy weights, got {}",
            weights.len()
        ));
    }

    let mut game = game::Game::new();
    game.agent_start(start_level, difficulty);
    let mut level_count = 0;
    let mut level_kills = 0;
    let mut total_kills = 0;
    let mut total_time = 0.0;
    let mut level_start_time = 0.0;
    let mut score = 0.0;
    let mut steps = 0;
    while steps < max_steps {
        let enemies_before = game.enemies.iter().filter(|enemy| enemy.alive).count();
        let previous_level = game.level_index;
        let action = game.agent_policy_action(&weights);
        game.agent_step(
            action.0, action.1, action.2, action.3, action.4, action.5, action.6, dt,
        );
        let enemies_after = game.enemies.iter().filter(|enemy| enemy.alive).count();
        let killed = if game.level_index == previous_level {
            enemies_before.saturating_sub(enemies_after)
        } else {
            0
        };
        level_kills += killed;
        total_kills += killed;
        steps += 1;
        total_time += dt;
        if game.level_index != previous_level {
            level_count += 1;
            score += level_completion;
            score += game.health as f32 * health_per_point;
            score += level_kills as f32 * kill_reward;
            score -= (total_time - level_start_time) * seconds_penalty;
            level_kills = 0;
            level_start_time = total_time;
        }
        if game.dead || level_count >= max_levels {
            break;
        }
    }
    if level_count < max_levels {
        score += level_count as f32 * partial_level;
        score += game.health as f32 * health_per_point;
        score += level_kills as f32 * kill_reward;
        score -= (total_time - level_start_time) * seconds_penalty;
    }
    println!(
        "{score},{level_count},{total_kills},{},{total_time:.2},{start_level}",
        game.health
    );
    Ok(())
}
