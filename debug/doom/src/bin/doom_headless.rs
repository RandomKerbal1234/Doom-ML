#![allow(dead_code)]

use std::io::{self, BufRead, Read};
use std::time::{Duration, Instant};

const WIDTH: usize = 320;
const HEIGHT: usize = 200;
const VIEW_HEIGHT: usize = 170;
const HUD_TOP: usize = 172;

#[path = "../game.rs"]
mod game;
#[path = "../levels.rs"]
mod levels;
#[path = "../settings.rs"]
mod settings;

fn main() {
    let main_start = Instant::now();
    if let Err(error) = levels::initialize_levels() {
        eprintln!("level initialization failed: {error}");
        std::process::exit(1);
    }
    let levels_elapsed = main_start.elapsed();
    let profiling = std::env::var_os("DOOM_PROFILE").is_some_and(|value| value != "0");
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("--evaluate") => {
            if let Err(error) = evaluate_episode(
                &arguments[1..],
                main_start,
                levels_elapsed,
                profiling,
            ) {
                eprintln!("evaluation failed: {error}");
                std::process::exit(2);
            }
            return;
        }
        Some("--evaluate-server") => {
            if let Err(error) = evaluate_server(profiling) {
                eprintln!("evaluation server failed: {error}");
                std::process::exit(2);
            }
            return;
        }
        _ => {}
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

struct EpisodeParams {
    start_level: usize,
    difficulty: String,
    dt: f32,
    max_steps: usize,
    level_completion: f32,
    partial_level: f32,
    health_per_point: f32,
    kill_reward: f32,
    seconds_penalty: f32,
}

struct ProfileSnapshot {
    steps: usize,
    levels: usize,
    main_ns: u64,
    levels_ns: u64,
    setup_ns: u64,
    policy_ns: u64,
    sim_ns: u64,
    killcount_ns: u64,
    loop_other_ns: u64,
    total_ns: u64,
    level_load_ns: u64,
    movement_ns: u64,
    fire_ns: u64,
    update_ns: u64,
    enemy_ns: u64,
    danger_ns: u64,
    pickup_ns: u64,
}

impl ProfileSnapshot {
    fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"levels\":{},\"main_ns\":{},\"levels_ns\":{},\
             \"setup_ns\":{},\"policy_ns\":{},\"sim_ns\":{},\"killcount_ns\":{},\
             \"loop_other_ns\":{},\"total_ns\":{},\"level_load_ns\":{},\
             \"movement_ns\":{},\"fire_ns\":{},\"update_ns\":{},\"enemy_ns\":{},\
             \"danger_ns\":{},\"pickup_ns\":{}}}",
            self.steps,
            self.levels,
            self.main_ns,
            self.levels_ns,
            self.setup_ns,
            self.policy_ns,
            self.sim_ns,
            self.killcount_ns,
            self.loop_other_ns,
            self.total_ns,
            self.level_load_ns,
            self.movement_ns,
            self.fire_ns,
            self.update_ns,
            self.enemy_ns,
            self.danger_ns,
            self.pickup_ns,
        )
    }
}

struct EpisodeResult {
    score: f32,
    levels: usize,
    kills: usize,
    health: i32,
    seconds: f32,
    start_level: usize,
    profile: Option<ProfileSnapshot>,
}

fn parse_cli_params(arguments: &[String]) -> Result<EpisodeParams, String> {
    if arguments.len() != 9 {
        return Err(
            "expected start_level, difficulty, dt, max_steps, and five fitness weights".into(),
        );
    }
    let parse = |index: usize| {
        arguments[index]
            .parse::<f32>()
            .map_err(|error| format!("invalid argument {}: {error}", index + 1))
    };
    Ok(EpisodeParams {
        start_level: arguments[0]
            .parse::<usize>()
            .map_err(|error| format!("invalid start level: {error}"))?,
        difficulty: arguments[1].clone(),
        dt: parse(2)?,
        max_steps: arguments[3]
            .parse::<usize>()
            .map_err(|error| format!("invalid max steps: {error}"))?,
        level_completion: parse(4)?,
        partial_level: parse(5)?,
        health_per_point: parse(6)?,
        kill_reward: parse(7)?,
        seconds_penalty: parse(8)?,
    })
}

fn parse_job(line: &str) -> Result<(EpisodeParams, Vec<f32>), String> {
    let fields = line.split(',').collect::<Vec<_>>();
    if fields.len() < 9 {
        return Err(format!("expected at least 9 fields, got {}", fields.len()));
    }
    let number = |index: usize, name: &str| {
        fields[index]
            .trim()
            .parse::<f32>()
            .map_err(|error| format!("invalid {name}: {error}"))
    };
    let start_level = number(0, "start_level")? as usize;
    let difficulty = fields[1].trim().to_string();
    let dt = number(2, "dt")?;
    let max_steps = number(3, "max_steps")? as usize;
    let level_completion = number(4, "level_completion")?;
    let partial_level = number(5, "partial_level")?;
    let health_per_point = number(6, "health_per_point")?;
    let kill_reward = number(7, "kill_reward")?;
    let seconds_penalty = number(8, "seconds_penalty")?;
    let weights = fields[9..]
        .iter()
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|error| format!("invalid weight: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.len() != 430 {
        return Err(format!(
            "expected 430 policy weights, got {}",
            weights.len()
        ));
    }
    Ok((
        EpisodeParams {
            start_level,
            difficulty,
            dt,
            max_steps,
            level_completion,
            partial_level,
            health_per_point,
            kill_reward,
            seconds_penalty,
        },
        weights,
    ))
}

/// Runs one complete episode. The caller owns the `Game`, which lets the
/// evaluate-server reuse the same allocation across many jobs.
fn run_episode(
    game: &mut game::Game,
    params: &EpisodeParams,
    weights: &[f32],
    profiling: bool,
) -> EpisodeResult {
    let total_start = Instant::now();
    let setup_start = Instant::now();
    game.agent_start(params.start_level, &params.difficulty);
    let setup_ns = setup_start.elapsed().as_nanos() as u64;

    let mut level_count = 0;
    let mut level_kills = 0;
    let mut total_kills = 0;
    let mut total_time = 0.0;
    let mut level_start_time = 0.0;
    let mut score = 0.0;
    let mut steps = 0;
    let mut policy_ns = 0u64;
    let mut sim_ns = 0u64;
    let mut killcount_ns = 0u64;
    let loop_start = Instant::now();
    while steps < params.max_steps {
        let scan_start = Instant::now();
        let kills_before = game.kills;
        killcount_ns += scan_start.elapsed().as_nanos() as u64;
        let previous_level = game.level_index;
        let policy_start = Instant::now();
        let action = game.agent_policy_action(weights);
        policy_ns += policy_start.elapsed().as_nanos() as u64;
        let sim_start = Instant::now();
        game.agent_step(
            action.0, action.1, action.2, action.3, action.4, action.5, action.6, params.dt,
        );
        sim_ns += sim_start.elapsed().as_nanos() as u64;
        let scan_start = Instant::now();
        let killed = if game.level_index == previous_level {
            (game.kills - kills_before) as usize
        } else {
            0
        };
        killcount_ns += scan_start.elapsed().as_nanos() as u64;
        level_kills += killed;
        total_kills += killed;
        steps += 1;
        total_time += params.dt;
        if game.level_index != previous_level {
            level_count += 1;
            score += params.level_completion;
            score += game.health as f32 * params.health_per_point;
            score += level_kills as f32 * params.kill_reward;
            score -= (total_time - level_start_time) * params.seconds_penalty;
            level_kills = 0;
            level_start_time = total_time;
        }
        if game.dead || level_count >= 10 {
            break;
        }
    }
    let loop_ns = loop_start.elapsed().as_nanos() as u64;
    if level_count < 10 {
        score += level_count as f32 * params.partial_level;
        score += game.health as f32 * params.health_per_point;
        score += level_kills as f32 * params.kill_reward;
        score -= (total_time - level_start_time) * params.seconds_penalty;
    }

    let profile = profiling.then(|| {
        let p = game.profile;
        let loop_other_ns = loop_ns
            .saturating_sub(policy_ns)
            .saturating_sub(sim_ns)
            .saturating_sub(killcount_ns);
        ProfileSnapshot {
            steps,
            levels: level_count,
            main_ns: 0,
            levels_ns: 0,
            setup_ns,
            policy_ns,
            sim_ns,
            killcount_ns,
            loop_other_ns,
            total_ns: total_start.elapsed().as_nanos() as u64,
            level_load_ns: p.level_load_ns,
            movement_ns: p.movement_ns,
            fire_ns: p.fire_ns,
            update_ns: p.update_ns,
            enemy_ns: p.enemy_ns,
            danger_ns: p.danger_ns,
            pickup_ns: p.pickup_ns,
        }
    });
    EpisodeResult {
        score,
        levels: level_count,
        kills: total_kills,
        health: game.health,
        seconds: total_time,
        start_level: params.start_level,
        profile,
    }
}

fn evaluate_episode(
    arguments: &[String],
    main_start: Instant,
    levels_elapsed: Duration,
    profiling: bool,
) -> Result<(), String> {
    let params = parse_cli_params(arguments)?;

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
    game.set_headless();
    if profiling {
        game.enable_profiling();
    }
    let result = run_episode(&mut game, &params, &weights, profiling);
    println!(
        "{},{},{},{},{:.2},{}",
        result.score, result.levels, result.kills, result.health, result.seconds,
        result.start_level
    );
    if let Some(profile) = result.profile {
        let mut profile = profile;
        profile.main_ns = main_start.elapsed().as_nanos() as u64;
        profile.levels_ns = levels_elapsed.as_nanos() as u64;
        eprintln!("PROFILE {}", profile.to_json());
    }
    Ok(())
}

fn evaluate_server(profiling: bool) -> Result<(), String> {
    let mut game = game::Game::new();
    game.set_headless();
    if profiling {
        game.enable_profiling();
    }
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let job_start = Instant::now();
        let (params, weights) = match parse_job(&line) {
            Ok(job) => job,
            Err(error) => {
                println!("{{\"error\":\"{}\"}}", error.replace('"', "'"));
                continue;
            }
        };
        let result = run_episode(&mut game, &params, &weights, profiling);
        if let Some(profile) = result.profile {
            let mut profile = profile;
            profile.main_ns = job_start.elapsed().as_nanos() as u64;
            println!(
                "{{\"score\":{},\"levels\":{},\"kills\":{},\"health\":{},\"seconds\":{:.2},\
                 \"start_level\":{},\"profile\":{}}}",
                result.score,
                result.levels,
                result.kills,
                result.health,
                result.seconds,
                result.start_level,
                profile.to_json(),
            );
        } else {
            println!(
                "{{\"score\":{},\"levels\":{},\"kills\":{},\"health\":{},\"seconds\":{:.2},\
                 \"start_level\":{}}}",
                result.score, result.levels, result.kills, result.health, result.seconds,
                result.start_level,
            );
        }
    }
    Ok(())
}
