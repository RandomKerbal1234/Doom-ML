use std::sync::OnceLock;

include!(concat!(env!("OUT_DIR"), "/embedded_levels.rs"));

pub(crate) const MAP_WIDTH: usize = 16;
pub(crate) const MAP_HEIGHT: usize = 12;

static LEVELS: OnceLock<Vec<Level>> = OnceLock::new();

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PickupKind {
    Shotgun,
    Chaingun,
    RocketLauncher,
    Medkit,
    Armor,
    Ammo,
}

#[derive(Clone, Copy)]
pub(crate) struct PickupDef {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) kind: PickupKind,
}

pub(crate) struct Level {
    pub(crate) code: String,
    pub(crate) title: String,
    pub(crate) map: Vec<String>,
    pub(crate) spawn: (f32, f32, f32),
    pub(crate) enemies: Vec<(f32, f32, u8)>,
    pub(crate) pickups: Vec<PickupDef>,
    pub(crate) exit: (f32, f32),
}

pub(crate) fn initialize_levels() -> Result<(), String> {
    let loaded = EMBEDDED_LEVELS
        .iter()
        .map(|(filename, contents)| parse_level_file(filename, contents))
        .collect::<Result<Vec<_>, _>>()?;
    for i in 0..loaded.len() {
        if loaded[..i].iter().any(|other| other.code == loaded[i].code) {
            return Err(format!("duplicate level code: {}", loaded[i].code));
        }
    }
    LEVELS
        .set(loaded)
        .map_err(|_| "levels were initialized more than once".to_string())
}

fn parse_level_file(source_name: &str, contents: &str) -> Result<Level, String> {
    let mut code = None;
    let mut title = None;
    let mut spawn = None;
    let mut exit = None;
    let mut map = Vec::new();
    let mut enemies = Vec::new();
    let mut pickups = Vec::new();
    let mut section = "";

    for (line_number, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with("//") {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = &line[1..line.len() - 1];
            if !matches!(section, "map" | "enemies" | "pickups") {
                return Err(level_parse_error(
                    source_name,
                    line_number,
                    "unknown section",
                ));
            }
            continue;
        }

        let parsed = match section {
            "map" => {
                map.push(line.to_string());
                Ok(())
            }
            "enemies" => parse_enemy(line).map(|enemy| enemies.push(enemy)),
            "pickups" => parse_pickup(line).map(|pickup| pickups.push(pickup)),
            _ => {
                let Some((key, value)) = line.split_once('=') else {
                    return Err(level_parse_error(
                        source_name,
                        line_number,
                        "expected key=value",
                    ));
                };
                match key.trim() {
                    "code" => {
                        code = Some(value.trim().to_string());
                        Ok(())
                    }
                    "title" => {
                        title = Some(value.trim().to_string());
                        Ok(())
                    }
                    "spawn" => parse_triplet(value).map(|pos| spawn = Some(pos)),
                    "exit" => parse_pair(value).map(|pos| exit = Some(pos)),
                    _ => Err("unknown field".to_string()),
                }
            }
        };
        parsed.map_err(|error| level_parse_error(source_name, line_number, &error))?;
    }

    if map.len() != MAP_HEIGHT || map.iter().any(|row| row.len() != MAP_WIDTH) {
        return Err(format!(
            "{source_name} must contain exactly {MAP_HEIGHT} map rows of {MAP_WIDTH} characters"
        ));
    }
    for (y, row) in map.iter().enumerate() {
        for (x, tile) in row.bytes().enumerate() {
            if !matches!(tile, b'#' | b'B' | b'M' | b'.' | b'X') {
                return Err(format!("{source_name} has invalid map tile at ({x}, {y})"));
            }
        }
    }

    let level = Level {
        code: code.ok_or_else(|| format!("{source_name} is missing code="))?,
        title: title.ok_or_else(|| format!("{source_name} is missing title="))?,
        map,
        spawn: spawn.ok_or_else(|| format!("{source_name} is missing spawn="))?,
        enemies,
        pickups,
        exit: exit.ok_or_else(|| format!("{source_name} is missing exit="))?,
    };
    validate_level(&level, source_name)?;
    Ok(level)
}

fn level_parse_error(source_name: &str, zero_based_line: usize, message: &str) -> String {
    format!("{source_name}:{}: {message}", zero_based_line + 1)
}

fn parse_number(value: &str) -> Result<f32, String> {
    let parsed = value
        .trim()
        .parse::<f32>()
        .map_err(|_| "invalid number".to_string())?;
    if !parsed.is_finite() {
        return Err("coordinates must be finite".to_string());
    }
    Ok(parsed)
}

fn parse_pair(value: &str) -> Result<(f32, f32), String> {
    let mut parts = value.split(',');
    let x = parse_number(parts.next().ok_or_else(|| "expected x,y".to_string())?)?;
    let y = parse_number(parts.next().ok_or_else(|| "expected x,y".to_string())?)?;
    if parts.next().is_some() {
        return Err("expected exactly two comma-separated values".to_string());
    }
    Ok((x, y))
}

fn parse_triplet(value: &str) -> Result<(f32, f32, f32), String> {
    let mut parts = value.split(',');
    let x = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,angle".to_string())?,
    )?;
    let y = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,angle".to_string())?,
    )?;
    let angle = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,angle".to_string())?,
    )?;
    if parts.next().is_some() {
        return Err("expected exactly three comma-separated values".to_string());
    }
    Ok((x, y, angle))
}

fn parse_enemy(line: &str) -> Result<(f32, f32, u8), String> {
    let mut parts = line.split(',');
    let x = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,type".to_string())?,
    )?;
    let y = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,type".to_string())?,
    )?;
    let kind = match parts.next().map(str::trim) {
        Some("imp") => 0,
        Some("demon") => 1,
        _ => return Err("enemy type must be imp or demon".to_string()),
    };
    if parts.next().is_some() {
        return Err("expected exactly three comma-separated values".to_string());
    }
    Ok((x, y, kind))
}

fn parse_pickup(line: &str) -> Result<PickupDef, String> {
    let mut parts = line.split(',');
    let x = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,type".to_string())?,
    )?;
    let y = parse_number(
        parts
            .next()
            .ok_or_else(|| "expected x,y,type".to_string())?,
    )?;
    let kind = match parts.next().map(str::trim) {
        Some("shotgun") => PickupKind::Shotgun,
        Some("chaingun") => PickupKind::Chaingun,
        Some("rocket_launcher") => PickupKind::RocketLauncher,
        Some("medkit") => PickupKind::Medkit,
        Some("armor") => PickupKind::Armor,
        Some("ammo") => PickupKind::Ammo,
        _ => return Err("unknown pickup type".to_string()),
    };
    if parts.next().is_some() {
        return Err("expected exactly three comma-separated values".to_string());
    }
    Ok(PickupDef { x, y, kind })
}

fn validate_level(level: &Level, source_name: &str) -> Result<(), String> {
    let is_open = |x: f32, y: f32| {
        x >= 0.0
            && y >= 0.0
            && x < MAP_WIDTH as f32
            && y < MAP_HEIGHT as f32
            && !matches!(
                level.map[y.floor() as usize].as_bytes()[x.floor() as usize],
                b'#' | b'B' | b'M'
            )
    };
    if !is_open(level.spawn.0, level.spawn.1) {
        return Err(format!(
            "{source_name} spawn must be inside an open map tile"
        ));
    }
    if !is_open(level.exit.0, level.exit.1) {
        return Err(format!(
            "{source_name} exit must be inside an open map tile"
        ));
    }
    if level.enemies.iter().any(|&(x, y, _)| !is_open(x, y))
        || level
            .pickups
            .iter()
            .any(|pickup| !is_open(pickup.x, pickup.y))
    {
        return Err(format!(
            "{source_name} has an enemy or pickup inside a wall"
        ));
    }
    Ok(())
}

pub(crate) fn levels() -> &'static [Level] {
    LEVELS
        .get()
        .expect("levels must be loaded before creating the game")
}

pub(crate) fn level_tile(level_index: usize, x: i32, y: i32) -> u8 {
    if x < 0 || y < 0 || x >= MAP_WIDTH as i32 || y >= MAP_HEIGHT as i32 {
        return b'#';
    }
    levels()
        .get(level_index)
        .and_then(|level| level.map.get(y as usize))
        .and_then(|row| row.as_bytes().get(x as usize))
        .copied()
        .unwrap_or(b'#')
}

pub(crate) fn tile_is_wall(tile: u8) -> bool {
    matches!(tile, b'#' | b'B' | b'M')
}

pub(crate) fn has_line_of_sight(
    level_index: usize,
    from_x: f32,
    from_y: f32,
    to_x: f32,
    to_y: f32,
) -> bool {
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 0.25 {
        return true;
    }
    let steps = (dist * 8.0) as usize;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        if tile_is_wall(level_tile(
            level_index,
            (from_x + dx * t).floor() as i32,
            (from_y + dy * t).floor() as i32,
        )) {
            return false;
        }
    }
    true
}
