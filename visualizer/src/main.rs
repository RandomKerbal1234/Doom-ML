use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

const BG: Color32 = Color32::from_rgb(12, 17, 25);
const PANEL: Color32 = Color32::from_rgb(20, 27, 38);
const TEXT: Color32 = Color32::from_rgb(224, 232, 242);
const MUTED: Color32 = Color32::from_rgb(126, 143, 162);
const POSITIVE: Color32 = Color32::from_rgb(52, 209, 199);
const NEGATIVE: Color32 = Color32::from_rgb(255, 105, 92);

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    modified: Option<SystemTime>,
    len: u64,
}

#[derive(Clone)]
struct Policy {
    shape: Vec<usize>,
    weights: Vec<f32>,
    generation: usize,
    difficulty: String,
}

struct Visualizer {
    path: PathBuf,
    stamp: Option<FileStamp>,
    policy: Option<Policy>,
    error: Option<String>,
    inspection: Option<String>,
    next_poll: Instant,
    zoom: f32,
    pan: Vec2,
    selected_node: Option<(usize, usize)>,
}

#[derive(Clone, Copy)]
struct Edge {
    from_layer: usize,
    from_node: usize,
    to_node: usize,
    weight: f32,
    a: Pos2,
    b: Pos2,
}

impl Visualizer {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            stamp: None,
            policy: None,
            error: None,
            inspection: None,
            next_poll: Instant::now(),
            zoom: 1.0,
            pan: Vec2::ZERO,
            selected_node: None,
        }
    }

    fn poll_policy(&mut self) {
        if Instant::now() < self.next_poll {
            return;
        }
        self.next_poll = Instant::now() + Duration::from_millis(700);
        let metadata = match fs::metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.error = Some(format!("{} ({})", error, self.path.display()));
                return;
            }
        };
        let stamp = FileStamp {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        };
        if self.stamp.as_ref() == Some(&stamp) {
            return;
        }
        match fs::read_to_string(&self.path)
            .map_err(|error| error.to_string())
            .and_then(|source| parse_policy(&source))
        {
            Ok(policy) => {
                self.policy = Some(policy);
                if self.selected_node.is_some_and(|(layer, node)| {
                    self.policy.as_ref().is_none_or(|policy| {
                        policy.shape.get(layer).is_none_or(|count| node >= *count)
                    })
                }) {
                    self.selected_node = None;
                }
                self.stamp = Some(stamp);
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
    }

    fn zoom_at(&mut self, pointer: Pos2, center: Pos2, factor: f32) {
        let old_zoom = self.zoom;
        let new_zoom = (old_zoom * factor).clamp(0.22, 50.0);
        let canvas_point = (pointer - center - self.pan) / old_zoom;
        self.zoom = new_zoom;
        self.pan = pointer - center - canvas_point * new_zoom;
    }
}

fn field<'a>(source: &'a str, key: &str) -> Result<&'a str, String> {
    let needle = format!("\"{key}\"");
    let start = source
        .find(&needle)
        .ok_or_else(|| format!("missing {key}"))?
        + needle.len();
    let rest = source[start..].trim_start();
    rest.strip_prefix(':')
        .map(str::trim_start)
        .ok_or_else(|| format!("invalid {key}"))
}

fn parse_string(source: &str, key: &str) -> Result<String, String> {
    let rest = field(source, key)?
        .strip_prefix('"')
        .ok_or_else(|| format!("invalid {key}"))?;
    let end = rest.find('"').ok_or_else(|| format!("invalid {key}"))?;
    Ok(rest[..end].to_string())
}

fn parse_usize(source: &str, key: &str) -> Result<usize, String> {
    let rest = field(source, key)?;
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().map_err(|_| format!("invalid {key}"))
}

fn parse_weights(source: &str) -> Result<Vec<f32>, String> {
    let rest = field(source, "weights")?
        .strip_prefix('[')
        .ok_or("weights must be an array")?;
    let end = rest.find(']').ok_or("unterminated weights array")?;
    rest[..end]
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|_| "invalid weight".to_string())
        })
        .collect()
}

fn parse_policy(source: &str) -> Result<Policy, String> {
    let shape: Vec<usize> = parse_string(source, "policy_shape")?
        .split(',')
        .map(|part| part.parse().map_err(|_| "invalid policy_shape".to_string()))
        .collect::<Result<_, _>>()?;
    if shape.len() < 2 || shape.iter().any(|width| *width == 0) {
        return Err("policy_shape needs at least two nonzero layers".into());
    }
    let expected: usize = shape.windows(2).map(|pair| pair[1] * (pair[0] + 1)).sum();
    let weights = parse_weights(source)?;
    if weights.len() != expected {
        return Err(format!(
            "expected {expected} weights, found {}",
            weights.len()
        ));
    }
    Ok(Policy {
        shape,
        weights,
        generation: parse_usize(source, "generation")?,
        difficulty: parse_string(source, "doom_difficulty").unwrap_or_else(|_| "unknown".into()),
    })
}

fn policy_path() -> PathBuf {
    env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../training/output/best_policy.json")
    })
}

fn output_label(index: usize) -> &'static str {
    match index {
        0 => "turn",
        1 => "forward",
        2 => "strafe",
        3 => "fire",
        4 => "exit",
        5 => "pistol",
        6 => "shotgun",
        7 => "chaingun",
        8 => "rockets",
        9 => "run",
        _ => "action",
    }
}

fn input_group(index: usize) -> &'static str {
    match index {
        0..=11 => "wall rays",
        12..=21 => "nearest enemies",
        22..=30 => "nearest pickup",
        31..=33 => "exit direction",
        34..=35 => "health / armor",
        36..=39 => "ammo",
        40..=43 => "weapon unlocks",
        44..=45 => "weapon / time",
        46..=57 => "ray memory",
        _ => "action memory",
    }
}

fn input_group_label(index: usize) -> &'static str {
    match index {
        0 => "RAYS",
        12 => "ENEMIES",
        22 => "PICKUP",
        31 => "EXIT",
        34 => "VITALS",
        36 => "AMMO",
        40 => "UNLOCKS",
        44 => "STATE",
        46 => "RAY MEM",
        _ => "ACTION MEM",
    }
}

fn node_positions(
    shape: &[usize],
    rect: Rect,
    center: Pos2,
    zoom: f32,
    pan: Vec2,
) -> Vec<Vec<Pos2>> {
    let left = rect.left() + 74.0;
    let right = rect.right() - 74.0;
    let top = rect.top() + 74.0;
    let bottom = rect.bottom() - 42.0;
    let layer_gap = if shape.len() > 1 {
        (right - left) / (shape.len() - 1) as f32
    } else {
        0.0
    };
    shape
        .iter()
        .enumerate()
        .map(|(layer, &count)| {
            let x = left + layer as f32 * layer_gap;
            let spacing = (bottom - top) / count.max(1) as f32;
            (0..count)
                .map(|node| {
                    let base = Pos2::new(x, top + spacing * (node as f32 + 0.5));
                    center + (base - center) * zoom + pan
                })
                .collect()
        })
        .collect()
}

fn network_edges(policy: &Policy, nodes: &[Vec<Pos2>]) -> Vec<Edge> {
    let mut result = Vec::new();
    let mut offset = 0;
    for layer in 0..policy.shape.len() - 1 {
        let fan_in = policy.shape[layer];
        let fan_out = policy.shape[layer + 1];
        for to in 0..fan_out {
            for from in 0..fan_in {
                let weight = policy.weights[offset + to * (fan_in + 1) + from];
                result.push(Edge {
                    from_layer: layer,
                    from_node: from,
                    to_node: to,
                    weight,
                    a: nodes[layer][from],
                    b: nodes[layer + 1][to],
                });
            }
        }
        offset += fan_out * (fan_in + 1);
    }
    result
}

fn segment_distance(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let denom = ab.length_sq();
    if denom <= f32::EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(ab) / denom).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

fn bias_for(policy: &Policy, layer: usize, node: usize) -> Option<f32> {
    if layer == 0 || layer >= policy.shape.len() {
        return None;
    }
    let mut offset = 0;
    for l in 0..layer - 1 {
        offset += policy.shape[l + 1] * (policy.shape[l] + 1);
    }
    let fan_in = policy.shape[layer - 1];
    policy
        .weights
        .get(offset + node * (fan_in + 1) + fan_in)
        .copied()
}

impl eframe::App for Visualizer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_policy();
        ctx.request_repaint_after(Duration::from_millis(250));
        ctx.set_visuals(egui::Visuals::dark());

        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("DOOM-ML  /  POLICY MAP");
                ui.separator();
                ui.label("Live view of the best agent network");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Reset view").clicked() {
                        self.zoom = 1.0;
                        self.pan = Vec2::ZERO;
                        self.selected_node = None;
                    }
                    if ui.button("−").clicked() {
                        self.zoom_at(ui.min_rect().center(), ui.min_rect().center(), 0.85);
                    }
                    if ui.button("+").clicked() {
                        self.zoom_at(ui.min_rect().center(), ui.min_rect().center(), 1.18);
                    }
                    ui.label(format!("Zoom {:.0}%", self.zoom * 100.0));
                });
            });
        });

        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(284.0)
            .size_range(240.0..=420.0)
            .show(ui, |ui| {
                ui.add_space(8.0);
                if let Some(policy) = &self.policy {
                    ui.label(egui::RichText::new("BEST AGENT").small().color(MUTED));
                    ui.heading(format!("Generation {}", policy.generation + 1));
                    ui.label(format!("Difficulty  {}", policy.difficulty));
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("ARCHITECTURE").small().color(MUTED));
                    ui.label(
                        policy
                            .shape
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(" → "),
                    );
                    ui.label(format!("{} parameters", policy.weights.len()));
                    ui.separator();
                    ui.label(egui::RichText::new("WEIGHT KEY").small().color(MUTED));
                    ui.horizontal(|ui| {
                        ui.colored_label(POSITIVE, "● positive");
                        ui.colored_label(NEGATIVE, "● negative");
                    });
                    ui.label(
                        egui::RichText::new("Brightness shows relative magnitude.")
                            .small()
                            .color(MUTED),
                    );
                    ui.label(
                        egui::RichText::new("Neuron size subtly reflects signed bias.")
                            .small()
                            .color(MUTED),
                    );
                    ui.separator();
                    ui.label(egui::RichText::new("INSPECTOR").small().color(MUTED));
                    ui.label(
                        self.inspection
                            .as_deref()
                            .unwrap_or("Hover a neuron or connection in the canvas."),
                    );
                    ui.separator();
                    ui.label(egui::RichText::new("NAVIGATION").small().color(MUTED));
                    ui.label("Scroll: zoom at cursor");
                    ui.label("WASD / arrows: move the view");
                    ui.label("Drag: move the network");
                    ui.label("Click a node: isolate its connections");
                    if let Some(error) = &self.error {
                        ui.separator();
                        ui.colored_label(NEGATIVE, "Policy file status");
                        ui.label(egui::RichText::new(error).small().color(MUTED));
                    }
                } else {
                    ui.heading("Waiting for policy");
                    ui.label(format!("Watching {}", self.path.display()));
                    if let Some(error) = &self.error {
                        ui.colored_label(NEGATIVE, error);
                    }
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(10.0))
            .show(ui, |ui| {
                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
                let rect = response.rect;
                painter.rect_filled(rect, 10.0, PANEL);
                let center = rect.center();

                let mut pan_delta = Vec2::ZERO;
                let scroll = if response.hovered() {
                    ctx.input(|input| {
                        let dt = input.stable_dt.min(0.1);
                        let mut direction = Vec2::ZERO;
                        for (key, vector) in [
                            (egui::Key::A, Vec2::new(1.0, 0.0)),
                            (egui::Key::ArrowLeft, Vec2::new(1.0, 0.0)),
                            (egui::Key::D, Vec2::new(-1.0, 0.0)),
                            (egui::Key::ArrowRight, Vec2::new(-1.0, 0.0)),
                            (egui::Key::W, Vec2::new(0.0, 1.0)),
                            (egui::Key::ArrowUp, Vec2::new(0.0, 1.0)),
                            (egui::Key::S, Vec2::new(0.0, -1.0)),
                            (egui::Key::ArrowDown, Vec2::new(0.0, -1.0)),
                        ] {
                            if input.key_down(key) {
                                direction += vector;
                            }
                        }
                        if direction.length_sq() > 0.0 {
                            pan_delta = direction.normalized() * (420.0 * dt / self.zoom);
                        }
                        input.smooth_scroll_delta.y
                    })
                } else {
                    0.0
                };
                self.pan += pan_delta;
                if response.dragged() {
                    self.pan += response.drag_delta();
                }
                if scroll.abs() > 0.0 {
                    let pointer = response.hover_pos().unwrap_or(center);
                    self.zoom_at(pointer, center, (scroll * 0.002).exp());
                }

                if let Some(policy) = &self.policy {
                    let nodes = node_positions(&policy.shape, rect, center, self.zoom, self.pan);
                    let edges = network_edges(policy, &nodes);
                    let pointer = response.hover_pos();
                    let nearest_node = pointer.and_then(|pointer| {
                        nodes
                            .iter()
                            .enumerate()
                            .flat_map(|(layer, layer_nodes)| {
                                layer_nodes
                                    .iter()
                                    .enumerate()
                                    .map(move |(index, &pos)| (layer, index, pos.distance(pointer)))
                            })
                            .filter(|(_, _, distance)| *distance < 11.0)
                            .min_by(|a, b| a.2.total_cmp(&b.2))
                            .map(|(layer, node, _)| (layer, node))
                    });
                    if response.clicked() {
                        self.selected_node = nearest_node;
                    }
                    let max_weight = edges
                        .iter()
                        .map(|edge| edge.weight.abs())
                        .fold(0.0_f32, f32::max)
                        .max(0.001);
                    for (layer, &width) in policy.shape.iter().enumerate() {
                        let pos = nodes[layer][0];
                        let heading = if layer == 0 {
                            "INPUT"
                        } else if layer + 1 == policy.shape.len() {
                            "ACTIONS"
                        } else {
                            "HIDDEN"
                        };
                        painter.text(
                            pos + Vec2::new(-22.0, -48.0 * self.zoom),
                            egui::Align2::CENTER_CENTER,
                            format!("{heading}  {width}"),
                            egui::FontId::proportional(12.0),
                            MUTED,
                        );
                    }
                    for edge in &edges {
                        let strength = (edge.weight.abs() / max_weight).clamp(0.0, 1.0);
                        let emphasis = strength.powf(1.55);
                        let selected = self.selected_node.is_some_and(|(layer, node)| {
                            (edge.from_layer + 1 == layer && edge.to_node == node)
                                || (edge.from_layer == layer && edge.from_node == node)
                        });
                        let base = if edge.weight >= 0.0 {
                            POSITIVE
                        } else {
                            NEGATIVE
                        };
                        let alpha = if self.selected_node.is_some() {
                            if selected {
                                (90.0 + strength.powf(0.8) * 165.0) as u8
                            } else {
                                51
                            }
                        } else {
                            (5.0 + emphasis * 250.0) as u8
                        };
                        let color =
                            Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha);
                        let width = if selected {
                            1.8 + strength.powf(0.8) * 4.0
                        } else {
                            0.2 + emphasis * 3.2
                        };
                        painter.line_segment([edge.a, edge.b], Stroke::new(width, color));
                    }
                    let hovered_node = pointer.and_then(|pointer| {
                        nodes
                            .iter()
                            .enumerate()
                            .flat_map(|(layer, layer_nodes)| {
                                layer_nodes
                                    .iter()
                                    .enumerate()
                                    .map(move |(index, &pos)| (layer, index, pos.distance(pointer)))
                            })
                            .filter(|(_, _, distance)| *distance < 9.0)
                            .min_by(|a, b| a.2.total_cmp(&b.2))
                            .map(|(layer, index, _)| (layer, index))
                    });
                    let hovered_edge = pointer.and_then(|pointer| {
                        edges
                            .iter()
                            .map(|edge| (segment_distance(pointer, edge.a, edge.b), edge))
                            .filter(|(distance, _)| *distance < 5.0)
                            .min_by(|a, b| a.0.total_cmp(&b.0))
                            .map(|(_, edge)| *edge)
                    });

                    for (layer, layer_nodes) in nodes.iter().enumerate() {
                        let base_radius = if policy.shape[layer] > 32 {
                            3.0
                        } else if policy.shape[layer] > 16 {
                            4.0
                        } else {
                            5.0
                        };
                        for (index, &pos) in layer_nodes.iter().enumerate() {
                            let bias = bias_for(policy, layer, index).unwrap_or(0.0);
                            let radius = base_radius * (1.0 + 0.14 * bias.tanh());
                            let hovered = hovered_node == Some((layer, index));
                            let focused = self.selected_node == Some((layer, index));
                            let base_fill = if layer == 0 {
                                Color32::from_rgb(134, 175, 219)
                            } else if layer + 1 == policy.shape.len() {
                                POSITIVE
                            } else {
                                Color32::from_rgb(212, 222, 235)
                            };
                            let alpha = if self.selected_node.is_some() && !focused {
                                51
                            } else {
                                255
                            };
                            let fill = Color32::from_rgba_unmultiplied(
                                base_fill.r(),
                                base_fill.g(),
                                base_fill.b(),
                                alpha,
                            );
                            painter.circle_filled(
                                pos,
                                if hovered || focused {
                                    radius + 2.5
                                } else {
                                    radius
                                },
                                fill,
                            );
                            if focused {
                                painter.circle_stroke(
                                    pos,
                                    radius + 5.0,
                                    Stroke::new(2.0, Color32::WHITE),
                                );
                            }
                            if layer == 0
                                && policy.shape[layer] == 64
                                && [0, 12, 22, 31, 34, 36, 40, 44, 46, 58].contains(&index)
                            {
                                painter.text(
                                    pos + Vec2::new(-14.0, 0.0),
                                    egui::Align2::RIGHT_CENTER,
                                    input_group_label(index),
                                    egui::FontId::proportional(9.0),
                                    MUTED,
                                );
                            }
                            if layer + 1 == policy.shape.len() {
                                painter.text(
                                    pos + Vec2::new(12.0, 0.0),
                                    egui::Align2::LEFT_CENTER,
                                    output_label(index),
                                    egui::FontId::proportional(11.0),
                                    if alpha == 255 { TEXT } else { MUTED },
                                );
                            }
                        }
                    }

                    if let Some((layer, index)) = hovered_node {
                        let detail = if layer == 0 {
                            format!("Input {index} · {}", input_group(index))
                        } else if layer + 1 == policy.shape.len() {
                            format!(
                                "Action {index} · {} · bias {:+.4}",
                                output_label(index),
                                bias_for(policy, layer, index).unwrap_or(0.0)
                            )
                        } else {
                            format!(
                                "Hidden layer {layer} · neuron {index} · bias {:+.4}",
                                bias_for(policy, layer, index).unwrap_or(0.0)
                            )
                        };
                        ui.painter().text(
                            rect.left_top() + Vec2::new(18.0, 18.0),
                            egui::Align2::LEFT_TOP,
                            &detail,
                            egui::FontId::proportional(13.0),
                            TEXT,
                        );
                        self.inspection = Some(detail);
                    } else if let Some(edge) = hovered_edge {
                        let detail = format!(
                            "Layer {} neuron {} → layer {} neuron {}   weight {:+.5}",
                            edge.from_layer,
                            edge.from_node,
                            edge.from_layer + 1,
                            edge.to_node,
                            edge.weight
                        );
                        ui.painter().text(
                            rect.left_top() + Vec2::new(18.0, 18.0),
                            egui::Align2::LEFT_TOP,
                            &detail,
                            egui::FontId::proportional(13.0),
                            if edge.weight >= 0.0 {
                                POSITIVE
                            } else {
                                NEGATIVE
                            },
                        );
                        self.inspection = Some(detail);
                    } else if let Some((layer, node)) = self.selected_node {
                        let detail = if layer == 0 {
                            format!("Selected input {node} · {}", input_group(node))
                        } else if layer + 1 == policy.shape.len() {
                            format!("Selected action {node} · {}", output_label(node))
                        } else {
                            format!(
                                "Selected layer {layer} neuron {node} · bias {:+.4}",
                                bias_for(policy, layer, node).unwrap_or(0.0)
                            )
                        };
                        self.inspection = Some(detail);
                    } else {
                        self.inspection = None;
                    }
                } else {
                    painter.text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        "Waiting for a valid best_policy.json",
                        egui::FontId::proportional(20.0),
                        TEXT,
                    );
                }
            });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Doom-ML Network Visualizer")
            .with_inner_size([1500.0, 940.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Doom-ML Network Visualizer",
        options,
        Box::new(|_creation_context| Ok(Box::new(Visualizer::new(policy_path())))),
    )
}
