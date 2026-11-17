# Doom-ML Network Visualizer

A native Rust app built with `eframe` and `egui` that maps the latest best-agent
policy as a layered `tanh` network. Positive weights are cyan, negative weights
are coral, and line brightness indicates relative weight magnitude. Hover a
node to identify its input, action, or bias; hover a connection to inspect its
weight.
Neuron size also subtly reflects signed bias: positive bias makes it slightly
larger, while negative bias makes it slightly smaller.

The visualizer watches `training/output/best_policy.json` and refreshes when the
trainer writes a new best policy. It displays the generation, difficulty,
network shape, and parameter count.

## Navigation

- Scroll over the network to zoom in or out, anchored at the pointer. Zoom ranges
  from 22% to 5000%.
- Press `W` / `A` / `S` / `D` or the arrow keys to move the view up, left, down,
  or right while the pointer is over the network.
- Click and drag the network to pan directly.
- Use the `+`, `−`, and `Reset view` buttons in the toolbar to adjust the view.

## Inspecting connections

- Click a neuron to highlight every connection entering or leaving it. The
  selected neuron and its connections stay bright while other nodes and
  connections fade to about 20% opacity. Selected line brightness and thickness
  still vary with each connection's weight magnitude.
- Click another neuron to inspect it, or click blank canvas space to clear the
  selection.
- Hover over a connection to see its weight. Hover over a neuron to see its
  input/action label or bias.

## Run

From this directory:

```sh
cargo run --release
```

By default it reads the policy file in the project’s `training/output`
directory. Pass another policy path as the first argument to inspect a
separate file:

```sh
cargo run --release -- /path/to/best_policy.json
```
