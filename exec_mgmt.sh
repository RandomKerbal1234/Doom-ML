#!/bin/bash

cd doom

build=false
run=false
help=false
vis=false

G='\033[0;32m'
Y='\033[0;33m'
R='\033[0m'

h_message="
${G}Usage:${Y} $0 ${R}[OPTIONS]

${G}Options:
   ${Y}-b
   	${R}Build DOOM executable with [cargo build -r]
   ${Y}-r
   	${R}Run DOOM executable with [cargo run -r]
   ${Y}-h
   	${R}Display this message\n
   ${Y}-v
    ${R}Run Neural Network Visualizer"

if [ $# -eq 0 ]; then
	printf "$h_message"
fi

while getopts "brhv" opt; do
	case "$opt" in 
		b) build=true ;;
		r) run=true ;;
		v) vis=true ;;
		h) help=true 
			printf "$h_message" ;;
		\?) help=true 
			printf "$h_message" ;;
	esac
done

executable="target/release/doom" # Remember that this script is running in Doom-ML/doom/
vis_exec="../visualizer/target/release/doom-network-visualizer"


if "$build"; then
	cargo build -r --bin doom
	cargo build -r --bin doom_headless --no-default-features
fi

if "$vis"; then
	if [ -f "$vis_exec" ]; then
		$vis_exec
	else
		cd ../visualizer
		cargo r -r
	fi
fi
	

if "$run"; then
	if [ -f "$executable" ]; then
		$executable
	else
		cargo run -r
	fi
fi

h1_message="
${G}Usage:${Y} $0 [OPTIONS]${R}

${G}Options:
   ${Y}-b
   	${R}Build DOOM executable with [cargo build -r]
   ${Y}-r
   	${R}Run DOOM executable with [cargo run -r]
   ${Y}-h
   	${R}Display this message"
