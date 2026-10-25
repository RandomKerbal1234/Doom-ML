#!/bin/bash

cd doom

build=false
run=false
help=false

G='\033[0;32m'
Y='\033[0;33m'
R='\033[0m'

h_message="
${G}Usage:${Y} $0 [OPTIONS]${R}

${G}Options:
   ${Y}-b
   	${R}Build DOOM executable with [cargo build -r]
   ${Y}-r
   	${R}Run DOOM executable with [cargo run -r]
   ${Y}-h
   	${R}Display this message\n"

if [ $# -eq 0 ]; then
	printf "$h_message"
fi

while getopts "brh" opt; do
	case "$opt" in 
		b) build=true ;;
		r) run=true ;;
		h) help=true 
			printf "$h_message" ;;
		\?) help=true 
			printf "$h_message" ;;
	esac
done

executable="target/release/doom" # Remember that this script is running in Doom-ML/doom/


if "$build"; then
	cargo build -r
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
