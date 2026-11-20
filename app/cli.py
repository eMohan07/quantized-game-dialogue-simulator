"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-game-dialogue-simulator", description="Game developers use this to build a small, quantized GPT for in-game dialogue, then simulate its memory footprint and generation speed for resource-constrained devices.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("corpus", help="prepare and inspect dialogue corpus (milestone 1)")
    sub.add_parser("train", help="train small GPT on dialogue corpus (milestone 2)")
    gen_parser = sub.add_parser("generate", help="generate sample dialogue (milestone 3)")
    gen_parser.add_argument("--prompt", default="[Hero] ", help="starting dialogue prompt")
    sub.add_parser("quantize", help="apply dynamic and static PTQ (milestone 4)")
    sub.add_parser("compare-sizes", help="measure and compare model sizes (milestone 5)")
    sub.add_parser("simulate-speed", help="simulate inference speed on resource-constrained devices (milestone 6)")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Quantized Game Dialogue Simulator is ready. Open README.md for your milestones.")
    elif args.command == "corpus":
        from app.core import milestone_1
        data = milestone_1()
        print(f"Corpus prepared. Characters: {len(data['corpus'])}, Vocab size: {data['vocab_size']}, Tokens: {len(data['tokens'])}")
    elif args.command == "train":
        from app.core import milestone_2
        res = milestone_2(steps=60)
        print(f"Model trained! Initial loss: {res['initial_loss']:.4f}, Final loss: {res['final_loss']:.4f}")
    elif args.command == "generate":
        from app.core import milestone_3
        res = milestone_3(prompts=[args.prompt], max_new_tokens=40)
        print(f"Generated dialogue:\n{res['generated_samples'][args.prompt]}")
    elif args.command == "quantize":
        from app.core import milestone_4
        res = milestone_4()
        print("Dynamic and static post-training quantization applied successfully!")
    elif args.command == "compare-sizes":
        from app.core import milestone_5
        res = milestone_5()
        print(f"Original Model Size: {res['original_size_kb']} KB")
        print(f"Dynamic Model Size:  {res['dynamic_size_kb']} KB (Reduction: {res['reduction_dynamic_pct']}%, Compression: {res['compression_ratio_dynamic']}x)")
        print(f"Static Model Size:   {res['static_size_kb']} KB (Reduction: {res['reduction_static_pct']}%, Compression: {res['compression_ratio_static']}x)")
    elif args.command == "simulate-speed":
        from app.core import milestone_6
        res = milestone_6(num_steps=20)
        print("=== Simulated Inference Speed Report ===")
        print(f"Original: {res['original']['latency_ms_per_step']} ms/step, {res['original']['steps_per_sec']} steps/s")
        print(f"Dynamic:  {res['dynamic']['latency_ms_per_step']} ms/step, {res['dynamic']['steps_per_sec']} steps/s (Simulated Device Speedup: {res['dynamic']['simulated_bandwidth_speedup']}x)")
        print(f"Static:   {res['static']['latency_ms_per_step']} ms/step, {res['static']['steps_per_sec']} steps/s (Simulated Device Speedup: {res['static']['simulated_bandwidth_speedup']}x)")


if __name__ == "__main__":
    main()
