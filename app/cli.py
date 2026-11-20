"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-game-dialogue-simulator", description="Game developers use this to build a small, quantized GPT for in-game dialogue, then simulate its memory footprint and generation speed for resource-constrained devices.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("corpus", help="prepare and inspect dialogue corpus (milestone 1)")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Quantized Game Dialogue Simulator is ready. Open README.md for your milestones.")
    elif args.command == "corpus":
        from app.core import milestone_1
        data = milestone_1()
        print(f"Corpus prepared. Characters: {len(data['corpus'])}, Vocab size: {data['vocab_size']}, Tokens: {len(data['tokens'])}")


if __name__ == "__main__":
    main()
