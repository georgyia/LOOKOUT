import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lookout",
        description="Local-first meeting gaze analysis.",
    )
    parser.add_argument("command", choices=["analyze"])
    parser.add_argument("video")
    args = parser.parse_args()

    if args.command == "analyze":
        print(f"LOOKOUT: analysis pipeline not implemented yet: {args.video}")
