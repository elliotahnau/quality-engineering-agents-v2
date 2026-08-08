"""Run the SUT: python -m sut [--port 8000]"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdOps Campaign API (SUT)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("sut.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
