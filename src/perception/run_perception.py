import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.receiver_app import PerceptionReceiverApp, PerceptionReceiverConfig


def main() -> None:
    app = PerceptionReceiverApp(PerceptionReceiverConfig())
    app.run()


if __name__ == "__main__":
    main()
