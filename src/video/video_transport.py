from __future__ import annotations

from visiongraph_ndi.NDIVideoOutput import NDIVideoOutput
from visiongraph_ndi.NDIVideoInput import NDIVideoInput


class VideoStreamSender:
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self._output: NDIVideoOutput | None = None

    def __enter__(self) -> "VideoStreamSender":
        self._output = NDIVideoOutput(self.stream_name).__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._output is not None:
            self._output.__exit__(exc_type, exc, tb)

    def send(self, frame) -> None:
        if self._output is None:
            raise RuntimeError("VideoStreamSender wurde nicht geöffnet.")
        self._output.send(frame)


class VideoStreamReceiver:
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self._input: NDIVideoInput | None = None

    @staticmethod
    def find_sources(timeout: float = 5.0):
        return NDIVideoInput.find_sources(timeout=timeout)

    def __enter__(self) -> "VideoStreamReceiver":
        self._input = NDIVideoInput(stream_name=self.stream_name).__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._input is not None:
            self._input.__exit__(exc_type, exc, tb)

    @property
    def is_connected(self) -> bool:
        return bool(self._input and self._input.is_connected)

    def read(self):
        if self._input is None:
            raise RuntimeError("VideoStreamReceiver wurde nicht geöffnet.")
        return self._input.read()
