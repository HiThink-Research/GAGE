# Retro Remote Game (WebSocket Stream)

Run a stable-retro game headless inside the container and stream it to Chrome via WebSocket JPEG frames.

## Requirements

Install these optional dependencies in the container (or your active venv):

- fastapi
- uvicorn
- numpy
- pillow

You also need a stable-retro ROM imported via:

```
python -m retro.import
```

## Run

```
PYTHONPATH=src python -m gage_eval.role.arena.games.retro.webrtc_server \
  --game SuperMarioBros3-Nes-v0 \
  --port 5800
```

Then open:

```
http://localhost:5800
```

## Controls

- Arrows: move
- Z: jump
- X: run
- Enter: start
- Shift: select

Click the video once to ensure it has focus.

## Notes

- The server streams JPEG frames directly from `StableRetroArenaEnvironment` in headless mode.
- If you see a black screen, confirm the ROM is imported and the game id is correct.
- The WebSocket server is standalone and does not require a PipelineConfig.
