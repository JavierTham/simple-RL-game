"""
FastAPI server with WebSocket for real-time training updates and match streaming.
"""
import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from trainer import Trainer

app = FastAPI(title="Arcane Arena")
trainer = Trainer()

SAVED_BOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_bots')
os.makedirs(SAVED_BOTS_DIR, exist_ok=True)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')


# ── helpers ──────────────────────────────────────────────────
async def _stream_frames(ws: WebSocket, result: dict, speed: float = 1.0):
    """Send recorded match frames to the client at ~60 fps."""
    frames = result.get('frames', [])
    delay = (1 / 60) / speed
    try:
        for frame in frames:
            await ws.send_json({'type': 'match_frame', **frame})
            await asyncio.sleep(delay)
        await ws.send_json({
            'type': 'match_result',
            'winner': result['winner'],
            'steps': result['steps'],
        })
    except Exception:
        pass  # client disconnected


async def _run_training(ws: WebSocket, config: dict):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def train_fn():
        def cb(stats):
            loop.call_soon_threadsafe(queue.put_nowait, ('progress', stats))
        weights = trainer.train(config, cb)
        loop.call_soon_threadsafe(queue.put_nowait, ('done', weights))

    loop.run_in_executor(None, train_fn)

    try:
        while True:
            kind, payload = await queue.get()
            if kind == 'progress':
                await ws.send_json({'type': 'training_progress', **payload})
            elif kind == 'done':
                await ws.send_json({'type': 'training_complete', 'weights': payload})
                break
    except Exception:
        trainer.is_training = False  # stop training if client disconnected


# ── WebSocket endpoint ───────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            t = data.get('type')

            if t == 'start_training':
                asyncio.create_task(_run_training(ws, data.get('config', {})))

            elif t == 'stop_training':
                trainer.is_training = False

            elif t == 'test_match':
                loop = asyncio.get_event_loop()
                rw = data.get('reward_weights')
                result = await loop.run_in_executor(None, lambda: trainer.test_match(rw))
                if result:
                    speed = data.get('speed', 1.0)
                    asyncio.create_task(_stream_frames(ws, result, speed))
                else:
                    await ws.send_json({'type': 'error', 'msg': 'No trained bot yet. Train first!'})

            elif t == 'pvp_match':
                loop = asyncio.get_event_loop()
                w1 = data.get('bot1_weights')
                w2 = data.get('bot2_weights')
                result = await loop.run_in_executor(None, lambda: trainer.run_match(w1, w2))
                speed = data.get('speed', 1.0)
                asyncio.create_task(_stream_frames(ws, result, speed))

            elif t == 'save_bot':
                name = data.get('name', 'unnamed')
                weights = data.get('weights')
                if weights is None and trainer.agent:
                    weights = trainer.agent.get_weights()
                if weights:
                    fp = os.path.join(SAVED_BOTS_DIR, f"{name}.json")
                    with open(fp, 'w') as f:
                        json.dump({'name': name, 'weights': weights}, f)
                    await ws.send_json({'type': 'bot_saved', 'name': name})

            elif t == 'list_bots':
                bots = [f[:-5] for f in os.listdir(SAVED_BOTS_DIR) if f.endswith('.json')]
                await ws.send_json({'type': 'bot_list', 'bots': bots})

            elif t == 'load_bot':
                name = data.get('name')
                slot = data.get('slot', 1)
                fp = os.path.join(SAVED_BOTS_DIR, f"{name}.json")
                if os.path.exists(fp):
                    with open(fp) as f:
                        bd = json.load(f)
                    await ws.send_json({'type': 'bot_loaded', 'data': bd, 'slot': slot})
                else:
                    await ws.send_json({'type': 'error', 'msg': f'Bot "{name}" not found'})

    except WebSocketDisconnect:
        pass


# ── static file serving ─────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
