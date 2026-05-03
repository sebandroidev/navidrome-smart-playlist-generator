import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── app lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from config import get_config
    from state.db import StateDB
    import api.router as api_router
    import scheduler

    cfg = get_config()
    db  = StateDB(cfg.state_db_path)

    api_router.init(cfg, db)
    scheduler.start(cfg, db)

    port = int(os.environ.get("PORT", 7070))
    import notifier
    notifier.start_bot_polling(cfg, port)

    log.info("Orly Jams started — library: %s  nav: %s",
             cfg.beets.db_path, cfg.navidrome.url)

    yield

    scheduler.stop()
    log.info("Orly Jams stopped")


app = FastAPI(
    title="Orly Jams",
    description="Self-hosted smart playlist generator for Navidrome",
    version="1.0.0",
    lifespan=lifespan,
)

from api.router import router
app.include_router(router)

# Serve the web dashboard at /
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/")
    def dashboard():
        return FileResponse(str(_static_dir / "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7070))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        loop="asyncio",   # avoid uvloop — saves ~15 MB on startup
        http="h11",       # lighter HTTP parser than httptools
        access_log=False, # reduce log overhead
    )
