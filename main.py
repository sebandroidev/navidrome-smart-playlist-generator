import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

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
