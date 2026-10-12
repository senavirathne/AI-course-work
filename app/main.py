"""Local inspection web application with a trained artifact required for inference."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from fabric_inspector.data.images import MAX_UPLOAD_BYTES
from fabric_inspector.serving.predictor import Predictor


def create_app(bundle_path=None, allow_development=False):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor, app.state.load_error = None, None
        try:
            app.state.predictor = Predictor(
                Path(bundle_path or os.getenv("FABRIC_BUNDLE", "artifacts/serving-bundle.json")),
                device=os.getenv("FABRIC_DEVICE", "cpu"),
                allow_development=allow_development or os.getenv("FABRIC_ALLOW_DEVELOPMENT") == "1",
            )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
            app.state.load_error = str(error)
        yield

    app = FastAPI(title="Fabric Inspector", version="0.3.0", lifespan=lifespan)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/health")
    def health():
        predictor = app.state.predictor
        return {
            "ready": predictor is not None,
            "load_error": app.state.load_error,
            "model_version": predictor.version if predictor else None,
            "provenance": predictor.bundle["provenance"] if predictor else None,
            "input_contract": "256x256 grayscale/RGB PNG or JPEG, max 5 MiB",
            "ood_detection": "Not implemented or claimed",
        }

    async def infer(file: UploadFile, explain=False, alpha=0.55):
        if app.state.predictor is None:
            raise HTTPException(
                503, "Trained model unavailable. Check /health and the model bundle."
            )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 5 MiB limit")
        try:
            return await run_in_threadpool(app.state.predictor.predict, content, explain, alpha)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        return await infer(file)

    @app.post("/explain")
    async def explain(file: UploadFile = File(...), alpha: float = Form(0.55)):
        if not 0 <= alpha <= 1:
            raise HTTPException(422, "Opacity must be between zero and one")
        return await infer(file, True, alpha)

    return app


app = create_app()
