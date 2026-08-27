"""
ArmanNN OpenAI-Compatible API Server

Serves the trained ArmanNN model with an API compatible with OpenAI's format.
Works as a drop-in replacement for OpenAI API calls — just change the base_url.

Usage:
    python serve.py --checkpoint checkpoints/step_022000.pt --port 8000

    # Then use it like OpenAI:
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "arman-nn", "prompt": "The meaning of life is", "max_tokens": 100}'

    # Or with the OpenAI Python client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    response = client.completions.create(model="arman-nn", prompt="Hello", max_tokens=50)
"""

import argparse
import time
import uuid
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from arman.model import ArmanConfig, ArmanNN
from generate import generate as generate_tokens


# ============================================================
# REQUEST / RESPONSE MODELS (OpenAI-compatible schema)
# ============================================================

class CompletionRequest(BaseModel):
    model: str = "arman-nn"
    prompt: str | list[str] = ""
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=4)
    stream: bool = False  # Streaming not implemented yet


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str = "length"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "arman-nn"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=4)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "length"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "arman-nn"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ============================================================
# GLOBALS (loaded at startup)
# ============================================================
model: ArmanNN | None = None
tokenizer = None
device = None
model_name = "arman-nn"


# ============================================================
# MODEL LOADING
# ============================================================
def load_model(checkpoint_path: str, device_name: str = "auto") -> tuple[ArmanNN, str]:
    global model, tokenizer, device

    # Device
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = state["config"]
    config = ArmanConfig(**config_dict)
    model = ArmanNN(config)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded | params={model.parameter_count():,} | device={device}")
    print(f"Vocab size: {config.vocab_size} | Max seq len: {config.max_seq_len}")
    return model, device


# ============================================================
# GENERATION HELPER
# ============================================================
@torch.no_grad()
def run_generation(
    prompt_text: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop: list[str] | None = None,
) -> tuple[str, int, int]:
    """Generate text and return (generated_text, prompt_tokens, completion_tokens)."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # Clamp to max_seq_len
    max_seq = model.config.max_seq_len
    if prompt_len >= max_seq:
        input_ids = input_ids[:, -max_seq + 1:]
        prompt_len = input_ids.shape[1]

    # Cap generation to not exceed model's max_seq_len
    max_new = min(max_tokens, max_seq - prompt_len)
    if max_new <= 0:
        return "", prompt_len, 0

    # Generate
    output_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new,
        temperature=max(temperature, 1e-7),  # Avoid division by zero
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Decode only the new tokens
    new_tokens = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Handle stop sequences
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        for s in stop:
            idx = generated_text.find(s)
            if idx != -1:
                generated_text = generated_text[:idx]
                break

    completion_tokens = len(new_tokens)
    return generated_text, prompt_len, completion_tokens


# ============================================================
# FASTAPI APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — model loaded via CLI args before uvicorn starts
    yield
    # Shutdown
    pass


app = FastAPI(
    title="ArmanNN API",
    description="OpenAI-compatible API for ArmanNN language model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[
        ModelInfo(id=model_name, created=int(time.time()))
    ])


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    if model_id != model_name:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelInfo(id=model_name, created=int(time.time()))


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle single or batch prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, prompt in enumerate(prompts):
        for n in range(request.n):
            generated_text, prompt_tokens, completion_tokens = run_generation(
                prompt_text=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            )
            choices.append(CompletionChoice(
                index=len(choices),
                text=generated_text,
                finish_reason="stop" if request.stop and any(s in generated_text for s in (request.stop if isinstance(request.stop, list) else [request.stop])) else "length",
            ))
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model_name,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert chat messages to a single prompt
    # Simple format: "role: content\n" for each message
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"{msg.content}\n\n")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}\n")
    prompt_parts.append("Assistant:")
    prompt_text = "".join(prompt_parts)

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for n in range(request.n):
        generated_text, prompt_tokens, completion_tokens = run_generation(
            prompt_text=prompt_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
        )
        choices.append(ChatCompletionChoice(
            index=n,
            message=ChatMessage(role="assistant", content=generated_text.strip()),
            finish_reason="stop" if request.stop else "length",
        ))
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model_name,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="ArmanNN OpenAI-compatible API Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model_name", type=str, default="arman-nn", help="Model name in API responses")
    args = parser.parse_args()

    model_name = args.model_name
    load_model(args.checkpoint, args.device)

    print(f"\nStarting ArmanNN API server at http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1")
    print(f"\nExample usage:")
    print(f"  curl http://localhost:{args.port}/v1/completions \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\": \"{model_name}\", \"prompt\": \"Hello world\", \"max_tokens\": 50}}'")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
