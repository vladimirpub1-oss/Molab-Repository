# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "torch", "psutil"]
# ///
import marimo
app = marimo.App(width="medium")

@app.cell
def _():
    import io, json, time, os, platform, shutil, subprocess
    import psutil
    import torch
    from torch import nn
    import marimo as mo
    return io, json, time, os, platform, shutil, subprocess, psutil, torch, nn, mo

@app.cell
def _(io, json, time, os, platform, shutil, subprocess, psutil, torch, nn, mo):
    torch.set_num_threads(2)
    report = {
        "test": "CEMP_TRAIN_001_MOLAB",
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "ram_total_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "disk_free_gib": round(shutil.disk_usage(".").free / 2**30, 2),
        "cuda_available": torch.cuda.is_available(), "runs": [],
    }
    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        report["nvidia_smi"] = "unavailable"
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    def make_model():
        return nn.Sequential(nn.Embedding(256,128), nn.Linear(128,512),
                             nn.ReLU(), nn.Linear(512,256))
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        torch.manual_seed(123)
        model = make_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.randint(0,256,(16,128),device=device)
        y = (x+1)%256
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x).reshape(-1,256),y.reshape(-1))
            loss.backward()
            optimizer.step()
            return loss.detach()
        for _ in range(3):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(20):
            loss = step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        report["runs"].append({
            "device":device, "steps":20, "seconds":elapsed,
            "tokens_per_second":20*16*128/elapsed, "loss":float(loss.cpu()),
            "parameters":sum(p.numel() for p in model.parameters()),
            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if device=="cuda" else None})
    model.eval()
    with torch.no_grad():
        before = model(x).clone()
    buffer = io.BytesIO()
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "step":23},buffer)
    checkpoint_bytes = buffer.tell()
    buffer.seek(0)
    state = torch.load(buffer,map_location=device,weights_only=True)
    restored = make_model().to(device)
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(),lr=0.001)
    restored_optimizer.load_state_dict(state["optimizer"])
    restored.eval()
    with torch.no_grad():
        error = (before-restored(x)).abs().max().item()
    report["checkpoint"] = {
        "device":device,"bytes":checkpoint_bytes,"max_output_error":error,
        "optimizer_state_entries":len(restored_optimizer.state),"step":state["step"]}
    assert error == 0, "Checkpoint mismatch"
    report["note"] = "Same 230144-parameter model, 2 CPU threads, batch 16x128, FP32 AdamW, 3 warmup and 20 timed steps as Colab. Synthetic throughput only. Checkpoint roundtrip in memory; no disk persistence or resumed-training test."
    report_text = json.dumps(report,indent=2)
    print("CEMP_TRAIN_BEGIN\n"+report_text+"\nCEMP_TRAIN_END")
    mo.md("# Molab compute test\n\n```json\n"+report_text+"\n```")
    return

if __name__ == "__main__":
    app.run()
