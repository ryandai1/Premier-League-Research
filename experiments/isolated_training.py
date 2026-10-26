"""Isolated-transfer trainer, retaining existing update and stopping rules.

Logging runs after optimization/evaluation and consumes no random numbers.
Train loss is the online, dropout-enabled minibatch average; validation uses eval mode.
"""

import copy, json, random, time, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from experiments.train import CONFIG, CLASSES, network, digest


def state_digest(model):
    digestor = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digestor.update(name.encode())
        digestor.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digestor.hexdigest()


def fit(name, train, val, folder, features, initial=None, freeze_hidden=False):
    started = time.time()
    print(
        f'\nSTART {name}: seed={CONFIG["seed"]}, initial={initial}, rows={len(train[1])}',
        flush=True,
    )
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    x, y = train
    xv, yv = val
    model = network(x.shape[1])
    scaler = StandardScaler()
    if initial:
        saved = torch.load(initial, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        scaler.mean_ = np.array(saved["scaler_mean"])
        scaler.scale_ = np.array(saved["scaler_scale"])
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(features)
    else:
        scaler.fit(x)
    if freeze_hidden and initial is None:
        raise ValueError("Frozen hidden layers require a pretrained checkpoint")
    if freeze_hidden:
        for layer in (model[0], model[3]):
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
    frozen_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    initial_hash = state_digest(model)
    audit = {
        "model": name,
        "freeze_hidden": freeze_hidden,
        "trainable_parameters": [
            n for n, p in model.named_parameters() if p.requires_grad
        ],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "initial_checkpoint": str(initial) if initial else None,
        "initial_weights_sha256": initial_hash,
        "initial_checkpoint_sha256": digest(initial) if initial else None,
        "training_rows": len(x),
        "validation_rows": len(yv),
        "optimizer_steps": 0,
    }
    (folder / f"{name}_execution.json").write_text(json.dumps(audit, indent=2))
    xt = torch.tensor(scaler.transform(x), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    vt = torch.tensor(scaler.transform(xv), dtype=torch.float32)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    wait = 0
    history = []
    best_epoch = 0
    # Fine-tuning may retain its initial checkpoint (epoch 0) if it never improves.
    if initial:
        model.eval()
        with torch.no_grad():
            best = float(criterion(model(vt), torch.tensor(yv, dtype=torch.long)))

    def save(epoch, loss):
        torch.save(
            {
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "validation_log_loss": loss,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "features": features,
                "class_order": CLASSES,
                "config": CONFIG,
                "training_rows": len(x),
                "initial_checkpoint": str(initial) if initial else None,
            },
            folder / f"{name}_best.pt",
        )

    if initial:
        save(0, best)
        print(f"{name} epoch=000 initial_validation_loss={best:.9f}", flush=True)
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_started = time.time()
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), CONFIG["batch_size"]):
            ids = order[start : start + CONFIG["batch_size"]]
            opt.zero_grad()
            loss = criterion(model(xt[ids]), yt[ids])
            loss.backward()
            opt.step()
            audit["optimizer_steps"] += 1
            total += float(loss.detach()) * len(ids)
        for n, p in model.named_parameters():
            if n in frozen_before:
                assert p.grad is None and torch.equal(p.detach(), frozen_before[n]), n
        model.eval()
        with torch.no_grad():
            logits = model(vt)
            vl = float(criterion(logits, torch.tensor(yv, dtype=torch.long)))
            va = float((logits.argmax(1) == torch.tensor(yv)).float().mean())
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": total / len(xt),
                "validation_log_loss": vl,
            }
        )
        # Save the true minimum; patience resets only on a material improvement.
        material = vl < best - CONFIG["min_delta"]
        checkpoint_saved = vl < best
        if checkpoint_saved:
            best = vl
            best_epoch = epoch
            save(epoch, vl)
        wait = 0 if material else wait + 1
        history[-1].update(
            {
                "validation_accuracy": va,
                "best_validation_loss": best,
                "best_epoch_so_far": best_epoch,
                "checkpoint_saved": checkpoint_saved,
                "patience_used": wait,
                "optimizer_steps": audit["optimizer_steps"],
                "epoch_seconds": time.time() - epoch_started,
            }
        )
        pd.DataFrame(history).to_csv(folder / f"{name}_history.csv", index=False)
        line = f'{name} epoch={epoch:03d} train_loss={total/len(xt):.9f} val_loss={vl:.9f} val_acc={va:.6f} best_epoch={best_epoch:03d} saved={checkpoint_saved} patience={wait}/20 steps={audit["optimizer_steps"]}'
        print(line, flush=True)
        with (folder / "epoch_log.txt").open("a") as log:
            log.write(line + "\n")
        if wait >= CONFIG["patience"]:
            break
    pd.DataFrame(history).to_csv(folder / f"{name}_history.csv", index=False)
    frozen = torch.load(
        folder / f"{name}_best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(frozen["model_state"])
    for n, p in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p.detach(), frozen_before[n]), n
    audit["frozen_hidden_layers_unchanged"] = bool(freeze_hidden)
    model.eval()
    audit.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "best_weights_sha256": state_digest(model),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "stop_reason": (
                "patience" if wait >= CONFIG["patience"] else "maximum epochs"
            ),
        }
    )
    (folder / f"{name}_execution.json").write_text(json.dumps(audit, indent=2))
    print(
        f'END {name}: best_epoch={best_epoch}, epochs={len(history)}, stop={audit["stop_reason"]}',
        flush=True,
    )
    for p in model.parameters():
        p.requires_grad_(False)

    def predict(a):
        with torch.no_grad():
            return (
                model(torch.tensor(scaler.transform(a), dtype=torch.float32))
                .softmax(1)
                .numpy()
            )

    assert best_epoch <= 100 and len(history) <= 100
    return predict, {
        "model": name,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_log_loss": best,
        "training_rows": len(x),
        "checkpoint_sha256": digest(folder / f"{name}_best.pt"),
    }
