# scnn_torch

Now the data flow is clean and simple:
```
Input (B, 3, 288, 800)
    │
    ▼
backbone ──────────────── (B, 512, 36, 100)
    │
    ▼
scnn_neck ─────────────── (B, 128, 36, 100)
    │
    ▼
message_passing ───────── (B, 128, 36, 100)
    │
    ▼
seg_neck ──────────────── (B, 5, 36, 100)    # shared features
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
seg_head                          exist_head
    │                                  │
    ▼                                  ▼
seg_pred ───── (B,5,288,800)      exist_pred ───── (B, 4)
```
