# CentralizedQNetwork Architecture

```mermaid
graph TD
    Input["Global State<br/>(obs_dim)"]

    subgraph Body["Shared Body (× n_body_layers)"]
        direction TB
        D1["Dense(hidden_dim)"]
        LN1{"LayerNorm?"}
        R1["ReLU"]
        D2["Dense(hidden_dim)"]
        LN2{"LayerNorm?"}
        R2["ReLU"]
        Dots1["⋮"]
        DN["Dense(hidden_dim)"]
        LNN{"LayerNorm?"}
        RN["ReLU"]

        D1 --> LN1 --> R1 --> D2 --> LN2 --> R2 --> Dots1 --> DN --> LNN --> RN
    end

    Input --> D1

    RN --> H0_Start & H1_Start & HN_Start

    subgraph Head0["Head · Agent 0"]
        direction TB
        H0_Start["Dense(hidden_dim / 2)"]
        H0_LN{"LayerNorm?"}
        H0_R["ReLU"]
        H0_Dots["⋮ × (n_head_layers − 1)"]
        H0_Out["Dense(actions_per_agent)"]

        H0_Start --> H0_LN --> H0_R --> H0_Dots --> H0_Out
    end

    subgraph Head1["Head · Agent 1"]
        direction TB
        H1_Start["Dense(hidden_dim / 2)"]
        H1_LN{"LayerNorm?"}
        H1_R["ReLU"]
        H1_Dots["⋮ × (n_head_layers − 1)"]
        H1_Out["Dense(actions_per_agent)"]

        H1_Start --> H1_LN --> H1_R --> H1_Dots --> H1_Out
    end

    subgraph HeadN["Head · Agent N"]
        direction TB
        HN_Start["Dense(hidden_dim / 2)"]
        HN_LN{"LayerNorm?"}
        HN_R["ReLU"]
        HN_Dots["⋮ × (n_head_layers − 1)"]
        HN_Out["Dense(actions_per_agent)"]

        HN_Start --> HN_LN --> HN_R --> HN_Dots --> HN_Out
    end

    H0_Out --> Q0["Q-values<br/>Agent 0"]
    H1_Out --> Q1["Q-values<br/>Agent 1"]
    HN_Out --> QN["Q-values<br/>Agent N"]

    Q0 & Q1 & QN --> Stack["jnp.stack → (num_agents, actions_per_agent)"]

    style Body fill:#2d3748,stroke:#4a5568,color:#e2e8f0
    style Head0 fill:#1a365d,stroke:#2b6cb0,color:#bee3f8
    style Head1 fill:#1a365d,stroke:#2b6cb0,color:#bee3f8
    style HeadN fill:#1a365d,stroke:#2b6cb0,color:#bee3f8
    style Input fill:#553c9a,stroke:#6b46c1,color:#e9d8fd
    style Stack fill:#22543d,stroke:#2f855a,color:#c6f6d5
```

## Scenario Presets

| Scenario | hidden_dim | n_body_layers | n_head_layers | use_layer_norm | B | alpha |
|---|---|---|---|---|---|---|
| `3m` | 128 | 2 | 1 | No | 32 | 0.0005 |
| `2s3z` | 128 | 3 | 1 | No | 32 | 0.0005 |
| `3s_vs_5z` | 128 | 3 | 1 | No | 32 | 0.0005 |
| `5m_vs_6m` | 192 | 3 | 1 | No | 32 | 0.0005 |
| `3s5z` | 256 | 3 | 2 | Yes | 64 | 0.0003 |
| `8m` | 256 | 3 | 1 | Yes | 64 | 0.0003 |
| `3s5z_vs_3s6z` | 256 | 3 | 2 | Yes | 64 | 0.0003 |
| `10m_vs_11m` | 512 | 4 | 2 | Yes | 128 | 0.0001 |
| `25m` | 512 | 4 | 2 | Yes | 128 | 0.0001 |

When `n_head_layers=1`, the head hidden layers are skipped — each head is just a single `Dense(actions_per_agent)` linear projection (matching the original architecture).
