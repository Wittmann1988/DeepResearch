# DeepResearch Self-Improving Pipeline

## Architektur — Inspiriert von MiroMind, adaptiert fuer unser Setup

```
┌─────────────────────────────────────────────────────────┐
│                    TRACE COLLECTION                      │
│  Claude Code Sessions | Agent Forge Tasks | Ollama Runs  │
│  → Jede Interaktion wird geloggt (Prompt + Response)     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   DATASET (HuggingFace Hub)              │
│  erik1988/agent-traces (Chat-Format)                     │
│  erik1988/agent-evaluations (Bewertungen)                │
│  → SQL-Queries, Filterung, Qualitaetskontrolle           │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 TRAINING (HF Jobs Cloud)                  │
│  SFT auf gesammelten Traces (trl + LoRA)                │
│  DPO auf Bewertungspaaren (chosen/rejected)              │
│  GRPO fuer Tool-Use Verbesserung (wie MiroRL)            │
│  → Modell wird auf HF Hub gepusht                        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              GGUF CONVERSION (HF Jobs)                   │
│  Trainiertes Modell → GGUF Q4/Q5/Q8                     │
│  → Download auf Geraet oder via Ollama pull              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                DEPLOYMENT (Lokal via Ollama)              │
│  Verbessertes Modell ersetzt altes                       │
│  Agent Forge / Sidekick / LocoOperator nutzen es         │
│  → Neue Traces werden gesammelt → Zyklus schliesst sich  │
└─────────────────────────────────────────────────────────┘
```

## Konkrete Schritte

### Phase 1: Trace Collection (JETZT machbar)
- Jede Claude Code Session: Learnings in Memory dokumentieren
- Agent Forge: Task-Ergebnisse loggen (erfolg/misserfolg/qualitaet)
- Ollama Sidekick: Alle Anfragen + Antworten speichern
- Format: HF Chat-Template (messages mit role/content)

### Phase 2: Dataset Management (JETZT machbar)
- HF Dataset erstellen: erik1988/agent-traces
- SQL-Queries fuer Qualitaetsfilterung (sql_manager.py)
- Automatische Kategorisierung: code, research, system, debug

### Phase 3: Training (Via HF Jobs)
- SFT: Gute Traces als Trainingsdaten
- DPO: Bewertungspaare (gute vs schlechte Antworten)
- GRPO: Tool-Use Optimierung (MiroRL-Ansatz)
- Hardware: a10g-large fuer 7B Modelle, t4 fuer Tests
- LoRA fuer Speichereffizienz

### Phase 4: Deployment
- GGUF Konvertierung via HF Jobs
- Download auf Geraet oder Ollama Registry
- Ersetze LocoOperator-4B / Sidekick-Modell

### Phase 5: Benchmarking
- MiroFlow-Benchmarks als Referenz
- Eigene Benchmarks fuer unsere Use-Cases
- Vergleich vor/nach Training

## Verfuegbare MiroMind Ressourcen auf HuggingFace

### Modelle (als Basis oder Referenz)
- miromind-ai/MiroThinker-v1.5-235B (Flaggschiff)
- miromind-ai/MiroMind-M1-RL-7B (Reasoning-Basis, GGUF verfuegbar)
- miromind-ai/MiroMind-M1-RL-32B (groessere Reasoning-Basis)
- bartowski/miromind-ai_MiroThinker-v1.0-8B-GGUF (lokal nutzbar!)

### Datasets (als Trainingsvorlage)
- miromind-ai/MiroVerse-v0.1 (147K Agent-Traces, gated)
- miromind-ai/MiroRL-GenQA (RL Trainingsdaten)
- miromind-ai/MiroMind-M1-SFT-719K (Mathe-Reasoning SFT)
- miromind-ai/MiroFlow-Benchmarks (Evaluation)

### Tools
- HF Jobs: Cloud-Training (SFT/DPO/GRPO via TRL)
- HF Datasets: SQL-Queries, Erstellung, Verwaltung
- GitHub MCP: Repo-Zugriff auf MiroMind-Quellcode
- Ollama: Lokale Inferenz + Deployment

## Projektverzahnung

| Projekt | Rolle in der Pipeline | Datenfluss |
|---|---|---|
| DeepResearch | Steuerung, Dokumentation, Forschung | → Learnings → Memory |
| Agent Forge | Agent-Ausfuehrung, Task-Traces | → Traces → Dataset |
| SystemManager | System-Optimierung, Ressourcen | → Performance-Daten |
| MemoryFramework | Langzeitspeicher, Retrieval | → Kontext fuer Agenten |
| Ollama Sidekick | Ko-Agent, Research-Tools | → Traces → Dataset |
| LocoOperator-4B | Codebase-Navigation | → Kandidat fuer Training |

## Memory-Integration

Alles was gelernt wird, fliesst in:
1. **MEMORY.md** — Sofortige Nutzung in naechster Session
2. **HF Dataset** — Langzeit-Trainingsdaten
3. **sqlite-vec** (TODO) — Semantisches Retrieval mit Embeddings
4. **Agent Forge Memory** — sql.js fuer Agent-Kontext
