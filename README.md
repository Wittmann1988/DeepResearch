# DeepResearch

AI-gestützte Tiefenrecherche und Wissensaggregation.

## Vision
Autonome Research-Pipeline die Themen mehrstufig analysiert:
1. **Breite Suche** — GitHub, Papers, Docs, Foren
2. **Tiefenanalyse** — Code lesen, Architekturen verstehen, Benchmarks vergleichen
3. **Synthese** — Erkenntnisse verdichten, Empfehlungen ableiten
4. **Persistenz** — Ergebnisse in Memory Framework speichern

## Stack
- Nemotron-3-Nano 30B (Cloud) — Analyse und Synthese
- LocoOperator-4B (Lokal) — Codebase-Navigation
- Ollama Sidekick MCP — GitHub Search, Research Tools
- Memory Framework — Langzeitspeicher für Forschungsergebnisse

## Bisherige Forschungen
- Memory Frameworks für Android/Termux (sqlite-vec vs Mem0 vs ChromaDB)
- LocoOperator-4B Evaluierung (Tool-Calling, JSON Validity)
- Ollama Cloud API Integration

## Status
Konzeptphase — Architektur und erste Implementierungen folgen.

---

## MiroMind AI – Vollständige Analyse aller 5 Repositories

### 1. EINZELANALYSE DER REPOSITORIES

#### MiroMind-M1 — Das Fundament: Mathematisches Reasoning-Modell

Was es ist: MiroMind-M1 ist eine vollständig quelloffene Serie von Reasoning-Sprachmodellen, aufgebaut auf Qwen-2.5, mit dem Fokus auf fortgeschrittenes mathematisches Schlussfolgern. Kern-Technologie: Das Modell wird in zwei Phasen trainiert: erstens durch Supervised Fine-Tuning (SFT) auf 719.000 kuratierten mathematischen Problemen, danach durch Reinforcement Learning with Verifiable Rewards (RLVR) auf 62.000 besonders schwierigen Beispielen. Das Herzstück ist eine eigens entwickelte Methode namens CAMPO (Context-Aware Multi-Stage Policy Optimization), die das Modell beibringt, schrittweise komplexer werdende Reasoning-Ketten zu generieren. Verfügbare Modellgrößen: 7B (SFT + RL) und 32B (RL), basierend auf Qwen-2.5. Benchmark-Leistung (MiroMind-M1-RL-7B):
* AIME 2024: 73,4% (Avg@64) — besser als DeepSeek-R1-Distill-Qwen-7B (55,5%)
* AIME 2025: 57,8% — besser als MiMo-7B-RL (55,4%)
* MATH500: 96,7%

Besonderheiten im Code: Das Training läuft auf mehreren GPUs via Ray (Multi-Node), nutzt verl als RL-Unterbau und stellt fertige Evaluation-Skripte für AIME24, AIME25 und MATH500 bereit. Zweck in der Pipeline: M1 ist das reine Reasoning-Basismodell — es lernt, logisch zu denken, ohne externe Tools. Es bildet die konzeptuelle Grundlage für alle anderen Agenten.

#### MiroTrain — Die Trainings-Fabrik: SFT & DPO für Agenten

Was es ist: MiroTrain ist ein effizienter, algorithmen-first Post-Training-Framework speziell für große agentische Sprachmodelle. Es baut auf TorchTune auf und erweitert es um agentenspezifische Fähigkeiten. Was es tut: Es stellt produktionsreife Trainingsrezepte für zwei Methoden bereit: SFT (Supervised Fine-Tuning) — das Modell lernt aus guten Beispieltraces — und DPO (Direct Preference Optimization) — das Modell lernt, welche Antworten bevorzugt werden. Konkret wurden damit die MiroThinker-Modelle (v0.1) trainiert, unter Verwendung des MiroVerse-Datensatzes.

Technische Highlights:
* Unterstützt 32B-Scale-Training auf einem einzigen Node mit 8x80GB GPUs
* Sequence Parallelism und CPU-Offloading für maximale Speichereffizienz
* streaming_pack: Packing von Trainingsdaten on-the-fly ohne Vorverarbeitung
* FlashAttention und Triton Kernels für maximalen Durchsatz
* FSDPv2-kompatibel (Distributed Training mit DTensor-basiertem Sharding)
* HuggingFace-kompatible Checkpoints -> direkt mit Transformers/vLLM/SGLang ladbar
* Docker-Image für schnellen Einstieg verfügbar

Konfigurationsbeispiel (MiroThinker-8B-SFT-v0.1): 4 Epochen, Lernrate 4e-5, 40K Kontextlänge, 128 Batch-Größe, Packed Data aktiviert. Zweck in der Pipeline: MiroTrain ist die Trainingsmaschine — sie verarbeitet annotierte Traces (aus MiroFlow) und produziert daraus bessere Agenten-Modelle. Es ist die Fabrik, die rohe Erfahrung in Modellgewichte umwandelt.

#### MiroRL — Das RL-Labor: Reinforcement Learning mit echten Tools

Was es ist: MiroRL ist das erste Reinforcement-Learning-Framework, das MCP (Model Context Protocol) nativ unterstützt — also Multi-Turn-Tool-Calls direkt im RL-Trainingsprozess. Das ist ein fundamentaler Unterschied zu klassischem RLHF. Was es revolutioniert: Klassisch wird RL isoliert trainiert und Tools erst später integriert. MiroRL trainiert den Agenten während er echte Tools benutzt: Web-Suche (Serper), Web-Scraping (Jina), Python-Interpreter (E2B), Linux-Shell. Der Agent lernt direkt aus realen Tool-Interaktionen mit echtem Environment-Feedback. Kern-Algorithmus: GRPO (Group Relative Policy Optimization) mit asynchronen Rollouts für Multi-Turn-Konversationen. Laut Dokumentation erzielt dies mehr als 2x schnellere End-to-End-Trainingsperformanz gegenüber Standard-Einstellungen.

Technische Besonderheiten:
* MCP-first Architektur: Tools werden als MCP-Server definiert (YAML-Konfiguration)
* Asynchrones Rollout: Partielle und Streaming-Rollouts für lange Kontexte
* Speichereffizienz: Triton-Kernels + Sequence Parallelism + CPU-Offloading -> 14B-Modell mit 64K Kontext auf einem einzigen GPU-Node trainierbar
* Erweiterbar: Eigene MCP-Tools können einfach per YAML hinzugefügt werden
* Pre-built Docker-Image für sofortigen Einstieg

Trainiertes Modell: MiroRL-14B-SingleAgent-Preview-v0.1 erreicht 40,29% (Avg@8) auf GAIA-text-103. Kosten pro Trainingsschritt: ca. $0,85 (Serper) + $1,23 (Jina) + $0,20 (OpenAI) ~ $2,28 pro Step — relevant für Budget-Planung. Zweck in der Pipeline: MiroRL ist das RL-Labor — es verbessert Agenten durch direkte Interaktion mit der realen Welt und ist damit die fortgeschrittenste Trainingsschicht.

#### MiroFlow — Das Betriebssystem: Agent-Ausführungs-Framework

Was es ist: MiroFlow ist das zentrale Ausführungs-Framework für KI-Agenten — das "Betriebssystem", auf dem alle Agenten laufen. Es koordiniert Tools, Modelle und Konversationsflüsse.

Kern-Fähigkeiten:
* Multi-Turn-Konversation mit vollständiger Kontextverwaltung
* Hierarchische Sub-Agenten-Orchestrierung: Ein Haupt-Agent kann spezialisierte Sub-Agenten delegieren (z.B. einen Browsing-Agenten für Webrecherche)
* Tool-Ecosystem: Audio-Transkription (Whisper), Python-Ausführung (E2B), Dateilesen (MarkItDown), Web-Suche (Google/Sogou/Serper), Vision (VQA via Claude oder Qwen2.5-VL), Reasoning-Engine
* MCP-Protokoll: Tools werden als MCP-Server angebunden — vollständig austauschbar
* Hohe Concurrency: Robust gegen Rate-Limits und instabile Netzwerke, fault-tolerant
* Trace-Sammlung: Vollständige Logging-Infrastruktur für SFT/DPO-Datengewinnung
* Hydra-Konfiguration: Agenten-Setups per YAML konfigurierbar, keine Code-Änderungen nötig

Modell-Unterstützung: GPT, Claude, Gemini, Qwen — über OpenRouter oder direkt. Benchmark-Leistung (mit GPT-5): GAIA Val 82,4%, HLE 27,2%, BrowseComp-EN 33,2%, BrowseComp-ZH 47,1%, xBench-DeepSearch 72,0% — #1 auf FutureX (Future Prediction Benchmark). Context Retention Strategy (keep_tool_result): Nur die K zuletzt erhaltenen Tool-Ergebnisse werden im Kontext behalten — spart Token, verbessert Fokus, ermöglicht längere Reasoning-Chains ohne Performanzverlust. Zweck in der Pipeline: MiroFlow ist die Runtime und Daten-Pipeline zugleich — es führt Agenten aus UND sammelt Traces für weiteres Training.

#### MiroThinker — Der Superstar: Optimierter Deep-Research-Agent

Was es ist: MiroThinker ist der Flaggschiff-Agent von MiroMind — ein speziell fein abgestimmtes Modell, das auf MiroFlow läuft und für tiefe Recherche- und Vorhersageaufgaben optimiert ist. Die wichtigste Innovation — Interactive Scaling: Neben Modellgröße und Kontextlänge führt MiroThinker eine dritte Dimension der Skalierung ein: die Tiefe und Häufigkeit von Agent-Umgebungs-Interaktionen. Das Modell lernt explizit, mehr und tiefere Tool-Calls zu machen, um Fehler zu korrigieren und Trajektorien zu verfeinern.

Modell-Versionen: v1.5 (aktuell): Basiert auf Qwen3-30B-A3B-Thinking-2507 (30B) und Qwen3-235B-A22B-Thinking-2507 (235B), 256K Kontext, bis zu 400 Tool-Calls pro Task. MiroThinker-v1.5-30B übertrifft Kimi-K2-Thinking auf BrowseComp-ZH bei nur 1/30 der Parameter. v1.0: 8B, 30B, 72B — bis zu 600 Tool-Calls, 256K Kontext. v0.1/v0.2: 4B bis 32B, SFT+DPO-Varianten, 40K-64K Kontext.

Benchmark-Highlights (v1.5-235B): HLE-Text 39,2%, BrowseComp 69,8%, BrowseComp-ZH 71,5%, GAIA-Val-165 80,8% — neues State-of-the-Art unter Open-Source-Agenten. Tool-Minimalsetup: Nur 3 MCP-Server nötig: Google-Suche (Serper), Web-Scraping (Jina), Code-Ausführung (E2B) + ein kleines Summary-LLM (Qwen3-14B genügt). Deployment-Optionen: SGLang, vLLM, llama.cpp (CPU-quantisiert), Ollama — vom Datacenter bis zur einzelnen RTX 4090. Zweck in der Pipeline: MiroThinker ist das Endprodukt — der tatsächlich einsetzbare Agent, der für Nutzer und Forscher arbeitet. Er ist gleichzeitig Demonstrationsobjekt und Produktionsagent.

### 2. DIE PIPELINE — WIE ALLES ZUSAMMENHÄNGT

Das Geniale an MiroMind ist, dass die fünf Repositories keine isolierten Projekte sind, sondern ein vollständig geschlossenes, sich selbst verbesserndes System bilden:

```
MiroMind-M1          -> Basis-Reasoning-Fähigkeiten (Mathe, Logik)
        |
    MiroTrain        -> SFT + DPO: Grundtraining auf Agenten-Traces
        |
    MiroFlow         -> Laufzeitumgebung + Trace-Sammlung aus echter Nutzung
        |
     MiroRL          -> RL mit echten Tools: Modell lernt live aus Feedback
        |
  MiroThinker        -> Fertig-Produkt: deployierbarer Forschungsagent
        |
   (Neue Traces)     -> fliessen zurueck in MiroFlow -> MiroTrain -> MiroRL
```

Phase 1 – Fundament (MiroMind-M1 + MiroTrain): Aus Qwen-2.5 wird durch mathematisches SFT + CAMPO-RLVR ein stark reasonendes Basis-LLM. MiroTrain fährt dann mit dieser Basis SFT und DPO auf agentischen Datensätzen (MiroVerse, 147K Samples) — das Modell lernt, Tool-Use-Trajektorien zu erzeugen.

Phase 2 – Agentenausführung (MiroFlow): Das trainierte Modell wird in MiroFlow eingebettet. Dort bekommt es Zugang zu echten Tools (Websuche, Code, Dateien) und kann komplexe Multi-Step-Tasks lösen. MiroFlow sammelt dabei vollständige Traces jeder Interaktion.

Phase 3 – RL-Verfeinerung (MiroRL): Die gesammelten Trajektorien + echte Tool-Interaktionen werden in MiroRL für GRPO-Training genutzt. Das Modell erhält direkt Rewards basierend auf der Qualität seiner Tool-Nutzung in der realen Umgebung — kein simuliertes Feedback mehr.

Phase 4 – Produkt (MiroThinker): Das RL-fein-abgestimmte Modell wird als MiroThinker deployt — mit bis zu 256K Kontext und 400 Tool-Calls pro Task. MiroThinker selbst generiert bei Nutzung wieder neue Traces -> der Zyklus schliesst sich.

Das entscheidende Alleinstellungsmerkmal: Die Feedback-Schleife ist vollständig geschlossen und vollständig Open-Source. Andere Systeme (OpenAI Deep Research, Gemini Deep Research) haben ähnliche Endprodukte, aber die Trainings-Pipeline ist proprietär. MiroMind veröffentlicht alles: Daten, Trainingsrezepte, Modellgewichte, Framework.

### 3. ZUSAMMENFASSUNG: WAS DIESES OEKOSYSTEM ERMOEGLICHT

MiroMind hat im Wesentlichen einen vollständigen, quelloffenen Workflow zum Bau autonomer Forschungsagenten geschaffen. Die praktischen Möglichkeiten lassen sich in drei Kategorien einteilen:

**Sofort nutzbar (Endanwender):** MiroThinker auf SGLang/Ollama lokal deployen, mit drei API-Keys (Serper, Jina, E2B) verbinden und einen leistungsstarken Forschungsagenten betreiben, der selbstständig Webrecherchen durchführt, Code schreibt und ausführt, Dokumente analysiert und Vorhersagen zu zukünftigen Ereignissen macht. Eine einzelne RTX 4090 genügt für die 30B-Version.

**Für Forscher und Entwickler:** Das vollständige Training-Stack ermöglicht eigene spezialisierte Agenten. Man kann MiroVerse-Daten nehmen, mit MiroTrain auf einer eigenen Domäne (z.B. Medizin, Recht, Finanzen) finetunen, mit MiroRL weiter durch echte Tool-Interaktionen verfeinern und in MiroFlow deployen — alles ohne proprietäre Komponenten.

**Für Unternehmen:** Ein vollständig On-Premise-deploybarer, hochleistungsfähiger Forschungsagent, der keine externen Dienste ausser optionalen Such-APIs benötigt, vollständig auditierbar und erweiterbar ist, und sich durch eigene Nutzungsdaten kontinuierlich verbessert.

### 4. AUSBLICK: WOHIN SICH DAS ENTWICKELN WIRD

Mehrere Entwicklungslinien zeichnen sich klar ab: Grössere Modelle, längere Horizonte. Der Sprung von 40K (v0.1) über 64K (v0.2) auf 256K (v1.0/v1.5) Kontext in wenigen Monaten zeigt die Richtung: Agenten werden zunehmend in der Lage sein, sehr lange Rechercheprozesse — stunden- oder tagelange Informationssammlung — in einem einzigen Kontext zu verarbeiten. Die 400-Tool-Call-Grenze wird weiter steigen.

Vollständig offline / Open-Source-Toolchain. Derzeit sind Serper (Google Search), Jina und E2B kommerzielle Services. MiroMind bietet bereits Open-Source-Alternativen für Vision (Qwen2.5-VL), Audio (Whisper) und Reasoning an. Es ist nur eine Frage der Zeit, bis auch Suche und Scraping vollständig lokal laufen — was echte Air-Gap-Deployments ermöglicht.

Breitere Sprach- und Domänenabdeckung. MiroMind selbst nennt die begrenzte Chinesisch-Performance als Schwäche von v0.1 und plant mehr mehrsprachige Trainingsdaten. Mit MiroTrain und MiroVerse als offenen Systemen werden Community-Beiträge für weitere Sprachen und Spezialdomänen entstehen.

MCP als Standard für agentische Tool-Integration. MiroRL und MiroFlow setzen beide auf MCP (Model Context Protocol von Anthropic) als Schnittstelle. Da dieses Protokoll sich als De-facto-Standard für LLM-Tool-Integration etabliert, wird das MiroMind-Ökosystem von jedem neuen MCP-kompatiblen Tool profitieren — ohne Code-Änderungen.

Geschlossene RL-Schleife mit echter Nutzerfeedback. Das nächste logische Upgrade von MiroRL ist die Integration von Human-Feedback-Signalen aus dem Online-Demo — wenn echter Nutzernutzen als Reward-Signal ins RL einfliesst, entsteht ein sich selbst verbesserndes Produktionssystem.

Wie Sie es heute konkret nutzen können: Den schnellsten Einstieg bieten MiroFlow (nur OpenRouter-Key nötig) für erste Experimente oder MiroThinker mit dem Minimal-Setup (Serper + Jina + E2B) für produktiven Einsatz. Wer eigene Modelle trainieren möchte, startet mit MiroTrain auf dem MiroVerse-Datensatz. Wer die volle Kontrolle über die RL-Schleife will und Rechenbudget hat, setzt MiroRL für domänenspezifisches Reinforcement Learning ein. Die Community ist aktiv (Discord, 1,2K Follower, 6,5K Sterne für MiroThinker), und alle Komponenten sind vollständig dokumentiert — ein ungewöhnlich durchdachtes und zusammenhängendes Open-Source-Ökosystem für autonome KI-Agenten.

---

## Agentic Architecture

> Konsolidiert aus dem ehemaligen AgenticFramework-Repository.

### Agent-Architekturen
- **ReAct** (Reason + Act)
- **Plan-and-Execute**
- **Multi-Agent Orchestrierung** (Nemotron 126-Agent System)
- **Tool-Calling Patterns**: XML vs JSON vs Function Calling

### Lokale vs Cloud Agenten
- LocoOperator-4B: 2.5GB, 100% JSON Validity, Tool-Calling
- Nemotron-3-Nano 30B: 126 spezialisierte Sub-Agenten
- Hybride Architektur: Cloud fuer Planung, Lokal fuer Ausfuehrung

### Hybrid Agent Loop
```
User Task
    |
[Claude Code] <-> [Nemotron Sidekick]
    |                    |
[Plan]            [Research]
    |                    |
[LocoOperator]    [Code Review]
    |                    |
[Execute]         [Improve]
    |                    |
[Memory Store] <-> [Task Decompose]
```

### Unsere Implementierung
- **Agent Forge**: Loop Agent + Self-Improve Pipeline
- **Ollama Sidekick**: Research-First + Auto-Review
- **LocoOperator**: Autonome Codebase-Navigation
- **Memory Framework**: sqlite + Decay + Reinforcement

### Self-Improvement Patterns
- Continuous Code Review
- Automatische Refactoring-Vorschlaege
- Performance-Monitoring
- A/B Testing von Agent-Strategien
