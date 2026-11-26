# Devpost submission text

Copy each block into the matching Devpost field. Markdown works in "About the project".

---

## Project name

Shadowfield

## Elevator pitch (max 200 characters)

Decodes satellite telemetry nobody can read: every byte is tested against the orbit's physics (sunlight, shadow, position), then Claude names each field like a spacecraft engineer.

---

## About the project

### Inspiration

Hundreds of small satellites are sending telemetry to Earth right now, and SatNOGS volunteers record it every day with their own ground stations. For many of these satellites, though, nobody outside the team can read the data. The format was never published, so the recordings are only hex.

Writing a decoder normally needs the builders' documentation. But one thing about every satellite is public: its **orbit**. From the orbit you know exactly when the spacecraft was in sunlight and when it was in Earth's shadow. Batteries charge in sunlight, solar panels cool in the dark, counters tick, GNSS receivers trace the orbit. That physics leaves fingerprints in the bytes. Shadowfield uses those fingerprints to read the data.

### What it does

Shadowfield needs only two inputs: the raw frames from SatNOGS and the satellite's public orbit (TLE). It gives back:

- **A field map.** Where each field starts, how wide it is, its byte order and its sign.
- **Names, units and scaling.** Claude proposes these from the evidence, together with a short health note on the spacecraft.
- **A Kaitai Struct decoder draft (`.ksy`).** This is the format SatNOGS decoders are written in. The drafts compile with the official compiler.

For every frame, Shadowfield first works out from the orbit:
- whether the satellite was in sunlight, and its sunlight history;
- the minutes since sunrise or sunset;
- the orbital phase;
- its position and velocity;
- Earth's magnetic field at that point.

Then it tests every possible reading of the bytes against that physics:

| What the bytes do | What the field is |
|---|---|
| Steps when the satellite enters or leaves Earth's shadow | Solar current, battery voltage |
| Follows sunlight with a lag of minutes | Temperatures, state of charge |
| Only goes up, at a fixed rate | Uptime and packet counters |
| Matches a coordinate of the orbit times a scale | GNSS position and velocity |
| Three values whose length follows Earth's field model | 3-axis magnetometer |
| Four floats whose squares add up to one | Attitude quaternion |

**Claude** (Sonnet 5.5 through the Anthropic API) then reads each field's evidence like a spacecraft engineer would. The evidence is its response to sunrise and sunset, its time constant, its range, and which fields move together. Claude proposes a name, unit and scale. It never sees the satellite's name or its decoder, so it cannot recall an answer it has seen before. It is asked three times and the majority answer wins, with structured output.

### How I built it

- **Physics:** SGP4 orbit propagation, Sun position and a model of Earth's shadow. It agrees with Skyfield's full-precision shadow on 99.7% of 10-second samples. The magnetic field comes from IGRF (`ppigrf`).
- **Byte engine:** NumPy, pandas and SciPy, split into small steps: readings, correlation, detectors, structure tests and selection.
  - Correlations are computed within each week, with significance counted in satellite passes rather than frames.
  - Structure tests settle the layout. They look for carries between bytes, sign extension and the point where a counter wraps.
- **Claude:** the `anthropic` Python SDK, with a pydantic schema for structured output and a disk cache. The whole 15-satellite benchmark costs about $2 in API calls.
- **Grading:** `satnogs-decoders`, the community's published decoders. They are used only *after* a run, to score it, never as input.
- **Demo page:** one self-contained HTML file with Three.js, NASA Blue Marble and Black Marble imagery, and the real benchmark results inlined.

### Results, graded blind against real decoders

Satellites that already have public decoders let us check the answers. Shadowfield runs on their raw frames as if no decoder existed, and only afterwards is it compared with the decoder.

| | Satellites | Decoder fields | Fields found | Reported fields that are real | Exact byte layout | Claude's category right |
|---|---|---|---|---|---|---|
| Development set (rules tuned while looking at it) | 9 | 172 | 76% | 95% | 63% | 77% |
| **Held-out set (never inspected, engine frozen)** | **6** | **126** | **66%** | **98%** | **40%** | **73%** |

The held-out set is six satellites from six different makers, run once. Its numbers are the honest estimate for a satellite nobody has looked at. Both sets are reported, and the weaker one is not hidden.

### Challenges I ran into

- **Mixing seasons fooled the physics, and Claude caught it first.** I added NetSat's eclipse season to its autumn data. Suddenly almost every byte "followed sunlight" with the same 10-minute lag. Asked to name the fields, Claude refused. It said they *"move together and share the same lag… the sun correlations are probably artefacts of sampling order."* It was right: anything that drifted between June and November looked like an eclipse response. Shadowfield now compares within weeks, and it uses sunlight correlations only from weeks where the sunlight history actually varies.
- **Satellites that never enter Earth's shadow.** A dawn-dusk orbit can stay in sunlight for months, so nothing changes for the physics to explain. Shadowfield computes each satellite's eclipse calendar from its TLE and fetches recordings from the weeks when it *did* pass through shadow.
- **Frames that look encrypted.** Amateur-band telemetry may not be encrypted (ITU Radio Regulations Art. 25). Random-looking frames are usually just scrambled, so Shadowfield tries the standard CCSDS and G3RUH layers. It can also recover an unknown fixed whitening key from the physics. Each layer is accepted only if a structural test passes, so clean frames are never "peeled". Real ciphertext is recognised and reported, never "decoded".
- **Messy real data:**
  - the same frame heard by several stations;
  - ground station clocks that disagree;
  - GNSS receivers repeating stale fixes;
  - signed currents that naive methods split into a "sign byte" and a "low byte".

### Accomplishments that I'm proud of

- On satellites it had never seen, **98% of the fields it reports are real**.
- Every generated `.ksy` **compiles with the official Kaitai Struct compiler**, and the compiled parser reads the real frames exactly as the engine does.
- 70 automated tests:
  - a random synthetic fleet of satellites with planted traps;
  - a regression lock on every real frame type;
  - regression tests for each bug an independent code review found.
- Claude's role is checkable. Every name it gives is graded against a decoder it never saw.

### What I learned

- An LLM is most useful when it reasons over measured evidence rather than guessing from raw data. Hiding the satellite's name made the test honest.
- A held-out set changes everything. My first held-out numbers were lower than the development numbers. Once I had inspected that set to fix its failures, it no longer counted as held-out, so I replaced it with a fresh one.
- Physics gives strong ground truth for free. Sunlight and shadow are known to the second from a public orbit.

### What's next

- Run on SatNOGS satellites that have **no** decoder, and share the `.ksy` drafts with their operators and the satnogs-decoders project.
- Improve the exact byte layout on unseen satellites (40% today), and add status bitfields, which physics alone cannot reach.
- A nightly job that re-checks fields as new frames arrive and flags health changes: a battery that charges more slowly, or a panel that runs hotter.

---

## Built with

python, numpy, pandas, scipy, sgp4, ppigrf, pydantic, anthropic, claude, satnogs, satnogs-decoders, kaitai-struct, three.js, webgl, javascript, html

## Try it out (links)

- GitHub repo: https://github.com/kadircanyildirm-crypto/shadowfield (make it public, or invite the judges, before submitting)
- Live demo: https://kadircanyildirm-crypto.github.io/shadowfield/ (works after GitHub Pages is turned on for `master` / `docs`)

---

## AI tools used (disclosure)

AI was used in two separate ways.

**1. Inside the product.**
- Claude Sonnet 5.5, through the Anthropic API, names the fields Shadowfield finds and writes the health note.
- It receives only a summary of the measured evidence. It never sees the satellite's name or its published decoder.
- Its answers are graded against those decoders, and the scores are shown in the results above.

**2. In development.**
- I built this project with heavy help from Claude Code, Anthropic's agentic coding tool, running Claude Opus 5.5.
- Claude Code wrote most of the code, the tests, the demo page and the documentation under my direction. It also helped with brainstorming and with searching for prior work.
- I chose the idea, set the goals and requirements, ran and checked the results on real data, asked for fixes and changes, and made the scope and design decisions.

**Data and assets.**
- Frames come from the SatNOGS Network (CC BY-SA).
- Grading uses the community's satnogs-decoders.
- Earth imagery is NASA Blue Marble and Black Marble (public domain).
