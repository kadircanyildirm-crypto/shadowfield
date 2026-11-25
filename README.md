# Shadowfield

**Decode a satellite's telemetry without its decoder, using the orbit it flies.**

Hundreds of small satellites broadcast housekeeping beacons that amateur ground stations
(SatNOGS) record every day. Many of those beacons have no public format: they are just bytes.
Shadowfield works out what the bytes mean from physics alone, then has Claude read the evidence
the way a spacecraft engineer would.

It needs only two inputs: the raw frames and the satellite's public orbit (TLE). From the orbit
it computes, for every frame:
- whether the spacecraft was in sunlight, and its sunlight history over the previous 200 minutes;
- minutes since sunrise or sunset;
- orbital phase and beta angle;
- GNSS-style position and velocity;
- the IGRF magnetic field at its position.

It then tests every possible reading of the frame bytes against that physics:

| behaviour in the bytes | what it is |
|---|---|
| steps when the satellite enters or leaves Earth's shadow | solar or charge current, battery voltage |
| follows sunlight with a lag of minutes | temperatures, state of charge |
| only increases, at a fixed rate | uptime, packet counters (also 1-byte rolling ones) |
| tracks wall-clock time | on-board clock (epoch detected) |
| equals a coordinate of the orbit times a scale | GNSS position / velocity (ECEF or inertial; int or float32) |
| three values whose vector length follows the IGRF model | 3-axis magnetometer |
| four float32 values whose squares sum to one | attitude quaternion |

This finds where each field starts, how wide it is, its byte order and its signedness. **Claude**
(Sonnet 5.5, structured output, majority of three samples) then reads the evidence and proposes a
name, unit and scaling for each field, plus a short health note. The evidence includes the
sunrise/sunset response shape, the time constant, the raw ranges and which fields move together.
Claude is never told the satellite's name, so it cannot recall a known decoder. The output is an
HTML report and a **Kaitai Struct decoder draft** (`.ksy`), the format satnogs-decoders uses.

## Does it work? Blind tests against real decoders

Some satellites do have public decoders
([satnogs-decoders](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders)). We run
Shadowfield on their raw frames as if they did not, and use the decoder only afterwards, to grade.

**Development set.** The engine's rules were tuned while looking at these satellites:

<!-- DEV_TABLE -->
| satellite | frame type | frames / passes | decoder fields | recall | exact | precision | kind | Claude | answered | when answered | battery / solar / bus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UWE-4 | `DP0UWH:len58` | 1105 / 411 | 15 | 100% | 93% | 100% | 87% | 100% | 100% | 100% | 86% |
| Geoscan | `len72:b0=0x84` | 1476 / 103 | 15 | 80% | 80% | 100% | 67% | 75% | 75% | 100% | 75% |
| Geoscan | `len72:b0=0x04:b3=0x05` | 321 / 99 | 14 | 79% | 64% | 100% | 64% | 91% | 91% | 100% | – |
| GRBBeta | `len79:b0=0x85` | 347 / 44 | 23 | 70% | 61% | 100% | 100% | 88% | 100% | 88% | 77% |
| SNUGLITE | `len97` | 1005 / 126 | 7 | 100% | 86% | 88% | 100% | 100% | 100% | 100% | 100% |
| SNUGLITE | `len214` | 405 / 108 | 30 | 90% | 83% | 94% | 96% | 96% | 96% | 100% | 100% |
| NetSat | `len89` | 1724 / 241 | 14 | 43% | 21% | 86% | 100% | 17% | 17% | 100% | – |
| AlSat-1N | `len19` | 1781 / 522 | 4 | 50% | 50% | 100% | 100% | 50% | 100% | 50% | 0% |
| AlSat-1N | `len20` | 910 / 201 | 1 | 100% | 100% | 50% | 100% | 100% | 100% | 100% | – |
| AlSat-1N | `len32` | 814 / 162 | 5 | 40% | 40% | 100% | 100% | 100% | 100% | 100% | – |
| Sputnix USP | `len103` | 474 / 80 | 10 | 80% | 70% | 100% | 50% | 38% | 100% | 38% | – |
| CubeBel-2 | `EU11S:len200:b0=0x14:b6=0x10` | 161 / 27 | 17 | 88% | 35% | 94% | 93% | 60% | 67% | 90% | 0% |
| Geoscan (2nd unit) | `len72:b0=0x09` | 360 / 81 | 4 | 0% | 0% | 33% | – | – | – | – | – |
| Geoscan (2nd unit) | `len72:b0=0x84` | 331 / 74 | 13 | 69% | 62% | 100% | 78% | 78% | 78% | 100% | 100% |
| **total** | | | **172** | **76%** | **63%** | **95%** | **87%** | **77%** | **84%** | **92%** | **77%** |
<!-- /DEV_TABLE -->

**Held-out set.** Six satellites from six different makers, run once with the engine frozen and
never inspected while tuning. This is the honest estimate for a satellite nobody has looked at. (An
earlier held-out set of Sputnix, CubeBel-2 and a second Geoscan scored 70% recall and 41% exact. It
was then inspected to fix what failed there, so it now belongs to development, and this fresh set
replaced it.)

<!-- HELDOUT_TABLE -->
| satellite | frame type | frames / passes | decoder fields | recall | exact | precision | kind | Claude | answered | when answered | battery / solar / bus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CP16 | `len239` | 970 / 111 | 37 | 84% | 35% | 100% | 84% | 39% | 58% | 67% | 38% |
| KNACKSAT-2 | `HS0K:len149` | 195 / 23 | 22 | 5% | 0% | 100% | 100% | 100% | 100% | 100% | 0% |
| Kostka | `OK0KST:len59` | 180 / 54 | 1 | 100% | 100% | 3% | 100% | 0% | 0% | – | – |
| GBSAT | `GBSAT0:len68` | 897 / 94 | 14 | 71% | 50% | 90% | 80% | 100% | 100% | 100% | 100% |
| CANVAS | `LASP:len212` | 2998 / 46 | 2 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | – |
| CANVAS | `LASP:len248` | 1753 / 46 | 44 | 82% | 57% | 100% | 97% | 78% | 97% | 80% | 20% |
| CANVAS | `LASP:len84` | 652 / 43 | 2 | 100% | 100% | 67% | 100% | 100% | 100% | 100% | – |
| InnoSat-16 | `len72:b1=0x98` | 600 / 49 | 4 | 0% | 0% | – | – | – | – | – | – |
| **total** | | | **126** | **66%** | **40%** | **98%** | **92%** | **73%** | **85%** | **83%** | **36%** |
<!-- /HELDOUT_TABLE -->

Column definitions:
- *Recall*: share of the decoder's physical fields (temperature, power, clock, counter, navigation,
  magnetometer) that were found.
- *Exact*: the found reading reproduces the decoder's raw values bit for bit, so offset, width,
  byte order and sign are all right.
- *Precision*: share of reported fields that are real decoder fields.
- *Claude*: Claude's blind category matches the decoder's. An "unknown" answer counts as wrong.
- *Answered*: share of fields Claude did not mark unknown. *When answered*: its accuracy on those.
- *Battery / solar / bus*: the finer power class, scored where the decoder's field name says which.

Frame types are skipped when they have fewer than 20 passes; those are usually stored-telemetry
dumps that cannot be tied to the orbit. Rows marked ¹ had no eclipse in the hours before any frame,
so only counters, clocks and GNSS can be recovered; they are shown but not counted in the totals.

## When the obvious approach fails

**No eclipses? Go back to when there were, and don't fool yourself doing it.** A satellite in a
dawn-dusk orbit can spend months in permanent sunlight, and then nothing changes for the physics to
explain. Its orbit plane still turns relative to the Sun over the year. So Shadowfield computes the
satellite's **eclipse calendar** from its TLE and fetches SatNOGS recordings from the weeks when it
*did* pass through Earth's shadow (`fetch NORAD --eclipse-season`). For AlSat-1N (beta 76-81° in
November) the calendar finds eclipses from May to early August, of up to 20 minutes per orbit.

Mixing seasons is a trap, and Claude caught it first. With NetSat's eclipse season added to its
autumn data, almost every byte suddenly "followed sunlight" with the same 10-minute lag. Asked to
name the fields, Claude refused: they *"move together and share the same lag... the sun correlations
are probably artefacts of sampling order"*. Anything that drifted between June and November looked
like a response to eclipses. Shadowfield now:
- compares within weeks;
- takes sunlight correlations only from weeks where the sunlight history actually varies (counters,
  clocks, GNSS and orbit phase still use every frame);
- notes when the shadow fell out of view of the ground stations (over a polar cap), where only the
  sunlight *history* differs.

**"Encrypted"? Usually just whitened.** Amateur-band telemetry may not be encrypted to hide its
meaning (ITU Radio Regulations Art. 25). What makes a frame look random is usually a scrambler.
Shadowfield tries the standard layers, but only after each passes a structural test, so clean
frames are never "peeled":
- CCSDS pseudo-randomizer and any fixed whitening key: the key must make several byte columns
  clearly more physical than leaving them alone; an unknown key is recovered byte by byte from the
  physics.
- G3RUH self-synchronising scrambler: per-column entropy must drop clearly.

Real ciphertext is recognised by its uniform byte statistics and reported as such, never "decoded".
Controlled test on real UWE-4 frames (`python -m tests.peel_test`):

| layer applied to the frames | fields before | after peeling |
|---|---|---|
| CCSDS randomizer | 13 weak or wrong | 16, identical to the clean frames |
| G3RUH scrambler | 0 | 16, identical to the clean frames |
| unknown random 58-byte key | 12 weak or wrong | 15, recall 87% |
| per-frame random bytes (AES-like) | 0 | flagged as encrypted, no false fields |

## How it works

```
SatNOGS frames ──► strip AX.25, dedupe across stations, split frame types (two levels of type bytes)
TLE per pass ───► SGP4 + Sun + Earth shadow ──► sunlit, 200-min sunlight history (lags 5–80 min), minutes
                                                 since sunrise/sunset, orbit phase, beta, ECI/ECEF r and v, IGRF |B|
every reading (offset × width 1–4 × byte order × sign, and float32)
   ──► rank correlation with the physics, within weeks, significance counted in passes (not frames)
   ──► counters/clocks (judged on frames >5 s apart: station clocks disagree), GNSS, magnetometer, quaternion
   ──► byte order (smoothness; the frame's proven byte order), width (carry, sign extension, wrap point,
       low-byte continuity), canonical widths (constant top bytes, counter top bytes that carry)
evidence ──► Claude (Sonnet 5.5, blind, three samples, majority) ──► names, units, scales, health note
         ──► HTML report + Kaitai .ksy draft, graded against satnogs-decoders when one exists
```

Things that had to be solved along the way:
- Frames heard by several stations are deduplicated, and passes are rebuilt from time gaps.
- Database frames carry no orbit, and low satellites drift several km a day under drag, so each
  frame gets the historical TLE closest to its time.
- A signed current is split by naive methods into a "sign byte" and a "low byte". Shadowfield detects
  the 0xFF/0x00 sign extension, and uses the frame's proven byte order to decide which neighbour a
  shared byte belongs to.
- GNSS receivers repeat stale fixes or send -1 when they have none, and ground stations cluster over
  Europe. A position field must match the orbit within 1% on a quarter of its frames, sweeping most
  of the coordinate's range.

## Tested and validated

- `python -m pytest` runs 37 tests:
  - Unit tests for decoding, the structure tests, peeling, grading and voting.
  - A **random synthetic fleet**. Each satellite gets its own byte order, field order, gaps and traps
    (signed currents, counters with constant top bytes, rolling sequence numbers, status and noise
    bytes, stale GNSS fixes, float32s), so rules are tested in combination, not one at a time
    (`python -m tests.fleet`: 99% recall, 91% exact, no false fields over 12 satellites).
  - A **regression lock** on every real development frame type: a rule change that lowers recall,
    exact layout or precision anywhere fails the suite.
- **Every generated `.ksy` compiles with the official Kaitai Struct compiler** (its JavaScript build,
  `tools/ksc`). The compiled parser reads the real frames exactly as the engine does (`tests/test_ksy.py`).
- The engine is a package of small steps (`shadowfield/engine`: readings, motion, correlation,
  classification, detectors, structure tests, selection, evidence). Splitting it out of one file was
  checked against a golden snapshot of every field on every satellite: identical.
- The fast physics model agrees with Skyfield's full-precision shadow on 99.7% of 10-second samples
  (`python -m tests.check_physics`).
- Grading locates each decoder field by exact value equality, or by a linear scale (r > 0.9995) when the
  decoder converts units. Fields the decoder exposes but the bytes cannot reproduce are left out, not
  guessed.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; .venv/bin on Linux/macOS
cp .env.example .env        # ANTHROPIC_API_KEY, optional SATNOGS_DB_TOKEN
python -m shadowfield fetch 43880 --eclipse-season    # frames + per-pass TLEs from SatNOGS Network
python -m shadowfield run 43880 --decoder uwe4        # find, name, grade -> results/43880.html, results/ksy/
python -m shadowfield bench                           # blind scoreboard -> results/index.html
(cd tools/ksc && npm install) && python -m pytest     # tests, including compiling the .ksy drafts
```

Claude cost is about $0.05-0.15 per frame type: Sonnet 5.5 at medium effort, three samples sharing one
cached prompt. Answers are cached on disk, so re-runs are free. `SHADOWFIELD_MODEL`,
`SHADOWFIELD_EFFORT` and `SHADOWFIELD_VOTES` override the defaults.

## Related work

The idea of reading telemetry against eclipses is old craft. Daniel Estévez (gr-satellites) has plotted
unidentified LES-5 fields around an eclipse to guess which are voltages or temperatures
([LES-5](https://destevez.net/?p=7944)). Shadowfield automates and grades that craft.

The closest automated method is in cars. **LibreCAN**, **CANClassify** and **CAN-D** label CAN-bus
signals by correlating them with independently recorded physical values (OBD-II, IMU), and CAN-D also
recovers byte order and sign ([LibreCAN](https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2019/fv075-peseA.pdf),
[CANClassify](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-151.html),
[CAN-D](https://arxiv.org/pdf/2006.05993)). They need a second sensor recording; Shadowfield gets its
ground truth for free from orbital mechanics.

General protocol reverse engineering finds field boundaries and types without physical context. Examples:
- [BinaryInferno](https://www.ndss-symposium.org/wp-content/uploads/2023-131-paper.pdf)
- [NETPLIER](https://www.cs.purdue.edu/homes/ye203/pub/NDSS21a.pdf)
- [BinPRE](https://arxiv.org/html/2409.01994v1)
- [SEMPRE](https://arxiv.org/pdf/2610.31844), which asks an LLM for field-type hypotheses

SatNOGS decoders are written by hand, sometimes from CSV with helper scripts.

## Limits

- It needs a few hundred frames over tens of passes.
- Fields with no physical driver (status bits, CRCs, IDs, settings, angular rates) are invisible to it by
  design. So are bit-packed fields narrower than a byte.
- Sunlight-driven fields need eclipses in the hours before the frames. The eclipse calendar finds them
  where the orbit has a season, but a satellite heard only in permanent sunlight keeps only its counters,
  clocks and GNSS.
- Real encryption or compression is detected and reported, not decoded.
- On satellites it has never seen, the engine finds two thirds of the physical fields with almost no
  false ones, but gets the exact byte layout of only about 40%. The finer power class (battery vs
  solar vs bus) reaches 77% on development satellites but only 36% on the held-out ones.
- Claude's names are a draft. Its categories are measured, its exact field names are not.
- Precision is measured against the decoder's fields, so on a satellite whose decoder covers only part
  of the frame (Kostka) real but undecoded fields count as false.

## Built with AI

Built for *Build Your First AI App* with Claude Code (Claude Opus) as a pair programmer. At runtime the app
uses the Claude API (Sonnet 5.5) to name and interpret fields. The physics, statistics, grading and the
decoder generator are plain Python (sgp4, numpy, scipy, ppigrf).

Data: [SatNOGS](https://satnogs.org) Network and DB (CC BY-SA). Ground truth for grading: satnogs-decoders.
