# Shadowfield

**Decode a satellite's telemetry without its decoder, using the orbit it flies.**

Hundreds of small satellites broadcast housekeeping beacons that amateur ground stations
(SatNOGS) record every day. Many of those beacons have no public format: they are just bytes.
Shadowfield works out what the bytes mean from physics alone.

It needs only the raw frames and the satellite's orbit (TLE). From the orbit it computes,
for every frame, whether the spacecraft was in sunlight, how long ago it crossed into or out
of Earth's shadow, its orbital phase, its altitude, and the Earth's magnetic field at its
position. Then it tests every possible reading of the frame bytes against that physics:

| behaviour in the bytes | what it means |
|---|---|
| steps when the satellite enters or leaves Earth's shadow | solar or charge current, battery voltage |
| follows sunlight with a lag of minutes | temperatures, state of charge |
| only increases, at a fixed rate | uptime, packet counters |
| tracks wall-clock time | on-board clock (epoch detected) |
| three values whose vector length follows the IGRF field model | 3-axis magnetometer |

This finds where each field starts, how wide it is, its byte order and its signedness.
**Claude** then reads that evidence like a spacecraft engineer would: the sunrise/sunset
response shape, the time constant, the raw ranges and which fields move together. From that
it proposes a name, a unit and a scaling for each field. It is never told the satellite's name,
so it cannot recall a known decoder.

## Does it work? A blind test against real decoders

Some satellites do have public decoders ([satnogs-decoders](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders)).
We run Shadowfield on their raw frames as if they did not, and only afterwards use the decoder to
grade the result.

| satellite | frame type | frames / passes | decoder fields | recall | exact | precision | Claude |
|---|---|---|---|---|---|---|---|
| UWE-4 | `DP0UWH:len58` | 1105 / 411 | 15 | 100% | 93% | 100% | 87% |
| Geoscan | `len72:b0=0x84` | 1476 / 103 | 15 | 60% | 60% | 100% | 100% |
| Geoscan | `len72:b0=0x04:b3=0x05` | 321 / 99 | 14 | 50% | 36% | 100% | 71% |
| GRBBeta | `len79:b0=0x85` | 347 / 44 | 23 | 70% | 39% | 100% | 100% |
| SNUGLITE | `len97` | 1005 / 126 | 7 | 100% | 71% | 88% | 100% |
| SNUGLITE | `len214` | 405 / 108 | 30 | 90% | 77% | 93% | 89% |
| NetSat | `len89` | 1724 / 241 | 14 | 100% | 36% | 84% | 14% |
| AlSat-1N | `len19` | 217 / 74 | 4 | 25% | 25% | 100% | 100% |
| AlSat-1N ¹ | `len20` | 678 / 136 | 1 | 100% | 100% | 50% | 100% |
| AlSat-1N ¹ | `len32` | 623 / 126 | 5 | 60% | 40% | 100% | 100% |
| **total (counted rows)** | | | **122** | **79%** | **58%** | **96%** | **83%** |

¹ No eclipses in the hours before any frame (beta angle above 69°), so only counters and clocks can be recovered. Shown, but not counted in the totals. Frame types with fewer than 20 passes (stored-telemetry dumps) are skipped and not listed.

*Recall*: share of the decoder's physical fields (temperature, power, clock, counter, magnetometer)
that were found. *Exact*: found at exactly the decoder's byte offset and width.
*Precision*: share of reported fields that are real decoder fields. *Claude*: Claude's blind
category for the field matches the decoder's.

## When the obvious approach fails

**No eclipses? Go back to when there were, and don't fool yourself doing it.** A satellite in a
dawn-dusk orbit can spend months in permanent sunlight, and then nothing changes for the physics to
explain. Its orbit plane still turns relative to the Sun over the year, so Shadowfield computes the
satellite's **eclipse calendar** from its TLE and fetches SatNOGS recordings from the weeks when it
*did* pass through Earth's shadow (`fetch NORAD --eclipse-season`). For AlSat-1N (beta 76-81° in
November) the calendar finds eclipses from May to early August, of up to 20 minutes per orbit.

Mixing seasons is a trap, and Claude caught it first. With NetSat's eclipse season added to its
autumn data, almost every byte suddenly "followed sunlight" with the same 10-minute lag. Asked to
name the fields, Claude refused: they *"move together and share the same lag... the sun
correlations are probably artefacts of sampling order"*. It was right. Anything that drifted
between June and November looked like a response to eclipses. Shadowfield now:
- analyses separate seasons separately;
- correlates within weeks, so slow drift cannot pose as physics;
- notes when the shadow fell out of view of the ground stations (over a polar cap), where only the
  sunlight *history* differs.

With these controls the honest gain on AlSat-1N is small: about 200 eclipse-season frames is too few.
The point is that the tool tells you so instead of inventing fields.

**"Encrypted"? Usually just whitened.** Amateur-band telemetry may not be encrypted to hide its
meaning (ITU Radio Regulations Art. 25). What makes a frame look random is usually a scrambler.
Shadowfield tries the standard layers and keeps the one that makes physics appear:
- CCSDS pseudo-randomizer
- G3RUH self-synchronising scrambler
- **any fixed whitening key**, recovered byte by byte: only the right key makes a byte column follow
  sunlight or count upward

Real ciphertext is recognised by its uniform byte statistics and reported as such, never "decoded".
Controlled test on real UWE-4 frames (`python -m tests.peel_test`):

| layer applied to the frames | fields before | after peeling |
|---|---|---|
| CCSDS randomizer | 13 weak or wrong | 16 (identical to the clean frames) |
| G3RUH scrambler | 0 | 16 (identical to the clean frames) |
| unknown random 58-byte key | 12 weak or wrong | 14, recall 87%, 9 key bytes recovered |
| per-frame random bytes (AES-like) | 0 | flagged as encrypted, no false fields |

**Outputs people can use:**
- A Kaitai Struct decoder draft (`results/ksy/*.ksy`), in the format satnogs-decoders uses, with every
  field's evidence in its doc string.
- A short **health note** per frame type: reboots from counter resets, and drift of battery and
  thermal daily lows and highs, written up by Claude in the same call that names the fields.

## How it works

```
SatNOGS frames ──► strip AX.25, dedupe, split frame types
TLE per pass ───► SGP4 + Sun + Earth shadow ──► sunlit, 200-min sunlight history (lags 5–80 min),
                                                 minutes since sunrise/sunset, orbit phase, beta, IGRF |B|
every byte reading (offset × width 1–4 × endian × sign) ──► rank correlation with the physics,
                                                 significance counted in passes (not frames)
                                                 ──► pick byte order (smoothness), width (carry / sign-extension
                                                     / wrap tests), non-overlapping set
fields + response-shape evidence ──► Claude (Sonnet 5.5, structured output, blind) ──► names, units, scaling
                                                 ──► HTML report, graded against the decoder when one exists
```

Things that had to be solved along the way:
- Frames heard by several stations at once are deduplicated, and passes are rebuilt from time gaps.
  Significance uses the number of passes, because frames inside one pass are not independent.
- Database frames carry no orbit, and low satellites drift several km per day under drag, so each
  frame gets the historical TLE closest to its time.
- Ground stations often hear a satellite only at a few orbital phases (sun-synchronous orbits
  over Europe), so the response profile adapts to whatever minutes-since-sunrise the data covers.
- A signed current is split by naive methods into a "sign byte" and a "low byte". Shadowfield
  detects the 0xFF/0x00 sign extension, and uses the frame's byte order to decide which
  neighbour the sign byte belongs to.

The fast physics model agrees with Skyfield's full-precision shadow calculation on 99.7% of
10-second samples (about 8 s per shadow transition; `python -m tests.check_physics`). A synthetic
satellite with seven planted fields is recovered 7/7 with no false fields (`python -m tests.synth`).

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin on Linux/macOS
cp .env.example .env        # ANTHROPIC_API_KEY, optional SATNOGS_DB_TOKEN
python -m shadowfield fetch 43880            # raw frames + per-pass TLEs from SatNOGS Network
python -m shadowfield run 43880 --decoder uwe4   # find, name, grade -> results/43880.html
python -m shadowfield bench                  # blind scoreboard -> results/index.html
```

One Claude call per frame type, about $0.05 with Sonnet 5.5 at medium effort. Answers are cached
on disk, so re-runs are free (`SHADOWFIELD_MODEL` / `SHADOWFIELD_EFFORT` override).

## Limits

- It needs a few hundred frames over tens of passes. Fewer passes do not reach significance.
- Sunlight-driven fields need eclipses within the hours before the frames, found automatically via the eclipse calendar where the satellite has one.
- Fields with no physical driver (status bits, CRCs, IDs, settings) are invisible to it by design.
- Real encryption or compression is detected and reported, not decoded.
- Clocks wider than 4 bytes are found as their lower 4 bytes.

## Built with AI

Built for *Build Your First AI App* with Claude Code (Claude Opus) as a pair programmer. At runtime
the app itself uses the Claude API (Sonnet 5.5) to name and interpret fields. The physics,
statistics and evaluation are plain Python (sgp4, numpy, scipy, ppigrf).

Data: [SatNOGS](https://satnogs.org) Network and DB (CC BY-SA). Ground truth: satnogs-decoders.
