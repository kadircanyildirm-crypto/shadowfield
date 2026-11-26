# Demo video script (about 2:30, under 3 minutes)

The **narration** is in English, written in short sentences so it is easy to read aloud. The **Ekran** lines tell you what to show on screen.

## Hazırlık (5 dakika)

1. `docs/index.html` dosyasını Chrome'da aç. Tam ekran için F11'e bas, yakınlaştırma %100 olsun (Ctrl+0).
2. Sayfayı en üste kaydır ve 5 saniye bekle; dünya ve uydular yüklensin.
3. Kayıt için iki seçenek var:
   - Windows'ta **Win+Alt+R** (Xbox Game Bar). Ayarlardan mikrofonu aç.
   - OBS, 1920×1080.
4. Bir kere sesli prova yap. Yavaş oku; cümleler arası duraklama sorun değil.
5. Videoyu YouTube'a **Liste dışı (Unlisted)** olarak yükle, linki Devpost'a yapıştır.

Konuşmak istemezsen narration metnini YouTube'a altyazı olarak ekleyebilirsin, ya da kaydı sessiz yapıp üstüne yazı koyabilirsin.

---

## 0:00 – 0:15 · The problem

**Ekran:** Sayfanın en üstü. Dünya dönüyor, uydular etrafında. Hiçbir şeye tıklama.

> Hundreds of small satellites are sending us data right now.
> Volunteers record it every day with their own ground stations.
> But for many of these satellites, nobody can read it. There is no public format. Just bytes.

## 0:15 – 0:35 · The idea

**Ekran:** Fareyi sağ üstteki "15 SATELLITES · N IN EARTH'S SHADOW" yazısının üstüne götür. Sonra bir uydunun gölgeye girip griye dönmesini bekle.

> Shadowfield reads those bytes through physics.
> Every satellite's orbit is public. From the orbit, we know exactly when it was in sunlight, and when it was in Earth's shadow.
> Batteries charge in the sun. Panels cool in the dark. Clocks tick.
> Those fingerprints are in the bytes.

## 0:35 – 1:15 · A real frame

**Ekran:** "Decrypt a frame" butonuna tıkla. Baytlar renklenirken bekle (UWE-4 seçili).

> This is a real frame from UWE-4, a German university CubeSat, exactly as ground stations received it.
> Shadowfield tries every way to read these bytes: every position, width, byte order and sign.
> Each reading is tested against the orbit. The ones that follow physics become fields.
> Orange are temperatures. Green is power. Grey are counters.

**Ekran:** Soldaki listede **"Panel temperature 1"** satırına tıkla. Sağdaki grafik açılınca fareyle gölge/güneş sınırını göster.

> This byte falls in Earth's shadow and climbs again in sunlight, about ten minutes behind.
> That is a solar panel temperature.

## 1:15 – 1:45 · Claude

**Ekran:** Sağ paneli yavaşça aşağı kaydır: "Claude's reading", kanıt satırları, güven çubuğu ve en alttaki yeşil ✓.

> Then Claude reads the evidence the way a spacecraft engineer would: the response to sunrise, the time constant, the range.
> It names the field, its unit and its scale.
> Claude never sees the satellite's name or its decoder, so it cannot remember an answer. We ask three times and keep the majority.
> Here it matches the published decoder exactly. And we only checked that afterwards.

## 1:45 – 2:00 · Second satellite

**Ekran:** Yukarıdaki **SNUGLITE** butonuna tıkla, animasyon bitsin. Listede **"GNSS pos ECEF x"** satırına tıkla.

> On SNUGLITE, a Korean satellite, it even finds the GPS position.
> This field is the orbit's own coordinate, in centimetres.

## 2:00 – 2:20 · Proof

**Ekran:** Aşağı kaydırıp "Graded blind against real decoders" bölümünde dur. Sayılar animasyonla dolsun.

> We graded it blind against real decoders.
> On six satellites we never looked at while building it, ninety-eight percent of the fields it reports are real.
> It finds two thirds of the physical fields. The exact byte layout is right forty percent of the time. We show that too.

## 2:20 – 2:35 · Output and close

**Ekran:** En alttaki ".ksy" kod bölümüne kaydır. Son 3 saniyede tekrar en üste çık ve dünyayı göster.

> The output is a decoder file in Kaitai Struct, the format SatNOGS uses.
> It compiles with the official compiler and reads the real frames.
> Shadowfield. The orbit is the decoder.

---

## Kontrol listesi

- [ ] Video 3 dakikanın altında.
- [ ] Ses net, arka plan sessiz.
- [ ] YouTube'da Unlisted, link Devpost'ta.
- [ ] Ekranda şifre, mail ya da başka sekme görünmüyor.
