# HC - Обратимый трансформер с памятью

Первый обратимый трансформер, который объединяет:
- 12 обратимых блоков (INN)
- 16-head механизм внимания
- Встроенную память (как в LSTM)
- Асинхронное управление 4 роторами

## Демо
https://github.com/AleksMG/HC/

## Что внутри
- Полная обратимость (что зашифровали — то расшифруется)
- Каждый ротор получает уникальный сигнал управления
- Память сохраняется между шагами

## Лицензия
MIT — делайте что хотите, но ссылайтесь на автора.

## Дата публикации
17 февраля 2026 — первый публичный релиз






# ⚡ HC — Hybrid Cryptographic Transformer

**Reversible Neural Architecture with Attention & Memory**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 What

Reversible neural network (338 dim, 16-head attention, 30% memory).

Built for crypto, works for: robotics, RL, science simulation, generative models, causal inference.

---

## 🏗️ Architecture

```
338 dim State → 12 Reversible Blocks → Driver → 4 Rotors
```

| | |
|---|---|
| Dimensions | 338 |
| Attention | 16 heads |
| Blocks | 12 (coupling layers) |
| Memory | 102 dim |
| Reversible | ✅ 100% |

---

## 🎯 Use

```javascript
// Crypto
cipher = encrypt(text)
plain = decrypt(cipher) // exact

// Robotics / RL
action = forward(state)
prev = inverse(state) // exact rollback
```

---

## 🔬 Why

Coupling layers guarantee reversibility (Jacobian = 1). Attention + memory enable flexible decisions.

---

## 📄 License

MIT. Prior art: GitHub 17.02.2026.
