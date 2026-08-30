# no_robots

↑ [index](README.md) · ← [natural_reasoning](natural_reasoning.md) · [next → numinamath](numinamath.md)

**Script:** `pipe/clean_no_robots.py`

## Purpose

No Robots (HuggingFace): 10k instructions written by human annotators (not GPT), modelled after InstructGPT, across 10 categories (Generation, Open QA, ...). Only the first user->assistant turn is kept (system prompt prepended).

## Before (raw storage)

- Source: HF `HuggingFaceH4/no_robots` (splits `train`+`test`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `messages` | arrow list — arrow struct {content: UTF-8 text (arrow `string`), role: UTF-8 text (arrow `string`)} | conversation turns; only the optional system message + first user/assistant pair are read |

## After (transformed)

- Location: `data/no_robots.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 10,000

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `cot` | 10,000 | free-form human-written response (tagged cot by convention) | human-quality SFT signal for general instruction following |

_exact counts (full scan of the jsonl file)_

## Keyword: `category` (input)

Human-assigned instruction category (from the dataset card taxonomy).

| value | rows |
|---|---|
| `Generation` | 4,346 |
| `Open QA` | 1,182 |
| `Brainstorm` | 1,060 |
| `Chat` | 796 |
| `Rewrite` | 625 |
| `Summarize` | 395 |
| `Coding` | 334 |
| `Classify` | 334 |
| `Closed QA` | 245 |
| `Extract` | 183 |

_exact counts (all 9,500 rows)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `HuggingFaceH4/no_robots`, split `train`

One row of HF `HuggingFaceH4/no_robots`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| messages |
|---|
| [{"content": "Please summarize the goals for scientists in this text:\n\nWithin three days, the intertwined cup nest of grasses was complete, featuring a canopy of overhanging grasses to conceal it. And decades later, it served as Rinkert’s portal to the past inside the California Academy of Sciences. Information gleaned from such nests, woven long ago from species in plant communities called transitional habitat, could help restore the shoreline in the future. Transitional habitat has nearly disappeared from the San Francisco Bay, and scientists need a clearer picture of its original species composition—whic… [truncated, 1055 chars total]", "role": "user"}, {"content": "Scientists are studying nests hoping to learn about transitional habitats that could help restore the shoreline of San Francisco Bay.", "role": "assistant"}] |

### Raw — `category` = `Generation`

One raw row with this value (same storage as above), shown as a table:

| messages | category |
|---|---|
| [{"content": "Help write a letter of 100 -200 words to my future self for Kyra, reflecting on her goals and aspirations.", "role": "user"}, {"content": "Dear Future Self,\n\nI hope you're happy and proud of what you've achieved. As I write this, I'm excited to think about our goals and how far you've come. One goal was to be a machine learning engineer. I hope you've worked hard and become skilled in this field. Keep learning and innovating. Traveling was important to us. I hope you've seen different places and enjoyed the beauty of our world. Remember the memories and lessons. Starting a family mattered to us. If you have kids, treasure every moment. Be patient, loving, and grateful for your family.\n\nTake care of yourself. Rest, reflect, and ch… [truncated, 901 chars total]", "role": "assistant"}] | Generation |

### Raw — `category` = `Open QA`

One raw row with this value (same storage as above), shown as a table:

| messages | category |
|---|---|
| [{"content": "Why is Aberdeen called the Granite City?", "role": "user"}, {"content": "Alberdeen's nickname, the Granite City, comes from the city's history and contribution of granite, in addition to the architectural style of more than fifty percent of the city's buildings. Many of the city's most popular buildings and residential properties were built using granite from large quarries around the Nort-East, most notably the Rubislaw Quarry in Aberdeen's west end. As technology and practices for handling a large amount of rock improved, Aberdeen quickly became the granite capital of the world, exporting granite by sea to several areas globally. ", "role": "assistant"}] | Open QA |

### Transformed — condition=`cot`

One line of `data/no_robots.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "cot", "instruction": "Please summarize the goals for scientists in this text:\n\nWithin three days, the intertwined cup nest of grasses was complete, featuring a canopy of overhanging grasses to conceal it. And decades later, it served as Rinkert’s portal to the past inside the California Academy of Sciences. Information gleaned from such nests, woven long ago from species in plant communities called transitional habitat, could help restore the shoreline in the future. Transitional habitat has nearly disappeared from the San Francisco Bay, and scientists need a clearer picture of its original species composition—whic… [truncated, 1055 chars total]", "response": "Scientists are studying nests hoping to learn about transitional habitats that could help restore the shoreline of San Francisco Bay."}
````

---

↑ [index](README.md) · ← [natural_reasoning](natural_reasoning.md) · [next → numinamath](numinamath.md)
