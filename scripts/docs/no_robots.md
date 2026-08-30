# no_robots

**Script:** `pipe/clean_no_robots.py`

## Purpose

Human-written instruction-following conversations (HuggingFaceH4/no_robots). Only the first user->assistant turn is kept; an optional system prompt is prepended to the instruction. Broad human-quality SFT data.

## Before (raw storage)

- Source: HF `HuggingFaceH4/no_robots` (all splits)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `messages` | list[{role, content}] | conversation turns; only the optional system message + first user/assistant pair are read |

## After (transformed)

- Location: `data/no_robots.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 10,000

`condition` values used here:

- `cot` — free-form human-written response (tagged cot by convention)

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `HuggingFaceH4/no_robots`, split `train`)

````text
{
  "messages": [
    {
      "content": "Please summarize the goals for scientists in this text:\n\nWithin three days, the intertwined cup nest of grasses was complete, featuring a canopy of overhanging grasses to conceal it. And decades later, it served as Rinkert’s portal to the past inside the California Academy of Sciences. Information gleaned from such nests, woven long ago from species in plant communities called transitional habitat, could help restore the shoreline in the future. Transitional habitat has nearly disappeared from the San Francisco Bay, and scientists need a clearer picture of its original species composition—whic\n… [truncated, 1055 chars total]",
      "role": "user"
    },
    {
      "content": "Scientists are studying nests hoping to learn about transitional habitats that could help restore the shoreline of San Francisco Bay.",
      "role": "assistant"
    }
  ]
}
````

### Transformed record (`data/no_robots.jsonl`, record 1)

````text
{
  "condition": "cot",
  "instruction": "Please summarize the goals for scientists in this text:\n\nWithin three days, the intertwined cup nest of grasses was complete, featuring a canopy of overhanging grasses to conceal it. And decades later, it served as Rinkert’s portal to the past inside the California Academy of Sciences. Information gleaned from such nests, woven long ago from species in plant communities called transitional habitat, could help restore the shoreline in the future. Transitional habitat has nearly disappeared from the San Francisco Bay, and scientists need a clearer picture of its original species composition—whic\n… [truncated, 1055 chars total]",
  "response": "Scientists are studying nests hoping to learn about transitional habitats that could help restore the shoreline of San Francisco Bay."
}
````

### Transformed record (`data/no_robots.jsonl`, record 2)

````text
{
  "condition": "cot",
  "instruction": "Help write a letter of 100 -200 words to my future self for Kyra, reflecting on her goals and aspirations.",
  "response": "Dear Future Self,\n\nI hope you're happy and proud of what you've achieved. As I write this, I'm excited to think about our goals and how far you've come. One goal was to be a machine learning engineer. I hope you've worked hard and become skilled in this field. Keep learning and innovating. Traveling was important to us. I hope you've seen different places and enjoyed the beauty of our world. Remember the memories and lessons. Starting a family mattered to us. If you have kids, treasure every moment. Be patient, loving, and grateful for your family.\n\nTake care of yourself. Rest, reflect, and ch\n… [truncated, 901 chars total]"
}
````
