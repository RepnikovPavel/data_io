# synth

**Script:** `pipe_clustered/clean_SYNTH.py`

## Purpose

PleIAs SYNTH: a large synthetic instruction dataset. Kept English-only; self-knowledge queries and cooking exercises filtered out. The condition tag is derived from the exercise type.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/PleIAs/SYNTH/*.parquet`
- Format: parquet files

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `query` | string | the prompt (constraints appended for rag exercises) |
| `constraints` | string | extra requirements (rag exercises only) |
| `synthetic_answer` | string | generated answer -> response |
| `exercise` | string | exercise type; drives the condition tag and the 'cooking' filter |
| `language` | string | filter: only 'en' kept |
| `query_seed_url` | string | filter: 'Pleias self-knowledge' rows dropped |

## After (transformed)

- Location: `data_clustered/synth/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 500 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 60,934,844

`condition` values used here:

- `synth,cot` — creative writing / rag / memorization / constrained writing / editing
- `synth,direct` — math mcq / mcq
- `synth,noisy,cot` — math exercise

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/PleIAs/SYNTH/synth_001.parquet`)

````text
{
  "query": "Wie genau interagieren die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind mit der Topographie der japanischen Inseln, um diese speziellen Kármán-Wirbelwolken im Meer zu erzeugen?",
  "constraints": "",
  "synthetic_answer": "Die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind spielen eine entscheidende Rolle bei der Entstehung der charakteristischen Kármán-Wirbelwolken über dem Japanischen Meer.\n\nIm Winter bringt der starke nordwestliche Monsun mit Geschwindigkeiten von 12-15 m/s kalte, trockene kontinentale Luft aus Richtung Sibirien. Diese Luftmasse führt zu markanten Temperaturunterschieden zwischen den kalten nördlichen Gebieten mit durchschnittlichen Lufttemperaturen von -20°C im Januar/Februar und den wärmeren südlichen Regionen mit etwa 5°C. Der Wintermonsun verstär\n… [truncated, 1686 chars total]",
  "exercise": "memorization",
  "language": "de",
  "query_seed_url": "https://en.wikipedia.org/wiki/Sea_of_Japan"
}
````

### Transformed record (`data_clustered/synth/synth_001.parquet`, record 1)

````text
{
  "instruction": "hey um can someone tell me why the decagon has like so many different ways to look the same but not completely the same?? like i see there's dih and z and stuff but how does that work with the big one that looks normal??",
  "condition": "synth,cot",
  "response": "You're right to notice that the decagon has multiple ways to look similar but not identical! This happens because of its rich symmetry structure.\n\nThe regular decagon has what's called Dih10 symmetry with order 20, meaning there are 20 different ways you can rotate or reflect it and still have it look the same. But within this, there are several smaller symmetry groups that work differently.\n\nThe decagon contains 8 distinct symmetry types that can be seen in various ways on the shape. These symmetries can be broken down into different categories based on whether they pass through vertices (d f\n… [truncated, 1518 chars total]"
}
````

### Transformed record (`data_clustered/synth/synth_001.parquet`, record 2)

````text
{
  "instruction": "Which animal has more poison - the salamander that sticks out its bone or the frog with the sharp head thing, and how do they both make their enemies hurt?",
  "condition": "synth,cot",
  "response": "Both animals you're describing have developed effective venom delivery systems, though their methods are quite different.\n\nThe salamander with the sticking-out bone has sharp venom-tipped ribs that it can extrude when threatened. These ribs deliver venom directly to whatever attacks it. The frog from Brazil has tiny spines around its skull crown that act like tiny needles - when something hits the frog, the spines can inject venom into the attacker.\n\nBoth animals make enemies hurt through their venomous secretions. Their skin produces toxic mucus that's harmful when ingested or when it comes i\n… [truncated, 1112 chars total]"
}
````
