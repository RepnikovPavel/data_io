# synth

↑ [index](README.md) · ← [flan](flan.md) · [next → dmmath](dmmath.md)

**Script:** `pipe_clustered/clean_SYNTH.py`

## Purpose

SYNTH (PleIAs): ~80M synthetic samples amplified from ~59k Wikipedia 'vital articles', with model-written reasoning traces; ~20% non-English. The transform keeps English only, drops self-knowledge queries and cooking exercises, and derives the condition tag from the exercise type.

Created: 2025-04 · Domain: synthetic general knowledge (Wikipedia-derived) _(date source: PleIAs SYNTH release (blog announcement 2025-04))_

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/PleIAs/SYNTH/*.parquet`
- Storage: 500 parquet files (synth_001.parquet ...), one row per sample

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `query` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt (constraints appended for rag exercises) |
| `constraints` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | extra requirements (rag exercises only) |
| `synthetic_answer` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | generated answer -> response |
| `exercise` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | exercise type; drives the condition tag and the 'cooking' filter |
| `language` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | filter: only 'en' kept |
| `query_seed_url` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | filter: 'Pleias self-knowledge' rows dropped |

## After (transformed)

- Location: `data_clustered/SYNTH/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 500 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 60,934,844

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | the prompt |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | comma-separated tags (see keyword table below) |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `synth,cot` | 58,586,918 | creative writing / rag / memorization / constrained writing / editing | synthetic reasoning-trace supervision |
| `synth,direct` | 2,347,926 | math mcq / mcq | synthetic answer-only MCQ supervision |

_exact counts (condition column read in full)_

## Keyword: `exercise` (input)

Exercise type — the field that decides the condition tag (and whether the row is dropped).

| value | rows | meaning | why it matters |
|---|---|---|---|
| `memorization` | 422,558 | recall facts from the seed article | mapped to synth,cot; by far the largest slice |
| `mcq` | 9,657 | multiple-choice question | mapped to synth,direct |
| `constrained writing` | 7,796 | write text under explicit constraints | mapped to synth,cot |
| `math exercise` | 7,458 | open-ended math problem | mapped to synth,noisy,cot in the script, but every such row has a null query_seed_url and is dropped by the filter — 0 rows in the output |
| `rag` | 4,778 | retrieval-augmented QA (constraints appended to the query) | mapped to synth,cot |
| `math mcq` | 4,428 | multiple-choice math | mapped to synth,direct |
| `creative writing` | 3,888 | stories/poems etc. | mapped to synth,cot |
| `editing` | 3,523 | rewrite/improve a text | mapped to synth,cot |
| `cooking` | 2,235 | recipe exercises | DROPPED by the transform (no output rows) |

_estimated from the first 3 of 500 files (466,321 rows)_

## Keyword: `language` (input)

Sample language; only `en` survives the transform (~80% of the raw data per the dataset card).

| value | rows |
|---|---|
| `en` | 376,731 |
| `de` | 14,830 |
| `es` | 14,822 |
| `fr` | 14,755 |
| `pl` | 14,733 |
| `it` | 14,705 |
| `nl` | 7,496 |
| `la` | 7,449 |
| `None` | 630 |
| `pt` | 134 |
| `uk` | 12 |
| `ru` | 5 |

… 26 distinct values total, top 12 shown.

_estimated from the first 3 of 500 files (466,321 rows)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/PleIAs/SYNTH/synth_001.parquet`

One row of `/mnt/hdd2/datasets_text/PleIAs/SYNTH/synth_001.parquet` (parquet table), shown with the columns the transform reads:

| query | constraints | synthetic_answer | exercise | language | query_seed_url |
|---|---|---|---|---|---|
| Wie genau interagieren die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind mit der Topographie der japanischen Inseln, um diese speziellen Kármán-Wirbelwolken im Meer zu erzeugen? |  | Die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind spielen eine entscheidende Rolle bei der Entstehung der charakteristischen Kármán-Wirbelwolken über dem Japanischen Meer.⏎⏎Im Winter bringt der starke nordwestliche Monsun mit Geschwindigkeiten von 12-15 m/s kalte, trockene kontinentale Luft aus Richtung Sibirien. Diese Luftmasse führt zu markanten Temperaturunterschieden zwischen den kalten nördlichen Gebieten mit durchschnittlichen Lufttemperaturen von -20°C im Januar/Februar und den wärmeren südlichen Regionen mit etwa 5°C. Der Wintermonsun verstär… [truncated, 1686 chars total] | memorization | de | https://en.wikipedia.org/wiki/Sea_of_Japan |

### Raw — `exercise` = `memorization`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| Wie genau interagieren die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind mit der Topographie der japanischen Inseln, um diese speziellen Kármán-Wirbelwolken im Meer zu erzeugen? |  | Die Wärmegradienten zwischen dem kalten nordwestlichen Wintermonsun und dem warmen Sommerwind spielen eine entscheidende Rolle bei der Entstehung der charakteristischen Kármán-Wirbelwolken über dem Japanischen Meer.⏎⏎Im Winter bringt der starke nordwestliche Monsun mit Geschwindigkeiten von 12-15 m/s kalte, trockene kontinentale Luft aus Richtung Sibirien. Diese Luftmasse führt zu markanten Temperaturunterschieden zwischen den kalten nördlichen Gebieten mit durchschnittlichen Lufttemperaturen von -20°C im Januar/Februar und den wärmeren südlichen Regionen mit etwa 5°C. Der Wintermonsun verstär… [truncated, 1686 chars total] | memorization | de |

### Raw — `exercise` = `mcq`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| Based on the most recent scientific understanding of the evolutionary adaptation of viruses, what is considered the primary adaptive advantage of recombination in HIV-1?⏎⏎A. To generate genetic variation that can be selected for by the host's immune system.⏎B. To allow two different parental viruses to combine their genetic information.⏎C. To serve as a mechanism for repairing damage in the viral RNA genomes.⏎D. To overcome the chronic inflammation and reactive oxygen species produced by the host. | domain:medicine, exercise:fact recall, answer_shape:standard, exercise_pattern:other | C. To serve as a mechanism for repairing damage in the viral RNA genomes. | mcq | en |

### Raw — `exercise` = `constrained writing`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| I'm working on promotional copy for a new historical documentary about the aftermath of the Gulf War that focuses on what happened after the ceasefire agreement. We want to highlight how the peace conference authorized Iraqi helicopters for government transit but these were later used to suppress rebellions. The content should emphasize the uprising in Basra that spread to major Shia cities and the brutal response by Iraqi forces that resulted in 25,000 to 100,000 civilian casualties. We also need to mention how Kurdish rebels were abandoned by American support, forcing millions to flee to Tur… [truncated, 908 chars total] | length_constraints:number_sentences[15], oulipo:avoid_letter[t] | A fragile accord ends a major war. A peace conference allows Iraqi helicopters for official use. Such machines were soon used for crushing rebellion. A huge Shia uprising began in Basra. Iraqi forces responded by firing on civilians. A shocking number of people died; some say 25,000, some say 100,000. Kurdish rebels were also abandoned. American help for Kurds was gone. Millions of Kurds escaped, a huge forced exodus. Many fled over borders, seeking refuge in foreign lands. In oil-rich Kuwait, a new era began. A huge population divide showed a deep schism. A new leader, a brave Emir, came home… [truncated, 738 chars total] | constrained writing | en |

### Raw — `exercise` = `math exercise`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| A commercial bait company receives a shipment of 186 crates of fresh bait. After processing, they discover that 8 of the crates are spoiled and cannot be used. For distribution, the company decides to pack the remaining crates into smaller units, with each crate containing 90 pounds of bait. How many pounds of usable bait are packed in total? | result:16020 | **16,020 pounds**⏎⏎This is a multi-step word problem involving **subtraction and multiplication** - fundamental operations that model real-world inventory management scenarios.⏎⏎## Problem Structure⏎⏎The bait company starts with 186 crates, but discovers 8 are spoiled and unusable. After removing these unusable units, they pack the remainder into smaller units of 90 pounds each.⏎⏎## Solution Path⏎⏎**Step 1: Calculate usable crates**⏎- Total crates - Unusable crates = 186 - 8 = **178 usable crates**⏎⏎**Step 2: Calculate total bait**⏎- Usable crates × Pounds per crate = 178 × 90⏎⏎**Step 3: Perfo… [truncated, 988 chars total] | math exercise | en |

### Raw — `exercise` = `rag`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| How the pledge of loyalty requirements for New South Wales parliamentarians have evolved from allegiance to the monarchy to direct commitment to Australia and local population. | <source_1>⏎Constitution⏎⏎The form of the Government of New South Wales is prescribed in its Constitution, dating from 1856 and currently the Constitution Act 1902 (NSW). Since 1901 New South Wales has been a state of the Commonwealth of Australia, and the Australian Constitution regulates its relationship with the Commonwealth.⏎⏎In 2006, the Constitution Amendment Pledge of Loyalty Act 2006 No 6, was enacted to amend the NSW Constitution Act 1902 to require Members of the New South Wales Parliament and its Ministers to take a pledge of loyalty to Australia and to the people of New South Wales … [truncated, 13132 chars total] | In New South Wales, the requirement for parliamentarians to pledge loyalty to the monarch evolved into a direct commitment to the state and its people. The Constitution Amendment Pledge of Loyalty Act 2006 changed the state constitution to require members of the New South Wales Parliament and their ministers to take a pledge of loyalty to Australia and the people of New South Wales, instead of swearing allegiance to the monarch<ref name="source_1">In 2006, the Constitution Amendment Pledge of Loyalty Act 2006 No 6, was enacted to amend the NSW Constitution Act 1902 to require Members of the Ne… [truncated, 1278 chars total] | rag | en |

### Raw — `exercise` = `math mcq`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| In a large park, a group of friends is playing a tag game. At the beginning of the game, 7 players are "it". The total number of players in the group is 50. If the game is played for a total of 51 rounds, what is the average number of players who are *not* "it" per round, rounded to the nearest whole number?⏎⏎A. 0⏎B. 1⏎C. 2⏎D. 43 | result:1 | B. 1 | math mcq | en |

### Raw — `exercise` = `creative writing`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| I'm hoping for some poetry from 1984 that deals with the idea of managing pain over time versus getting quick relief - kind of like how some medications work gradually throughout the day while others are for emergency situations. I'd love to find poems set in hospitals or medical facilities that capture that tension between steady, long-term treatment and those moments when regular medicine just isn't enough. Something that explores how people navigate the day-to-day reality of chronic pain versus those breakthrough moments that require immediate intervention. | form:Poetry, spatial:place, temporal:publication_date[1984] | Morning Rounds, 1984⏎⏎The hospital swallows the city in fog.⏎Refrigerators hum like ventilators.⏎Oxygen lines snake the corridors,⏎white arteries feeding the body politic.⏎⏎I take the day in capsules:⏎sustained-release in the dim ward light,⏎a slow-acting dawn to keep the alarms down.⏎The nurse says, Take one, then wait.⏎I watch the hallway’s linoleum tide⏎lap against the IV poles,⏎count the seconds between drips⏎like rosary beads of mercy.⏎⏎By noon the ceiling tiles are halos.⏎I learn the map of pain the way a commuter learns⏎the quickest way around the bridge work:⏎left shoulder, a knot at t… [truncated, 2861 chars total] | creative writing | en |

### Raw — `exercise` = `editing`

One raw row with this value (same storage as above), shown as a table:

| query | constraints | synthetic_answer | exercise | language |
|---|---|---|---|---|
| Hey, so I've got this text about Thomas K. Kuhn and his book "The Structure of Scientific Revolutions" but I think there might be some errors from when I scanned it or something. Could you clean up any mistakes that look wrong and maybe... um, put it in a better format? Like maybe break it down into bullet points or something? The original was kinda dense and hard to follow. Here's what I have:⏎⏎The Structure of Scientific Revolutions is a book about the history of science by the philosopher Thomas K. Kuhn. Its publication was a landmark event in the history, philosophy, and sociology of scien… [truncated, 2299 chars total] | quality:corruption, structural:reformatting | Here is a cleaned-up and reformatted version of the text about Thomas Kuhn's book, *The Structure of Scientific Revolutions*.⏎⏎*** | editing | en |

### Transformed — condition=`synth,cot`

One row of `data_clustered/SYNTH/synth_001.parquet` (parquet table, columns: `instruction`, `condition`, `response`), shown as a table:

| instruction | condition | response |
|---|---|---|
| hey um can someone tell me why the decagon has like so many different ways to look the same but not completely the same?? like i see there's dih and z and stuff but how does that work with the big one that looks normal?? | synth,cot | You're right to notice that the decagon has multiple ways to look similar but not identical! This happens because of its rich symmetry structure.⏎⏎The regular decagon has what's called Dih10 symmetry with order 20, meaning there are 20 different ways you can rotate or reflect it and still have it look the same. But within this, there are several smaller symmetry groups that work differently.⏎⏎The decagon contains 8 distinct symmetry types that can be seen in various ways on the shape. These symmetries can be broken down into different categories based on whether they pass through vertices (d f… [truncated, 1518 chars total] |

### Transformed — condition=`synth,direct`

One row of `data_clustered/SYNTH/synth_001.parquet` (parquet table, columns: `instruction`, `condition`, `response`), shown as a table:

| instruction | condition | response |
|---|---|---|
| Based on the most recent scientific understanding of the evolutionary adaptation of viruses, what is considered the primary adaptive advantage of recombination in HIV-1?⏎⏎A. To generate genetic variation that can be selected for by the host's immune system.⏎B. To allow two different parental viruses to combine their genetic information.⏎C. To serve as a mechanism for repairing damage in the viral RNA genomes.⏎D. To overcome the chronic inflammation and reactive oxygen species produced by the host. | synth,direct | C. To serve as a mechanism for repairing damage in the viral RNA genomes. |

---

↑ [index](README.md) · ← [flan](flan.md) · [next → dmmath](dmmath.md)
