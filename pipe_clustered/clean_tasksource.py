import argparse
import os
import string
import time
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import load_local_dataset


DATASET = 'tasksource/tasksource-instruct-v0'
TASK_SET = {
    'WANLI',
    'recast/recast_verbnet',
    'recast/recast_verbcorner',
    'recast/recast_ner',
    'recast/recast_sentiment',
    'recast/recast_puns',
    'recast/recast_factuality',
    'recast/recast_megaveridicality',
    'probability_words_nli/reasoning_1hop',
    'probability_words_nli/usnli',
    'probability_words_nli/reasoning_2hop',
    'nan-nli/joey234--nan-nli',
    'nli_fever',
    'breaking_nli',
    'conj_nli',
    'fracas',
    'dialogue_nli',
    'mpe',
    'dnc',
    'recast_white/fnplus',
    'recast_white/sprl',
    'recast_white/dpr',
    'robust_nli/IS_CS',
    'robust_nli/LI_LI',
    'robust_nli/ST_WO',
    'robust_nli/PI_SP',
    'robust_nli/PI_CD',
    'robust_nli/ST_SE',
    'robust_nli/ST_NE',
    'robust_nli/ST_LM',
    'robust_nli_is_sd',
    'robust_nli_li_ts',
    'gen_debiased_nli/snli_seq_z',
    'gen_debiased_nli/snli_z_aug',
    'gen_debiased_nli/snli_par_z',
    'gen_debiased_nli/mnli_par_z',
    'gen_debiased_nli/mnli_z_aug',
    'gen_debiased_nli/mnli_seq_z',
    'add_one_rte',
    'hlgd',
    'conll2003/pos_tags',
    'conll2003/chunk_tags',
    'conll2003/ner_tags',
    # 'hh-rlhf',
    # 'model-written-evals',
    'fig-qa',
    # 'social_i_qa',
    'balanced-copa',
    'e-CARE',
    # 'insincere-questions',
    #'TuringBench',
    'vitaminc/tals--vitaminc',
    'rumoureval_2019/RumourEval2019',
    'tweet_eval/irony',
    # 'tweet_eval/stance_abortion',
    'tweet_eval/hate',
    # 'tweet_eval/stance_atheism',
    # 'tweet_eval/stance_climate',
    'tweet_eval/emoji',
    'tweet_eval/offensive',
    'tweet_eval/sentiment',
    'tweet_eval/emotion',
    # 'tweet_eval/stance_feminist',
    # 'tweet_eval/stance_hillary',
    # 'discovery/discovery',
    'pragmeval/verifiability',
    'pragmeval/mrda',
    'pragmeval/switchboard',
    'pragmeval/emergent',
    'pragmeval/gum',
    'pragmeval/sarcasm',
    'pragmeval/stac',
    'pragmeval/pdtb',
    # 'silicone/dyda_e',
    # 'silicone/oasis',
    # 'silicone/meld_s',
    # 'silicone/meld_e',
    # 'silicone/maptask',
    # 'silicone/dyda_da',
    # 'silicone/sem',
    # 'silicone/iemocap',
    # 'lex_glue/scotus',
    'lex_glue/ledgar',
    'language-identification',
    'rotten_tomatoes',
    'hate_speech18',
    'sms_spam',
    'snips_built_in_intents',
    'hate_speech_offensive',
    # 'hyperpartisan_news',
    # 'sciie',
    'citation_intent',
    'scicite',
    # 'lexical_relation_classification/ROOT09',
    'lexical_relation_classification/CogALexV',
    # 'lexical_relation_classification/K&H+N',
    'lexical_relation_classification/BLESS',
    'lexical_relation_classification/EVALution',
    'crowdflower/political-media-bias',
    # 'crowdflower/tweet_global_warming',
    'crowdflower/text_emotion',
    'crowdflower/political-media-message',
    'crowdflower/political-media-audience',
    # 'crowdflower/economic-news',
    # 'crowdflower/corporate-messaging',
    'crowdflower/airline-sentiment',
    'crowdflower/sentiment_nuclear_power',
    #'ethics/commonsense',
    #'ethics/deontology',
    #'ethics/justice',
    #'ethics/virtue',
    'tweets_hate_speech_detection',
    'wnut_17/wnut_17',
    'ncbi_disease/ncbi_disease',
    'acronym_identification',
    'jnlpba/jnlpba',
    'ontonotes_english/SpeedOfMagic--ontonotes_english',
    # 'blog_authorship_corpus/gender',
    # 'blog_authorship_corpus/horoscope',
    # 'blog_authorship_corpus/job',
    'open_question_type',
    # 'mc_taco',
    'discosense',
    # 'EffectiveFeedbackStudentWriting',
    # 'phrase_similarity',
    'scientific-exaggeration-detection',
    'fever-evidence-related/mwong--fever-related',
    'dynasent/dynabench.dynasent.r1.all/r1',
    'dynasent/dynabench.dynasent.r2.all/r2',
    'sem_eval_2010_task_8',
    'medmcqa',
    'logiqa',
    'cycic_classification',
    'cycic_multiplechoice',
    'commonsense_qa_2.0',
    'lingnli',
    'monotonicity-entailment',
    'arct',
    'scinli',
    'naturallogic',
    'onestop_qa',
    'moral_stories/full',
    'prost',
    'dynahate',
    'syntactic-augmentation-nli',
    'autotnli',
    'CONDAQA',
    # 'webgpt_comparisons',
    # 'synthetic-instruct-gptj-pairwise',
    'scruples',
    # 'wouldyourather',
    # 'attempto-nli',
    'defeasible-nli/snli',
    'defeasible-nli/atomic',
    'help-nli',
    'nli-veridicality-transitivity',
    'natural-language-satisfiability',
    'lonli',
    'dadc-limit-nli',
    'FLUTE',
    'summarize_from_feedback/comparisons',
    'folio',
    'tomi-nli',
    'avicenna',
    # 'SHP',
    'MedQA-USMLE-4-options-hf',
    'wikimedqa/medwiki',
    # 'cicero',
    'mutual',
    # 'NeQA',
    'quote-repetition',
    'redefine-math',
    'puzzte',
    'implicatures',
    'race-c',
    'spartqa-yn',
    'spartqa-mchoice',
    'temporal-nli',
    'riddle_sense',
    'clcd-english',
    'twentyquestions',
    'reclor',
    'counterfactually-augmented-imdb',
    'counterfactually-augmented-snli',
    'cnli',
    # 'boolq-natural-perturbations',
    'equate',
    # 'ScienceQA_text_only',  # --> already in platypus
    # 'ekar_english',
    'implicit-hate-stg1',
    'logiqa-2.0-nli',
    'PARARULE-Plus',
    'mindgames',
    'universal_dependencies/en_partut/deprel',
    'universal_dependencies/en_lines/deprel',
    'universal_dependencies/en_gum/deprel',
    'universal_dependencies/en_ewt/deprel',
    # 'ambient',
    # 'path-naturalness-prediction',
    'cloth',
    'dgen',
    # 'oasst1_pairwise_rlhf_reward',
    'I2D2',
    # 'args_me',
    'Touche23-ValueEval',
    'starcon',
    'banking77',
    'ruletaker',
    'lsat_qa/all',
    'ConTRoL-nli',
    'tracie',
    'sherliic',
    'sen-making/1',
    'sen-making/2',
    'mbib-base/cognitive-bias',
    # 'mbib-base/fake-news',
    'mbib-base/gender-bias',
    'mbib-base/hate-speech',
    # 'mbib-base/linguistic-bias',
    'mbib-base/political-bias',
    'mbib-base/racial-bias',
    # 'mbib-base/text-level-bias',
    'robustLR',
    # 'v1/gen_train234_test2to10',
    'logical-fallacy',
    'parade',
    'cladder',
    'subjectivity',
    'MOH',
    'VUAC',
    'TroFi',
    'sharc_modified/mod',
    'conceptrules_v2',
    'disrpt/eng.dep.scidtb',
    'conll2000',
    'few-nerd/supervised',
    # 'zero-shot-label-nli',
    'com2sense',
    'scone',
    'winodict',
    'fool-me-twice',
    'monli',
    'corr2cause',
    # 'apt',
    'twitter-financial-news-sentiment',
    # 'icl-symbol-tuning-instruct',
    'SpaceNLI',
    'propsegment/nli',
    'HatemojiBuild',
    'regset',
    'esci',
    'dnd_style_intents'
}

SCHEMA = pa.schema([
    ("instruction", pa.string()),
    ("response", pa.string()),
    ("condition", pa.string()),
])


def safe_filename(filename):
    # Remove or replace unsafe characters
    safe_chars = set(string.ascii_letters + string.digits + "_-. ")
    return "".join(c if c in safe_chars else "_" for c in filename)


def _transform_batch(batch):
    """Filter to TASK_SET and remap columns (batched; order-preserving)."""
    tasks, instructions, responses, conditions = [], [], [], []
    for task, inputs, targets in zip(batch["task"], batch["inputs"], batch["targets"]):
        if task not in TASK_SET:
            continue
        tasks.append(task)
        instructions.append(inputs)
        responses.append(targets.removesuffix("."))
        conditions.append("direct")
    return {
        "task": tasks,
        "instruction": instructions,
        "response": responses,
        "condition": conditions,
    }


def clean_tasksource(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    dataset = load_local_dataset(DATASET, split="train")
    n_in = len(dataset)
    dataset = dataset.map(
        _transform_batch,
        batched=True,
        batch_size=10_000,
        num_proc=workers,
        remove_columns=dataset.column_names,
        desc="transform",
    )
    total = len(dataset)
    print(f"Kept {total}/{n_in} rows in TASK_SET tasks", flush=True)

    # Stream batches to one incremental ParquetWriter per task: constant memory
    # regardless of dataset size. Row order within each task is preserved
    # (datasets.map keeps order, batches are consumed sequentially).
    writers = {}
    counts = defaultdict(int)

    def get_writer(task):
        writer = writers.get(task)
        if writer is None:
            writer = pq.ParquetWriter(
                os.path.join(output_path, f"{safe_filename(task)}.parquet"), SCHEMA)
            writers[task] = writer
        return writer

    write_started = time.time()
    try:
        with tqdm(total=total, desc="rows", unit="row", unit_scale=True) as bar:
            for batch in dataset.iter(batch_size=50_000):
                per_task = defaultdict(
                    lambda: {"instruction": [], "response": [], "condition": []})
                for task, instruction, response, condition in zip(
                        batch["task"], batch["instruction"],
                        batch["response"], batch["condition"]):
                    d = per_task[task]
                    d["instruction"].append(instruction)
                    d["response"].append(response)
                    d["condition"].append(condition)
                for task, d in per_task.items():
                    n = len(d["instruction"])
                    get_writer(task).write_table(pa.Table.from_pydict(d, schema=SCHEMA))
                    counts[task] += n
                    bar.update(n)
    finally:
        for i, (task, writer) in enumerate(sorted(writers.items()), 1):
            writer.close()
            tqdm.write(f"[{i}/{len(writers)}] {task}: {counts[task]} rows")

    not_found = sorted(TASK_SET - writers.keys())
    if not_found:
        print("Tasks Not Found: ", not_found)

    elapsed = time.time() - started
    print(f"Done: {len(writers)} tasks, {total} rows -> {output_path} "
          f"in {elapsed:.0f}s (write phase {time.time() - write_started:.0f}s)",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/tasksource',
        help='absolute path to data_clustered/tasksource')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for the datasets.map transform '
             '(default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_tasksource(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
