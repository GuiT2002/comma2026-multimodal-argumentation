# Multimodal argumentation experiments

Run argument formalization experiments with DeepSeek, OpenAI, and/or Claude, validate the generated arguments, and print evaluation statistics.

| Script | Purpose |
| --- | --- |
| `main.py` | Generate arguments for every case and input configuration. |
| `validation.py` | Check the argumentation scheme locally and compare conclusions using DeepSeek. |
| `statistics.py` | Summarize the saved binary evaluations in the terminal. |

## Classes and input configurations

| File identifier | Scenario class |
| --- | --- |
| `class1` | Geographic |
| `class2` | Temporal |
| `class3` | Geographic + Temporal |

Every case is evaluated under four input configurations, in this order:

| Result column | Experiment identifier | Input configuration |
| --- | --- | --- |
| `col1` | `without_enthymeme` | Image + Context |
| `col2` | `without_context` | Enthymeme + Image |
| `col3` | `without_image` | Enthymeme + Context |
| `col4` | `all_inputs` | Enthymeme + Image + Context |

The configuration without an image uses a dedicated prompt without image references. The other three use the multimodal prompt. Argumentation scheme examples come from `all_examples.txt`.

## Installation

Requires Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Dependencies are the `openai` and `anthropic` SDKs. The OpenAI SDK also accesses DeepSeek. Statistics and validation's `--dry-run` need only the Python standard library.

## Data layout

```text
repository/
├── main.py
├── validation.py
├── statistics.py
├── requirements.txt
├── all_examples.txt
├── tests/
│   ├── class1.tsv
│   ├── class2.tsv
│   └── class3.tsv
├── images/
│   ├── class1/test1.png, test2.png, ...
│   ├── class2/test1.png, test2.png, ...
│   └── class3/test1.png, test2.png, ...
├── expected_outputs/
│   ├── class1.txt
│   ├── class2.txt
│   └── class3.txt
├── results/
│   └── <model-directory>/class1.txt, class2.txt, class3.txt
└── validation/
    └── <model-directory>_class<N>_col<N>.csv
```

`main.py` requires exactly three `.tsv` files directly in `tests/`. Each UTF-8 file has no header and contains two tab-separated fields per record:

```text
enthymeme<TAB>context
```

Use a real tab in place of `<TAB>`. To omit a field, keep the separator and leave that field empty. Standard CSV quoting is supported with a tab delimiter: double-quoted fields may span lines or contain tabs, and embedded double quotes are doubled. The reader parses complete records before replacing tabs and line breaks inside fields with spaces. A case is one record, which may occupy several physical lines in the source TSV.

`images/class1/test1.png` corresponds to the first record of `tests/class1.tsv`; numbering restarts for each class. Each class must contain at least one case and have an image for every case. All datasets and image paths are checked before API calls. Case counts come from the input files rather than fixed per-class totals.

For validation, each `expected_outputs/class<N>.txt` must contain one nonempty argument per line, starting with `[`, in the same case order as the results. The number of expected arguments determines the required number of result rows for that class.

Default paths are relative to the scripts' directory. Explicit directory options are resolved from the current working directory when they are relative paths.

## Running the experiments

Select providers with command-line API keys. These are the model identifiers configured in `main.py` and their output directories:

| Option | Model identifier | Result directory |
| --- | --- | --- |
| `--deepseek-api-key` | `deepseek-flash` | `results/deepseek-flash/` |
| `--openai-api-key` | `gpt-6.1-sol` | `results/gpt-6.1/` |
| `--claude-api-key` | `claude-sonnet-5-5` | `results/claude-sonnet/` |

Run one provider or combine the options:

```bash
python3 main.py --deepseek-api-key "YOUR_DEEPSEEK_API_KEY"
python3 main.py --openai-api-key "YOUR_OPENAI_API_KEY"
python3 main.py --claude-api-key "YOUR_CLAUDE_API_KEY"
```

```bash
python3 main.py \
  --deepseek-api-key "YOUR_DEEPSEEK_API_KEY" \
  --openai-api-key "YOUR_OPENAI_API_KEY" \
  --claude-api-key "YOUR_CLAUDE_API_KEY"
```

At least one key is required. `main.py` does not select providers from environment variables or load `.env`. Each case generates four API calls per selected model, covering all three classes automatically.

### Generated results

After processing a class, `main.py` writes a UTF-8, tab-delimited `class<N>.txt` in each selected model's directory, replacing that model/class's existing file. Each case occupies one row with eight fields:

| Fields | Contents |
| --- | --- |
| 1–4 | `Enthymeme`, `Context`, `Image`, `Expected Output` |
| 5–8 | The four experiment responses in the order shown above. |

The metadata preserves the original case. `Image` contains a relative path such as `images/class1/test1.png`; `Expected Output` is left blank. Validation reads its reference answers from `expected_outputs/`, not this blank metadata field.

Response headers use `<model-identifier>__<experiment-identifier>`, such as `claude-sonnet-5-5__without_enthymeme`. Response cells contain only the text extracted from the model's `<final_output>` block, without its enclosing tags or reasoning outside the block. Tabs and line breaks inside every field are replaced with spaces, so result rows have no multiline fields or CSV quoting.

Missing or empty final-output blocks, empty API responses, and truncated responses stop execution. Claude's output limit is configured by `CLAUDE_MAX_TOKENS` in `main.py` (currently 8192). Experiment generation has no automatic resume: rerunning regenerates the selected models' results.

## Validating the arguments

`validation.py` discovers the non-hidden model directories present in `results/` and requires `class1.txt`, `class2.txt`, and `class3.txt` in each. It reads tab-separated TXT results; CSV copies in `results/` are not used. Before API calls, it checks headers, experiment order, row counts, image/case alignment, and any saved evaluations.

Each expected argument is paired with the corresponding model response separately for all four experiment columns. The two judgments are independent:

- **Scheme correctness:** the professor's predicate-presence criterion returns `yes` when the normalized output contains all three predicates `position_to_know`, `asserts`, and `contain`. This checks their presence, not the full logical validity of the argument.
- **Semantic correspondence:** `deepseek-v4-pro` compares only the conclusions. The prompt accepts synonymous predicates and different levels of detail when the core claim is preserved, for example `cold_front_will_pass_through_philadelphia_tonight` and `cold_front(philadelphia)`. Explicit contradictions in location, time, event, or polarity still receive `no`.

The semantic call enables thinking with `extra_body={"thinking": {"type": "enabled"}}`. Its current `reasoning_effort` is `"low"`, configured in `semantic_correct`. Only `message.content` is accepted as the final judgment; `reasoning_content` is never used. The response must be exactly lowercase `yes` or `no` after trimming surrounding whitespace, with a completed response. Invalid answers are retried up to three evaluation attempts; persistent failures stop the run while preserving completed rows.

Check inputs and saved evaluations without API calls or writes:

```bash
python3 validation.py --dry-run
```

Run validation with an explicit key:

```bash
python3 validation.py --deepseek-api-key "YOUR_DEEPSEEK_API_KEY"
```

Alternatively, set `DEEPSEEK_API_KEY` in the process environment:

```bash
export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
python3 validation.py
```

The command-line key takes precedence. None of the scripts automatically loads `.env`.

### Validation files and resuming

Validation creates one CSV per model, class, and experiment, for example `validation/claude-sonnet_class1_col1.csv`. Each file contains four columns:

```csv
expected_output,model_output,scheme_correct,semantic_match
```

Rows preserve case order and both argument texts. The last two fields contain only `yes` or `no`. Standard CSV quoting preserves commas within argument representations. Each successfully evaluated case is saved immediately; rerun the same command to resume from the saved prefix. Completed files are skipped. Saved rows must match the current arguments and local scheme judgments; mismatches require a new output directory.

Changing the semantic prompt, judge model, or reasoning settings does not invalidate previously saved judgments automatically. Use a fresh directory to re-evaluate with the new configuration:

```bash
python3 validation.py --output-dir validation_rechecked --dry-run
python3 validation.py --output-dir validation_rechecked
python3 statistics.py --validation-dir validation_rechecked
```

The validation command above uses `DEEPSEEK_API_KEY`. Custom input directories are also supported:

```bash
python3 validation.py \
  --results-dir results \
  --expected-dir expected_outputs \
  --output-dir validation_rechecked
```

Removing a model directory from `results/` excludes it from future validation runs but does not remove its existing validation CSVs.

## Evaluation statistics

```bash
python3 statistics.py
python3 statistics.py --validation-dir validation_rechecked
```

The script reads every `.csv` file directly in the chosen directory. It validates filenames, headers, and binary judgments, then prints:

- Overall scheme correctness, semantic correspondence, and simultaneous correctness.
- The four joint outcomes: `yes/yes`, `yes/no`, `no/yes`, and `no/no`.
- Results by model, by scenario class, and by input configuration.
- Results by model and class, by model and configuration, and by class and configuration.

Each rate is the number of successful evaluations divided by all evaluations in that group (micro-average), displayed as `correct/total (percentage)`. One evaluation is one case for one model and one input configuration. Incomplete runs are summarized using only saved rows; totals are not extrapolated. Old model CSVs remain included until removed from the selected validation directory.

## Command help

```bash
python3 main.py --help
python3 validation.py --help
python3 statistics.py --help
```
