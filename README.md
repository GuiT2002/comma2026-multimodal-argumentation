# Multimodal argumentation experiments

Run argument formalization experiments with DeepSeek, OpenAI, and/or Claude. Each run processes all three test classes and sends three input combinations to the selected models for each case:

| Order | Experiment | Inputs sent |
| --- | --- | --- |
| 1 | `without_enthymeme` | Context and image |
| 2 | `without_context` | Enthymeme and image |
| 3 | `without_image` | Enthymeme and context |

The experiment without an image uses the dedicated prompt without image references. The other two use the original multimodal prompt. Argumentation scheme examples are loaded from `all_examples.txt`.

## Installation

Requires Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The dependencies are the `openai` and `anthropic` SDKs. The OpenAI SDK is also used to access the DeepSeek API.

## Input data

Place exactly three `.txt` files in the `tests/` directory. You can choose their names; each filename without its extension identifies the class and its image directory:

```text
repository/
├── main.py
├── requirements.txt
├── all_examples.txt
├── tests/
│   ├── class1.txt
│   ├── class2.txt
│   └── class3.txt
└── images/
    ├── class1/
    │   ├── test1.png
    │   └── test2.png
    ├── class2/
    │   ├── test1.png
    │   └── test2.png
    └── class3/
        ├── test1.png
        └── test2.png
```

Each TXT file must use UTF-8, with no header, one case per line, and exactly two columns separated by a real tab:

```text
enthymeme<TAB>context
```

Replace `<TAB>` with the tab character; do not write the literal characters `\t`. To leave a field empty, keep the tab. Do not include blank lines or tabs or line breaks within a field. The code keeps enthymemes and contexts in separate lists.

`test1.png` corresponds to the first line of that class's TXT file, `test2.png` to the second line, and so on. Numbering restarts for each class. Classes may contain different numbers of cases, but each class must have at least one case and every case must have an image, since all three experiments are always run.

All data and output paths are relative to the directory containing `main.py`, regardless of the directory from which the command is run. All three input files and their images are checked before any API calls. You must provide the data for all three classes; the program does not create or split the dataset automatically.

## Running the experiments

Select providers by passing their API keys on the command line. At least one provider is required:

| Option | Model |
| --- | --- |
| `--deepseek-api-key` | `deepseek-flash` |
| `--openai-api-key` | `gpt-6.1-sol` |
| `--claude-api-key` | `claude-sonnet-5-5` |

DeepSeek only:

```bash
python3 main.py --deepseek-api-key "YOUR_DEEPSEEK_API_KEY"
```

OpenAI only or Claude only:

```bash
python3 main.py --openai-api-key "YOUR_OPENAI_API_KEY"
python3 main.py --claude-api-key "YOUR_CLAUDE_API_KEY"
```

All three models in the same run:

```bash
python3 main.py \
  --deepseek-api-key "YOUR_DEEPSEEK_API_KEY" \
  --openai-api-key "YOUR_OPENAI_API_KEY" \
  --claude-api-key "YOUR_CLAUDE_API_KEY"
```

You can also combine any two providers. Only providers whose keys are supplied on the command line are used; environment variables or a `.env` file alone do not enable them. API keys are not included in the output files.

All three classes are discovered automatically in `tests/`; do not pass filenames as arguments. Each case generates three API calls per selected model. To display the available options:

```bash
python3 main.py --help
```

## Results

After completing each class, the program writes `results/<class>.txt`, overwriting any previous output for that class. A complete run produces three UTF-8 files with tab-separated columns, ready to copy and paste into a spreadsheet.

The header starts with `Enthymeme`, `Context`, `Image`, and `Expected Output`, followed by one column per selected model. Model columns always follow the order DeepSeek, OpenAI, and Claude, with model identifiers as column names. With all three models, the structure is:

```text
Enthymeme<TAB>Context<TAB>Image<TAB>Expected Output<TAB>deepseek-flash<TAB>gpt-6.1-sol<TAB>claude-sonnet-5-5
```

The generated files contain real tab characters as separators. `Expected Output` is always left blank for manual entry; no expected-answer file is required.

Each case occupies three consecutive rows in this order: `without_enthymeme`, `without_context`, and `without_image`. The omitted input field is left empty. `Image` contains the image path when an image is sent. There is no additional column for the experiment name.

Each response cell contains the text returned by the model, including its tags, without extracting only `<final_output>`. Tabs and line breaks within cells are replaced with spaces to keep each experiment on a single spreadsheet row.

## Common errors

- **No provider selected:** supply at least one API key option.
- **Incorrect number of classes:** keep exactly three `.txt` files directly in `tests/`.
- **Invalid input:** check for real tabs, exactly two columns per line, and no blank lines.
- **Image not found:** check that the TXT filename matches the directory in `images/` and that images follow the `test1.png`, `test2.png`, etc. naming convention.
- **Missing dependency:** activate the virtual environment and install `requirements.txt`.
- **API error:** check the key, model access, and the provider's error message. Execution stops; results for completed classes remain saved. The interrupted class is not written, and any previous output file for that class remains unchanged.

Empty or truncated responses also stop execution. Claude's output limit is 8192 tokens, configured through `CLAUDE_MAX_TOKENS` in `main.py`.

Provider documentation: [DeepSeek](https://api-docs.deepseek.com/), [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), and [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
