import argparse
import base64
import csv
import re
from contextlib import ExitStack
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "deepseek": "deepseek-flash",
    "openai": "gpt-6.1-sol",
    "claude": "claude-sonnet-5-5",
}
CLAUDE_MAX_TOKENS = 8192
EXPERIMENTS = (
    "without_enthymeme",
    "without_context",
    "without_image",
    "all_inputs",
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_tests(tests_path):
    enthymemes_list = []
    contexts_list = []

    with open(tests_path, "r", encoding="utf-8-sig", newline="") as tests_file:
        reader = csv.reader(tests_file, delimiter="\t", strict=True)
        try:
            for record_number, columns in enumerate(reader, start=1):
                if len(columns) != 2:
                    raise ValueError(
                        f"{tests_path}, record {record_number}: expected two columns "
                        "separated by a tab (enthymeme and context). "
                        "To omit a field, keep the tab and leave the field empty."
                    )
                # Parse complete records before normalizing whitespace within cells.
                enthymemes_list.append(spreadsheet_cell(columns[0]).strip())
                contexts_list.append(spreadsheet_cell(columns[1]).strip())
        except csv.Error as exc:
            raise ValueError(
                f"{tests_path}, line {reader.line_num}: invalid TSV: {exc}"
            ) from exc

    return enthymemes_list, contexts_list


def build_prompt(examples_list, enthymeme, context):
    case_parts = []
    if enthymeme:
        case_parts.append(f"Enthymeme: {enthymeme}")
    if context:
        case_parts.append(f"Context: {context}")
    case_inputs = "\n        ".join(case_parts)

    return f"""

        You are an AI assistant responsible for formalizing argumentative structures in the context provided by the user.
        You will receive three inputs: an enthymeme (an argument with missing premises or a missing conclusion), an image, and a brief description of the context in which the situation takes place.

        - The enthymeme will be an argumentative sentence that belongs to an argumentation scheme. You will receive only part of the argumentation scheme. The rest will be intentionally omitted.
        - The image depicts something the user observed but described using only a partial argumentative structure: the enthymeme mentioned earlier. Your task regarding the image will be to identify the elements necessary to complete the structure of the argumentation scheme based on the enthymeme provided by the user.
        - The context will be a brief description of how certain elements in the image might relate to the enthymeme written by the user.

        Your task is to use the enthymeme the user wrote, the image, and the provided context to instantiate the variables of the argumentation scheme to which that argumentative sentence belongs.
        The enthymeme is an argumentative sentence belonging to ONLY ONE of the argumentation schemes described below.
        The following list presents every argumentation scheme to be considered, along with a single example for each scheme showing how to translate an argument from natural language into a formal representation:

        {examples_list}


        You must think step by step to correctly instantiate the full argumentation scheme. Be specific when instantiating the variables using the content of the image.
        Your response should follow this exact structure:

        <reasoning>
        // Write your step-by-step reasoning process here.
        <reasoning_f>

        <final_output>
        // Write ONLY the final formal representation of the argument here, as specified earlier. Do not add anything other than the fully instantiated argumentation scheme.
        <final_output_f>

        Follow all these instructions and consider the following case:

        {case_inputs}
        """


def build_prompt_without_image(examples_list, enthymeme, context):
    case_parts = []
    if enthymeme:
        case_parts.append(f"Enthymeme: {enthymeme}")
    if context:
        case_parts.append(f"Context: {context}")
    case_inputs = "\n        ".join(case_parts)

    return f"""

        You are an AI assistant responsible for formalizing argumentative structures in the context provided by the user.
        You will receive two inputs: an enthymeme (an argument with missing premises or a missing conclusion) and a brief description of the context in which the situation takes place.

        - The enthymeme will be an argumentative sentence that belongs to an argumentation scheme. You will receive only part of the argumentation scheme. The rest will be intentionally omitted.
        - The context will be a brief description of how certain elements of the situation might relate to the enthymeme written by the user.

        Your task is to use the enthymeme the user wrote and the provided context to instantiate the variables of the argumentation scheme to which that argumentative sentence belongs.
        The enthymeme is an argumentative sentence belonging to ONLY ONE of the argumentation schemes described below.
        The following list presents every argumentation scheme to be considered, along with a single example for each scheme showing how to translate an argument from natural language into a formal representation:

        {examples_list}


        You must think step by step to correctly instantiate the full argumentation scheme. Be specific when instantiating the variables using the enthymeme and the provided context.
        Your response should follow this exact structure:

        <reasoning>
        // Write your step-by-step reasoning process here.
        <reasoning_f>

        <final_output>
        // Write ONLY the final formal representation of the argument here, as specified earlier. Do not add anything other than the fully instantiated argumentation scheme.
        <final_output_f>

        Follow all these instructions and consider the following case:

        {case_inputs}
        """


def load_datasets(base_dir):
    tests_dir = base_dir / "tests"
    test_paths = sorted(path for path in tests_dir.glob("*.tsv") if path.is_file())
    if len(test_paths) != 3:
        raise ValueError(
            f"{tests_dir} must contain exactly three .tsv files, one per class "
            f"(found: {len(test_paths)})."
        )

    datasets = []
    for test_path in test_paths:
        class_name = test_path.stem
        enthymemes_list, contexts_list = load_tests(test_path)
        if not enthymemes_list:
            raise ValueError(f"{test_path}: the class file contains no test cases.")
        image_paths = [
            base_dir / "images" / class_name / f"test{test_n}.png"
            for test_n in range(1, len(enthymemes_list) + 1)
        ]
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
        datasets.append((class_name, enthymemes_list, contexts_list, image_paths))
    return datasets


def create_model_clients(args, stack):
    model_clients = []
    for provider, model in MODELS.items():
        api_key = getattr(args, f"{provider}_api_key")
        if api_key is None:
            continue
        if provider == "claude":
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, base_url="https://api.anthropic.com")
        else:
            from openai import OpenAI

            base_url = (
                "https://api.deepseek.com" if provider == "deepseek"
                else "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
        model_clients.append((provider, model, stack.enter_context(client)))
    return model_clients


def request_model(provider, model, client, prompt, image_base64):
    content = []
    if provider == "claude":
        if image_base64 is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64,
                },
            })
        content.append({"type": "text", "text": prompt})
        response = client.messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{model}: response truncated by the token limit.")
        text = "\n".join(block.text for block in response.content if block.type == "text")
    else:
        content.append({"type": "text", "text": prompt})
        if image_base64 is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        if response.choices[0].finish_reason == "length":
            raise RuntimeError(f"{model}: response truncated by the token limit.")
        text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError(f"{model}: the API did not return a text response.")
    return text


def run_experiment(model_clients, examples_list, enthymeme, context, image_path, experiment):
    supplied_enthymeme = enthymeme if experiment != "without_enthymeme" else ""
    supplied_context = context if experiment != "without_context" else ""
    image_base64 = None
    if experiment == "without_image":
        prompt = build_prompt_without_image(examples_list, supplied_enthymeme, supplied_context)
    else:
        prompt = build_prompt(examples_list, supplied_enthymeme, supplied_context)
        image_base64 = encode_image(image_path)

    return [
        request_model(provider, model, client, prompt, image_base64)
        for provider, model, client in model_clients
    ]


def spreadsheet_cell(value):
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")


def extract_final_output(response):
    # Accept the closing-tag variants already present in model responses.
    match = re.search(
        r"<final_output>(.*?)<(?:/?final_output_f|/final_output)>",
        response,
        flags=re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError("Model response must contain a nonempty final_output block.")
    return match.group(1).strip()


def write_results(output_path, model_clients, rows):
    header = ["Enthymeme", "Context", "Image", "Expected Output"]
    header.extend(
        f"{model}__{experiment}"
        for _, model, _ in model_clients
        for experiment in EXPERIMENTS
    )
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Each result row must contain one response per model and experiment.")
    # Validate all responses before opening the file to preserve existing results on error.
    output_rows = [
        list(row[:4]) + [extract_final_output(response) for response in row[4:]]
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        output_file.write("\t".join(header) + "\n")
        for row in output_rows:
            output_file.write("\t".join(spreadsheet_cell(value) for value in row) + "\n")


def nonempty_api_key(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("The API key must not be empty.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run all three classes in tests/ with the models selected by their API keys."
        ),
        allow_abbrev=False,
    )
    for provider, model in MODELS.items():
        parser.add_argument(
            f"--{provider}-api-key", type=nonempty_api_key, metavar="KEY",
            help=f"Enable {model} using the supplied API key.",
        )
    args = parser.parse_args(argv)
    if not any(getattr(args, f"{provider}_api_key") is not None for provider in MODELS):
        parser.error(
            "Provide at least one API key: --deepseek-api-key, --openai-api-key "
            "or --claude-api-key."
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    datasets = load_datasets(BASE_DIR)
    with (BASE_DIR / "all_examples.txt").open("r", encoding="utf-8") as examples_file:
        examples_list = examples_file.readlines()

    with ExitStack() as stack:
        model_clients = create_model_clients(args, stack)
        for class_name, enthymemes_list, contexts_list, image_paths in datasets:
            rows = []
            for test_n, (enthymeme, context, image_path) in enumerate(
                zip(enthymemes_list, contexts_list, image_paths), start=1
            ):
                responses = {}
                for experiment in EXPERIMENTS:
                    print(f"{class_name}: case {test_n}/{len(enthymemes_list)}, {experiment}")
                    responses[experiment] = run_experiment(
                        model_clients, examples_list, enthymeme, context, image_path, experiment
                    )
                # Keep the original case and leave Expected Output for manual entry.
                row = [enthymeme, context, image_path.relative_to(BASE_DIR).as_posix(), ""]
                row.extend(
                    responses[experiment][model_index]
                    for model_index in range(len(model_clients))
                    for experiment in EXPERIMENTS
                )
                rows.append(row)
            output_path = BASE_DIR / "results" / f"{class_name}.txt"
            write_results(output_path, model_clients, rows)
            print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
