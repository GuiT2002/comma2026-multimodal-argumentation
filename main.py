import argparse
import base64
from contextlib import ExitStack
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "deepseek": "deepseek-flash",
    "openai": "gpt-6.1-sol",
    "claude": "claude-sonnet-5-5",
}
CLAUDE_MAX_TOKENS = 8192
EXPERIMENTS = ("without_enthymeme", "without_context", "without_image")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_tests(tests_path):
    enthymemes_list = []
    contexts_list = []

    with open(tests_path, "r", encoding="utf-8-sig") as tests_file:
        for line_number, line in enumerate(tests_file, start=1):
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 2:
                raise ValueError(
                    f"{tests_path}, linha {line_number}: esperadas duas colunas "
                    "separadas por um tab (entimema e contexto). "
                    "Para omitir um campo, mantenha o tab e deixe o campo vazio."
                )
            enthymemes_list.append(columns[0])
            contexts_list.append(columns[1])

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
    tests_dir = base_dir / "testes"
    test_paths = sorted(path for path in tests_dir.glob("*.txt") if path.is_file())
    if len(test_paths) != 3:
        raise ValueError(
            f"{tests_dir} deve conter exatamente três arquivos .txt, um por classe "
            f"(encontrados: {len(test_paths)})."
        )

    datasets = []
    for test_path in test_paths:
        class_name = test_path.stem
        enthymemes_list, contexts_list = load_tests(test_path)
        if not enthymemes_list:
            raise ValueError(f"{test_path}: o arquivo da classe não contém casos de teste.")
        image_paths = [
            base_dir / "images" / class_name / f"test{test_n}.png"
            for test_n in range(1, len(enthymemes_list) + 1)
        ]
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
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
            raise RuntimeError(f"{model}: resposta truncada pelo limite de tokens.")
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
            raise RuntimeError(f"{model}: resposta truncada pelo limite de tokens.")
        text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError(f"{model}: a API não retornou uma resposta textual.")
    return text


def run_experiment(model_clients, examples_list, enthymeme, context, image_path, experiment):
    supplied_enthymeme = enthymeme if experiment != "without_enthymeme" else ""
    supplied_context = context if experiment != "without_context" else ""
    image_base64 = None
    supplied_image = ""
    if experiment == "without_image":
        prompt = build_prompt_without_image(examples_list, supplied_enthymeme, supplied_context)
    else:
        prompt = build_prompt(examples_list, supplied_enthymeme, supplied_context)
        image_base64 = encode_image(image_path)
        supplied_image = (Path("images") / image_path.parent.name / image_path.name).as_posix()

    # Expected Output fica vazio para preenchimento manual na planilha.
    row = [supplied_enthymeme, supplied_context, supplied_image, ""]
    for provider, model, client in model_clients:
        row.append(request_model(provider, model, client, prompt, image_base64))
    return row


def spreadsheet_cell(value):
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")


def write_results(output_path, model_clients, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Enthymeme", "Context", "Image", "Expected Output"]
    header.extend(model for _, model, _ in model_clients)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        output_file.write("\t".join(header) + "\n")
        for row in rows:
            output_file.write("\t".join(spreadsheet_cell(value) for value in row) + "\n")


def nonempty_api_key(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("A chave de API não pode estar vazia.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Executa as três classes de testes/ com os modelos selecionados pelas chaves."
        ),
        allow_abbrev=False,
    )
    for provider, model in MODELS.items():
        parser.add_argument(
            f"--{provider}-api-key", type=nonempty_api_key, metavar="CHAVE",
            help=f"Habilita {model} usando a chave fornecida.",
        )
    args = parser.parse_args(argv)
    if not any(getattr(args, f"{provider}_api_key") is not None for provider in MODELS):
        parser.error(
            "Informe ao menos uma chave: --deepseek-api-key, --openai-api-key "
            "ou --claude-api-key."
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
                for experiment in EXPERIMENTS:
                    print(f"{class_name}: caso {test_n}/{len(enthymemes_list)}, {experiment}")
                    rows.append(run_experiment(
                        model_clients, examples_list, enthymeme, context, image_path, experiment
                    ))
            output_path = BASE_DIR / "results" / f"{class_name}.txt"
            write_results(output_path, model_clients, rows)
            print(f"Resultados salvos em {output_path}")


if __name__ == "__main__":
    main()
