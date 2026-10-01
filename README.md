# Experimentos de argumentação multimodal

Executa experimentos de formalização de argumentos com DeepSeek, OpenAI e/ou Claude. Cada execução percorre as três classes de testes e, para cada caso, envia três combinações de entrada aos modelos selecionados:

| Ordem | Experimento | Informações enviadas |
| --- | --- | --- |
| 1 | `without_enthymeme` | Contexto e imagem |
| 2 | `without_context` | Entimema e imagem |
| 3 | `without_image` | Entimema e contexto |

O experimento sem imagem usa o prompt específico sem referências a imagens. Os outros dois usam o prompt multimodal original. Os exemplos dos esquemas de argumentação são carregados de `all_examples.txt`.

## Instalação

Requer Python 3.10 ou superior. Na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

As dependências são os SDKs `openai` e `anthropic`. O SDK da OpenAI também é usado para acessar a API da DeepSeek.

## Dados de entrada

Prepare exatamente três arquivos `.txt` na pasta `testes/`. Os nomes são livres; o nome sem a extensão identifica a classe e sua pasta de imagens:

```text
repositorio/
├── main.py
├── requirements.txt
├── all_examples.txt
├── testes/
│   ├── classe1.txt
│   ├── classe2.txt
│   └── classe3.txt
└── images/
    ├── classe1/
    │   ├── test1.png
    │   └── test2.png
    ├── classe2/
    │   ├── test1.png
    │   └── test2.png
    └── classe3/
        ├── test1.png
        └── test2.png
```

Cada TXT deve estar em UTF-8, sem cabeçalho, com um caso por linha e exatamente duas colunas separadas por um tab real:

```text
entimema<TAB>contexto
```

Substitua `<TAB>` pelo caractere de tabulação; não escreva os caracteres `\t`. Para deixar um campo vazio, preserve o tab. Não inclua linhas vazias nem tabs ou quebras de linha dentro de um campo. O código mantém os entimemas e os contextos em listas separadas.

`test1.png` corresponde à primeira linha do TXT da classe, `test2.png` à segunda, e assim por diante. A numeração reinicia em cada classe. As classes podem ter quantidades diferentes de casos, mas cada classe precisa conter ao menos um caso e cada caso precisa ter sua imagem, pois os três experimentos sempre são executados.

Todos os caminhos de dados e resultados são relativos à pasta de `main.py`, independentemente do diretório de onde o comando for chamado. Os três arquivos e as imagens são verificados antes das chamadas às APIs. É necessário fornecer os dados das três classes; o programa não cria nem divide automaticamente o conjunto de testes.

## Executar os testes

Selecione os provedores passando suas chaves na linha de comando. É obrigatório selecionar pelo menos um:

| Opção | Modelo |
| --- | --- |
| `--deepseek-api-key` | `deepseek-flash` |
| `--openai-api-key` | `gpt-6.1-sol` |
| `--claude-api-key` | `claude-sonnet-5-5` |

Somente DeepSeek:

```bash
python3 main.py --deepseek-api-key "SUA_CHAVE_DEEPSEEK"
```

Somente OpenAI ou somente Claude:

```bash
python3 main.py --openai-api-key "SUA_CHAVE_OPENAI"
python3 main.py --claude-api-key "SUA_CHAVE_CLAUDE"
```

Os três modelos na mesma execução:

```bash
python3 main.py \
  --deepseek-api-key "SUA_CHAVE_DEEPSEEK" \
  --openai-api-key "SUA_CHAVE_OPENAI" \
  --claude-api-key "SUA_CHAVE_CLAUDE"
```

Também é possível combinar quaisquer dois provedores. Somente os provedores com chave informada no comando são executados; variáveis de ambiente ou um arquivo `.env`, por si só, não os ativam. As chaves não são incluídas nos arquivos de resultados.

As três classes são descobertas automaticamente em `testes/`; não passe nomes de arquivos como argumentos. Cada caso gera três chamadas por modelo selecionado. Para consultar as opções:

```bash
python3 main.py --help
```

## Resultados

Ao concluir cada classe, o programa grava `results/<classe>.txt`, sobrescrevendo o resultado anterior daquela classe. Uma execução completa gera três arquivos UTF-8 tabulados, prontos para copiar e colar em uma planilha.

O cabeçalho começa com `Enthymeme`, `Context`, `Image` e `Expected Output`, seguidos de uma coluna por modelo selecionado. As colunas de modelos aparecem sempre na ordem DeepSeek, OpenAI e Claude, usando seus identificadores como nomes. Com os três modelos, a estrutura é:

```text
Enthymeme<TAB>Context<TAB>Image<TAB>Expected Output<TAB>deepseek-flash<TAB>gpt-6.1-sol<TAB>claude-sonnet-5-5
```

No arquivo gerado, os separadores são tabs reais. `Expected Output` fica sempre em branco para preenchimento manual; não é necessário fornecer um arquivo de respostas esperadas.

Cada caso ocupa três linhas consecutivas, na ordem `without_enthymeme`, `without_context` e `without_image`. O campo da modalidade omitida fica vazio. `Image` contém o caminho da imagem quando ela é enviada. Não há uma coluna adicional para o nome do experimento.

Cada célula de resposta contém o texto retornado pelo modelo, incluindo suas tags, sem extrair apenas `<final_output>`. Tabs e quebras de linha dentro das células são substituídos por espaços para preservar uma linha da planilha por experimento.

## Erros comuns

- **Nenhum provedor selecionado:** informe pelo menos uma das opções de chave de API.
- **Quantidade incorreta de classes:** mantenha exatamente três arquivos `.txt` diretamente em `testes/`.
- **Entrada inválida:** confira os tabs reais, as duas colunas por linha e a ausência de linhas vazias.
- **Imagem não encontrada:** confira a correspondência entre o nome do TXT, a pasta em `images/` e a numeração `test1.png`, `test2.png` etc.
- **Dependência ausente:** ative o ambiente virtual e instale `requirements.txt`.
- **Erro da API:** confira a chave, o acesso ao modelo e a mensagem retornada pelo provedor. A execução é interrompida; os resultados de classes já concluídas permanecem salvos. A classe interrompida não é gravada, e um arquivo antigo dessa classe, se existir, permanece inalterado.

Respostas vazias ou truncadas também interrompem a execução. O limite de saída do Claude é de 8192 tokens, definido por `CLAUDE_MAX_TOKENS` em `main.py`.

Documentação dos provedores: [DeepSeek](https://api-docs.deepseek.com/), [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) e [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).
