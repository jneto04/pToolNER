
# pToolNER

`pToolNER` é uma ferramenta desenvolvida em Python para auxiliar em tarefas de Reconhecimento de Entidades Nomeadas (NER) em textos em língua portuguesa. Ela permite carregar corpus, aplicar modelos de NER (presumivelmente baseados na biblioteca Flair), filtrar entidades, mascarar informações sensíveis e gerar saídas em diferentes formatos.

## Instalação

Antes de usar a ferramenta, certifique-se de ter as seguintes dependências instaladas:

```bash
pip install nltk flair unidecode
```

Além disso, para a tokenização de sentenças com NLTK, você precisará do recurso `punkt`:

```python
import nltk
nltk.download('punkt')
```

## Principais Funcionalidades

- Carregamento de corpus nos formatos CoNLL e texto plano.
- Aplicação de modelos NER (Flair) para rotulagem de entidades.
- Filtragem de entidades nomeadas por categorias específicas.
- Mascaramento de entidades nomeadas com tokens especiais.
- Uso de listas auxiliares para expandir o escopo do mascaramento.
- Rotulagem de textos em arquivos ou "on the fly".
- Geração de arquivos de saída nos formatos CoNLL e texto plano.

## Exemplos de Uso

> **Nota sobre `entitiesToMask`:** Ao especificar entidades para mascaramento, utilize os tipos base da entidade (ex: `PER`, `LOC`). A ferramenta é projetada para lidar com as tags retornadas pelo modelo Flair, que geralmente não incluem os prefixos `B-` ou `I-`.

### 1. Filtragem de Entidades Nomeadas em Corpus CoNLL

```python
from pToolNER import PortugueseToolNER

tool = PortugueseToolNER()

tool.loadCorpusInCoNLLFormat(
    inputFilePath='InputCorpus.txt',
    setEncoding='utf-8',
    sepTokenTag=' '
)

_, filtered_sentences_token_and_labels = tool.filterCoNLLCorpusByCategories(
    acceptableLabels=['PER', 'LOC'],
    maskForUnacceptLabel='O',
    sepTokenTag=' '
)

tool.generateOutputFile(
    outputFileName='FilteredCorpus.txt',
    sentences=filtered_sentences_token_and_labels,
    outputFormat='CoNLL',
    shuffleSentences=False
)
```

### 2. Rotulagem de Arquivos `.txt` em uma Pasta (Sem Máscara)

```python
tool.loadNamedEntityModel('best-model.pt')

tool.sequenceTaggingOnText(
    rootFolderPath='./PredictablesFiles',
    fileExtension='.txt',
    useTokenizer_flair=True,
    maskNamedEntity=False,
    createOutputFile=True,
    outputFilePath='./TaggedTexts',
    outputFormat='plain',
    createOutputListSpans=True
)
```

### 3. Rotulagem com Máscara para Categorias Específicas

```python
tool.loadNamedEntityModel('best-model.pt')

tool.sequenceTaggingOnText(
    rootFolderPath='./PredictablesFiles',
    fileExtension='.txt',
    useTokenizer_flair=True,
    maskNamedEntity=True,
    specialTokenToMaskNE='[INFO-SIGILOSA]',
    sepTokenTag=' ',
    entitiesToMask=['PER', 'LOC'],
    createOutputFile=True,
    outputFilePath='./MaskedTexts',
    outputFormat='plain',
    createOutputListSpans=True
)
```

### 4. Rotulagem com Máscara e Lista Auxiliar de Nomes

```python
listStopNames = ['da', 'de', 'do', 'dos']
listNamesToMask = ['Manoel Francisco', 'Vbrs']

tool.getUniqueNames(rawListNames=listNamesToMask, listStopNames=listStopNames)

tool.loadNamedEntityModel('best-model.pt')

tool.sequenceTaggingOnText(
    rootFolderPath='./PredictablesFiles',
    fileExtension='.txt',
    useTokenizer_flair=True,
    maskNamedEntity=True,
    specialTokenToMaskNE='[INFO-SIGILOSA]',
    sepTokenTag=' ',
    entitiesToMask=['PER', 'LOC'],
    useAuxListNE=True,
    auxListNE=tool.uniqueStringNames,
    createOutputFile=True,
    outputFilePath='./MaskedTextsWithAuxList',
    outputFormat='plain',
    createOutputListSpans=True
)
```

### 5. Rotulagem "On The Fly" (Texto Direto como Entrada)

```python
tool.loadNamedEntityModel('best-model.pt')

texto_exemplo = "Finalmente o Manoel Francisco conseguiu encontrar com Xdfghuy."

text_id, masked_tokens, tagged_files_dict, _, _ = tool.sequenceTaggingOnTheFly(
    textToPredict=texto_exemplo,
    textId="exemplo_on_the_fly_1",
    useSentenceTokenize_nltk=True,
    useTokenizer_flair=True,
    maskNamedEntity=True,
    specialTokenToMaskNE='[INFO-SIGILOSA]',
    sepTokenTag=' ',
    entitiesToMask=['ORG'],
    useAuxListNE=True,
    auxListNE=tool.uniqueStringNames,
    createOutputFile=True,
    outputFilePath='./TaggedSingleTexts',
    outputFormat='plain',
    createOutputListSpans=True
)

print(f"Texto processado (ID: {text_id}):")
if tagged_files_dict and str(text_id) in tagged_files_dict:
    for sentence in tagged_files_dict[str(text_id)]:
        print(sentence)
```

### 6. Anonimização Rápida "On The Fly"

```python
tool.loadNamedEntityModel('best-model.pt')

texto_para_anonimizar = "Qtgho Vbrs será lançado no próximo mês em São Paulo."

_, masked_output_tokens, tagged_output_dict, _, _ = tool.sequenceTaggingOnTheFly(
    textToPredict=texto_para_anonimizar,
    textId="anon_exemplo_2",
    useSentenceTokenize_nltk=True,
    useTokenizer_flair=True,
    maskNamedEntity=True,
    specialTokenToMaskNE='[DADO_OCULTO]',
    sepTokenTag=' ',
    entitiesToMask=['PER', 'LOC', 'ORG'],
    useAuxListNE=False,
    createOutputFile=False,
    createOutputListSpans=False
)

print(f"Texto anonimizado (ID: anon_exemplo_2):")
if tagged_output_dict and "anon_exemplo_2" in tagged_output_dict:
    for sentence in tagged_output_dict["anon_exemplo_2"]:
        print(sentence)
```

> **Exemplo de saída esperada:**  
> `"Qtgho [DADO_OCULTO] será lançado no próximo mês em [DADO_OCULTO]."`

---

Adapte os caminhos dos arquivos, nomes de modelos e listas de entidades conforme necessário.
