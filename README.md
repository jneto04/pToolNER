# pToolNER
Ferramenta para trabalhar com Reconhecimento de Entidades Nomeadas em Português.

# Exemplos

## Como fazer a filtragem de Entidades Nomeadas permitindo apenas as EN do tipo PESSOA e LOCAL.
Aqui um corpus em formato CoNLL é carregado e em seguida um fitro deixa apenas as categorias PESSOA e LOCAL e substitui as demais categorias pela máscara 'O'.

```python
pToolNER = PortugueseToolNER()

pToolNER.loadCorpusInCoNLLFormat(
                 inputFilePath='InputCorpus.txt',
                 setEncoding='utf8',
                 sepTokenTag=' ')

pToolNER.filterCoNLLCorpusByCategories(
                    acceptableLabels=['PER', 'LOC'],
                    maskForUnacceptLabel='O',
                    sepTokenTag=' ')

pToolNER.generateOutputFile(
               outputFileName='FilteredCorpus.txt',
               sentences=pToolNER.sentencesTokenAndLabels,
               outputFormat='CoNLL')
```

## Como fazer a rotulção (sem uso de Máscara) de todos os arquivos .txt em uma pasta.

```python
pToolNER = PortugueseToolNER()

pToolNER.sequenceTaggingOnText(
               nerTrainedModelPath='best-model.pt',
               rootFolderPath='./PredictablesFiles',
               fileExtension='.txt',
               useTokenizer=True,
               maskNamedEntity=False,
               createOutputFile=True,
               outputFilePath='./TaggedTexts',
               outputFormat='plain',
               createOutputListSpans=True
               )
```
## Como fazer rotulação de textos usando Máscara para categorias específicas de Entidades Nomeadas (EN).
Aqui as EN PESSOA e LOCAL serão substituídas por um único simbolo '[HIDDEN-INFO]'.

```python
pToolNER = PortugueseToolNER()

pToolNER.sequenceTaggingOnText(
               nerTrainedModelPath='/content/drive/My Drive/NER-Models/Final/ourFlair+W2V-SKPG-NILC/best-model.pt',
               rootFolderPath='./PredictablesFiles',
               fileExtension='.txt',
               useTokenizer=True,
               maskNamedEntity=True,
               specialTokenToMaskNE='[HIDDEN-INFO]',
               sepTokenTag=' ',
               entitiesToMask=['B-PER', 'I-PER', 'B-LOC', 'I-LOC'],
               createOutputFile=True,
               outputFilePath='./TaggedTexts',
               outputFormat='plain',
               createOutputListSpans=True
               )
```
