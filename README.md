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

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnText(
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

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnText(
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

## Como fazer rotulação de textos usando Máscara para EN e Lista auxiliar de EN.
```python
pToolNER = PortugueseToolNER()

listStopNames = ['da', 'de', 'do', 'dos']
listNames = ['name name name', 'name', 'name name']

pToolNER.getUniqueNames(listNames, listStopNames)

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnText(
               rootFolderPath='./PredictablesFiles',
               fileExtension='.txt',
               useTokenizer=True,
               maskNamedEntity=True,
               specialTokenToMaskNE='[HIDDEN-INFO]',
               sepTokenTag=' ',
               entitiesToMask=['B-PER', 'I-PER', 'B-LOC', 'I-LOC'],
               useAuxListNE=True,
               auxListNE=pToolNER.uniqueStringNames,
               createOutputFile=True,
               outputFilePath='./TaggedTexts',
               outputFormat='plain',
               createOutputListSpans=True
               )
```
## Como fazer rotulação de sequencia de sentenças _(On the fly)_ sem carregar arquivos de texto.
```python
pToolNER = PortugueseToolNER()

listStopNames = ['da', 'de', 'do', 'dos']
listNames = ['name name name', 'name', 'name name']

pToolNER.getUniqueNames(listNames, listStopNames)

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnTheFly(
			   textToPredict = 'Put Your Sentence Here.',
			   textId = 1,
			   useTokenizer=True,
			   useSentenceTokenize=True,
			   maskNamedEntity=True,
			   specialTokenToMaskNE='[HIDDEN-INFO]',
			   sepTokenTag=' ',
			   entitiesToMask=['B-PER', 'I-PER', 'B-LOC', 'I-LOC'],
			   useAuxListNE=True,
			   auxListNE=pToolNER.uniqueStringNames,
			   createOutputFile=True,
			   outputFilePath='./TaggedTexts',
			   outputFormat='plain',
			   createOutputListSpans=True
			   )
```
## Anonimização de Entidades Nomeadas
Como fazer rotulação de sequencia de sentenças _(On the fly)_ sem carregar arquivos de texto e lista auxiliar.
```python
pToolNER = PortugueseToolNER()

pToolNER.loadNamedEntityModel('best-model.pt')

pToolNER.sequenceTaggingOnTheFly(
          textToPredict = 'Put Your Sentence Here.',
          textId = 1,
          useTokenizer=True,
          useSentenceTokenize=True,
          maskNamedEntity=True,
          specialTokenToMaskNE='[HIDDEN-INFO]',
          sepTokenTag=' ',
          entitiesToMask=['B-PER', 'I-PER', 'B-LOC', 'I-LOC'],
          useAuxListNE=False,
          createOutputFile=False,
	  createOutputListSpans=False
          )
```
