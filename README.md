# pToolNER
Ferramenta para trabalhar com Reconhecimento de Entidades Nomeadas em Português.

Exemplo de comofazer a rotulção de todos os arquivos .txt em uma pasta.

```python
pToolNER = PortugueseToolNER()

pToolNER.sequenceTaggingOnText(nerTrainedModelPath='/home/joaquim/Área de Trabalho/best-model.pt',
                 rootFolderPath='./',
                 fileExtension='.txt',
                 maskNamedEntity=False,
                 createOutputFile=True,
                 outputFilePath='./',
                 outputFormat='plain'
                 )
```
