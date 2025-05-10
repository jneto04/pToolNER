import re
import os
import nltk
import random
from pathlib import Path # Recomendado para manipulação de caminhos

# Assumindo que flair e unidecode são dependências necessárias
from flair.data import Sentence
from flair.models import SequenceTagger
from unidecode import unidecode

class PortugueseToolNER:
    """
    Classe para realizar Reconhecimento de Entidades Nomeadas (NER) 
    em textos em português, utilizando modelos pré-treinados (presumivelmente Flair).
    Inclui funcionalidades para carregar corpus, aplicar tagging, filtrar entidades,
    mascarar e gerar saídas em diferentes formatos.
    """

    def __init__(self):
        """
        Inicializa a classe com listas para armazenar dados de sentenças,
        tokens, rótulos e resultados de predições.
        """
        self.sentencesTokensKeysPreds: list[list[str]] = []
        self.sentencesPreds: list[list[str]] = []
        self.sentencesKeys: list[list[str]] = []
        
        self.taggedPlainTexts: list[str] = []
        self.filteredTaggedSentences: list[str] = []
        self.sentenceWithMaskedNE: list[str] = []
        self.taggedSentences: list[str] = [] # Pode ser redundante ou necessitar clarificação de uso
        
        self.sentencesPlain: list[str] = []
        self.sentencesTokens: list[list[str]] = []
        self.sentencesLabels: list[list[str]] = []
        self.sentencesTokenAndLabels: list[list[str]] = []
        
        self.filteredSentencesLabels: list[list[str]] = []
        self.unMaskedPlainSentences: list[str] = []
        self.uniqueStringNames: list[str] = [] # Considerar usar set para performance se a ordem não importar
        self.taggedPlainTextTokenAndLabels: list[list[str]] = []
        self.filteredSentencesTokenAndLabels: list[list[str]] = []

        self.uniqueLabels: list[str] = []
        self.tagger: SequenceTagger | None = None # Inicializa o tagger como None

        # Atributos para sequenceTaggingOnText / OnTheFly
        self.maskedSentencesToken: list[list[str]] = []
        self.maskedSentencesTokenAndLabel: list[list[str]] = []
        self.maskedPlainSentencesToken: list[str] = []
        self.taggedFilesDict: dict[str, list[str]] = {}
        self.namedEntitiesDict: dict[str, list] = {} # O valor é uma lista de tuplas
        self.namedEntitiesByFileDict: dict[str, list] = {} # O valor é uma lista de tuplas


    def __getListLabels(self) -> list[str]:
        """
        Extrai e retorna uma lista de rótulos (labels) únicos do corpus carregado.
        Labels do tipo 'I-' são ignoradas para a lista de rótulos únicos.
        """
        unique_labels_set = set()
        for sentenceLabels in self.sentencesLabels:
            for label in sentenceLabels:
                if not label.startswith('I-'):
                    unique_labels_set.add(label)
        self.uniqueLabels = sorted(list(unique_labels_set)) # Ordenar para consistência
        return self.uniqueLabels

    def __getMaskTokensIndex(self, spans, entitiesToMask: list[str]) -> list[int]:
        """
        Obtém os índices dos tokens que devem ser mascarados com base nas entidades e spans fornecidos.
        Requer que os spans sejam objetos com um método `tokens` e que cada token
        tenha um método `get_tag('label').value`. (Típico de Flair)

        Args:
            spans: Lista de spans de entidades (ex: resultado de sentence.get_spans()).
            entitiesToMask: Lista de strings representando os tipos de entidade a serem mascarados.

        Returns:
            Lista de índices de tokens a serem mascarados.
        """
        toMaskIDX = []
        for en_span in spans: # Renomeado para evitar conflito com 'en' de enumerate
            # Assumindo que en_span é um objeto Span do Flair
            if hasattr(en_span, 'tokens') and callable(getattr(en_span, 'get_tag', None)):
                 # Verifica se é um span de entidade e não um token simples com tag
                if en_span.tag in entitiesToMask: # Verifica a tag do span diretamente
                    for token in en_span.tokens:
                        toMaskIDX.append(token.idx)
            # Fallback se spans for uma lista de tokens individuais (menos provável para get_spans)
            elif hasattr(en_span, 'get_tag') and callable(getattr(en_span, 'get_tag', None)):
                token = en_span # en_span é um token
                if token.get_tag('label').value in entitiesToMask:
                     if token.idx not in toMaskIDX: # Evitar duplicatas se o token fizer parte de múltiplos spans
                        toMaskIDX.append(token.idx)
        return sorted(list(set(toMaskIDX))) # Garante unicidade e ordem

    def __getSpans(self, spans_data: list[tuple[str, str]]) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
        """
        Processa spans de entidades para extrair informações como n-gramas e contagens.

        Args:
            spans_data: Lista de tuplas, onde cada tupla é (texto_do_span, tag_do_span).

        Returns:
            Uma tupla contendo:
            - eNsAndAmount: Lista de tuplas (texto_da_entidade, contagem, tag).
            - nGramsCount: Lista de strings descrevendo a contagem de n-gramas.
            - uniqueLabels: Lista de tags de entidade únicas encontradas.
        """
        uniqueLabels_set = set()
        eNsAndAmount_map = {} # Usar um dicionário para contagem eficiente
        fullNGrams = []

        texts = [span_text for span_text, _ in spans_data]

        for span_text, tag in spans_data:
            length = len(span_text.split(' '))
            fullNGrams.append(length)
            uniqueLabels_set.add(tag)

            key = (span_text, tag)
            if key not in eNsAndAmount_map:
                eNsAndAmount_map[key] = texts.count(span_text) # Contagem baseada apenas no texto

        eNsAndAmount = [(text, str(count), tag) for (text, tag), count in eNsAndAmount_map.items()]
        
        uniqueNGrams = sorted(list(set(fullNGrams)))
        nGramsCount = [f"{ng}-gram: {fullNGrams.count(ng)}" for ng in uniqueNGrams]
        
        return eNsAndAmount, nGramsCount, sorted(list(uniqueLabels_set))

    def __getPossiblesTokens(self, token: str) -> list[str]:
        """
        Gera variações de um token (maiúsculas, minúsculas, capitalizado, com e sem acentos).

        Args:
            token: O token original.

        Returns:
            Lista de variações do token.
        """
        if not token:
            return []
        tokenUniCode = unidecode(token)
        variations = {
            token,
            token.lower(),
            token.upper(),
            token.capitalize(),
            tokenUniCode,
            tokenUniCode.lower(),
            tokenUniCode.upper(),
            tokenUniCode.capitalize()
        }
        return list(variations)

    def __sentenceTokenizer(self, text: str) -> list[str]:
        """
        Tokeniza o texto em sentenças usando o tokenizador Punkt do NLTK para português.

        Args:
            text: O texto a ser tokenizado.

        Returns:
            Lista de sentenças.
        
        Raises:
            LookupError: Se o tokenizador 'punkt' para português não for encontrado.
                         Sugere o download via nltk.download('punkt').
        """
        try:
            sent_detector = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        except LookupError:
            print("Recurso 'punkt' do NLTK não encontrado. Tente nltk.download('punkt')")
            raise
        return sent_detector.tokenize(text)

    def getUniqueNames(self, rawListNames: list[str], listStopNames: list[str]):
        """
        Extrai nomes únicos de uma lista bruta, excluindo stop names e 
        adicionando variações (maiúsculas, minúsculas, sem acento).

        Args:
            rawListNames: Lista de nomes brutos.
            listStopNames: Lista de nomes a serem ignorados (stop names).
        """
        exhaustiveListStopNames = set()
        for sN in listStopNames:
            exhaustiveListStopNames.update(self.__getPossiblesTokens(sN))

        unique_names_set = set(self.uniqueStringNames) # Inicia com os já existentes

        for rawName in rawListNames:
            tokens = rawName.split(' ')
            for token in tokens:
                if token and token not in exhaustiveListStopNames:
                    unique_names_set.update(self.__getPossiblesTokens(token))
        
        self.uniqueStringNames = list(unique_names_set)


    def loadCorpusInCoNLLFormat(self,
                                inputFilePath: str | Path,
                                setEncoding: str = 'utf-8',
                                sepTokenTag: str = ' ',
                                loadPredictedCorpus: bool = False
                               ) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        """
        Carrega um corpus no formato CoNLL.

        Args:
            inputFilePath: Caminho para o arquivo do corpus.
            setEncoding: Encoding do arquivo.
            sepTokenTag: Separador entre token e tag (e predição, se aplicável).
            loadPredictedCorpus: Se True, espera três colunas (token, chave, predição).
                                 Caso contrário, espera duas colunas (token, tag).

        Returns:
            Se loadPredictedCorpus for True: (sentencesTokens, sentencesKeys, sentencesTokensKeysPreds)
            Caso contrário: (sentencesTokens, sentencesLabels, sentencesTokenAndLabels)
        
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            Exception: Para outros erros de parsing.
        """
        self.sentencesTokens, self.sentencesLabels, self.sentencesTokenAndLabels = [], [], []
        self.sentencesKeys, self.sentencesPreds, self.sentencesTokensKeysPreds = [], [], []
        
        tokensInSentence, tagsInSentence, tokenAndTagInSentence = [], [], []
        predsInSentence, keysInSentence, tokenKeyPredInSentence = [], [], []

        try:
            with open(inputFilePath, 'r', encoding=setEncoding) as f:
                content = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {inputFilePath}")
        
        if not content:
            print(f"Arquivo {inputFilePath} está vazio ou não contém o delimitador de sentença '\\n\\n'.")
            if loadPredictedCorpus:
                return [], [], []
            else:
                return [], [], []


        sentences_raw: list[str] = content.split('\n\n')
        dataset = [line.split('\n') for line in sentences_raw if line] # Ignora blocos vazios

        for sentence_lines in dataset:
            for line in sentence_lines:
                parts = line.strip().split(sepTokenTag)
                if not parts or not parts[0]: # Linha vazia ou token vazio
                    continue

                token = parts[0].strip()

                if loadPredictedCorpus:
                    if len(parts) < 3:
                        # print(f"Aviso: Linha malformada (esperava 3 colunas) em modo predicted: '{line}' no arquivo {inputFilePath}")
                        continue
                    key = parts[1].strip()
                    predicted = parts[2].strip()

                    if token and key and predicted:
                        tokensInSentence.append(token)
                        keysInSentence.append(key)
                        predsInSentence.append(predicted)
                        tokenKeyPredInSentence.append(f"{token}{sepTokenTag}{key}{sepTokenTag}{predicted}")
                else:
                    if len(parts) < 2:
                        # print(f"Aviso: Linha malformada (esperava 2 colunas): '{line}' no arquivo {inputFilePath}")
                        continue
                    tag = parts[-1].strip() # Pega o último elemento como tag

                    if token and tag:
                        tokensInSentence.append(token)
                        tagsInSentence.append(tag)
                        tokenAndTagInSentence.append(f"{token}{sepTokenTag}{tag}")
            
            if tokensInSentence: # Adiciona a sentença apenas se ela tiver tokens
                if loadPredictedCorpus:
                    self.sentencesTokens.append(list(tokensInSentence))
                    self.sentencesKeys.append(list(keysInSentence))
                    self.sentencesPreds.append(list(predsInSentence))
                    self.sentencesTokensKeysPreds.append(list(tokenKeyPredInSentence))
                else:
                    self.sentencesTokens.append(list(tokensInSentence))
                    self.sentencesLabels.append(list(tagsInSentence))
                    self.sentencesTokenAndLabels.append(list(tokenAndTagInSentence))

            tokensInSentence.clear()
            tagsInSentence.clear()
            tokenAndTagInSentence.clear()
            predsInSentence.clear()
            keysInSentence.clear()
            tokenKeyPredInSentence.clear()

        if loadPredictedCorpus:
            print(f"Dataset com {len(self.sentencesTokensKeysPreds)} sentenças (preditas) carregado de {inputFilePath}!")
            return self.sentencesTokens, self.sentencesKeys, self.sentencesTokensKeysPreds
        else:
            print(f"Dataset com {len(self.sentencesTokenAndLabels)} sentenças carregado de {inputFilePath}!")
            return self.sentencesTokens, self.sentencesLabels, self.sentencesTokenAndLabels


    def loadCorpusInPlainFormat(self,
                                inputFilePath: str | Path,
                                withNamedEntities: bool = False,
                                acceptableLabels: list[str] | None = None,
                                encoding: str = "utf-8"
                               ) -> list[str] | tuple[list[str], list[list[str]]]:
        """
        Carrega um corpus em formato de texto plano, onde cada linha é uma sentença.

        Args:
            inputFilePath: Caminho para o arquivo do corpus.
            withNamedEntities: Se True, tenta extrair entidades e tags do texto plano.
                               Espera que as tags estejam no formato '<TAG>'.
            acceptableLabels: Lista de rótulos aceitáveis se withNamedEntities for True.
            encoding: Encoding do arquivo.

        Returns:
            Se withNamedEntities for False: Lista de sentenças (self.sentencesPlain).
            Se withNamedEntities for True: Tupla (self.sentencesPlain, self.sentencesPlainWithEntities).
        
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
        """
        self.sentencesPlain = []
        try:
            with open(inputFilePath, 'r', encoding=encoding) as f:
                self.sentencesPlain = [line.strip() for line in f if line.strip()]
            print(f"Dataset com {len(self.sentencesPlain)} sentenças carregado de {inputFilePath}!")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {inputFilePath}")

        if withNamedEntities:
            if acceptableLabels is None:
                # print("Aviso: 'acceptableLabels' não fornecido com 'withNamedEntities=True'. Todas as tags <...> serão processadas.")
                acceptableLabels = [] # Ou defina um comportamento padrão

            self.sentencesPlainWithEntities: list[list[str]] = []
            for sentence in self.sentencesPlain:
                tokens = sentence.split(' ')
                listTokenTag = []
                i = 0
                while i < len(tokens):
                    token = tokens[i]
                    # Verifica se o próximo token é uma tag e se está na lista de aceitáveis
                    if (i + 1 < len(tokens) and 
                        tokens[i+1].startswith('<') and 
                        tokens[i+1].endswith('>') and
                        (not acceptableLabels or tokens[i+1] in acceptableLabels)):
                        
                        tag_raw = tokens[i+1]
                        tag = tag_raw.replace('<', '').replace('>', '')
                        listTokenTag.append(f"{token} {tag}")
                        i += 1 # Pula a tag no próximo loop
                    elif not (token.startswith('<') and token.endswith('>')): # Não é uma tag em si
                        listTokenTag.append(f"{token} O") # Tag 'O' para Out-of-entity
                    i += 1
                
                if listTokenTag: # Adiciona apenas se houver tokens processados
                    self.sentencesPlainWithEntities.append(listTokenTag)
            
            return self.sentencesPlain, self.sentencesPlainWithEntities
        else:
            return self.sentencesPlain

    def loadNamedEntityModel(self, nerTrainedModelPath: str | Path):
        """
        Carrega um modelo NER treinado (presumivelmente Flair).

        Args:
            nerTrainedModelPath: Caminho para o modelo treinado.
        """
        try:
            self.tagger = SequenceTagger.load(nerTrainedModelPath)
            print(f"Modelo NER carregado de: {nerTrainedModelPath}")
        except Exception as e:
            print(f"Erro ao carregar o modelo NER de {nerTrainedModelPath}: {e}")
            self.tagger = None
            raise

    def filterCoNLLCorpusByCategories(self,
                                      acceptableLabels: list[str],
                                      maskForUnacceptLabel: str,
                                      sepTokenTag: str = ' '
                                     ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Filtra um corpus CoNLL carregado (self.sentencesLabels, self.sentencesTokenAndLabels)
        mantendo apenas as categorias de entidades aceitáveis. Outras são substituídas
        pela máscara fornecida.

        Args:
            acceptableLabels: Lista de rótulos de entidade aceitáveis (sem prefixo B-/I-).
            maskForUnacceptLabel: Rótulo a ser usado para entidades não aceitáveis (ex: 'O').
            sepTokenTag: Separador entre token e tag.

        Returns:
            Tupla (filtered_sentencesLabels, filtered_sentencesTokenAndLabels).
        """
        new_sentencesLabels: list[list[str]] = []
        new_sentencesTokenAndLabels: list[list[str]] = []

        for sentence_idx, sentenceLabels in enumerate(self.sentencesLabels):
            auxSentenceLabels: list[str] = []
            for tag_idx, full_tag in enumerate(sentenceLabels):
                # Remove prefixo B- ou I- para verificação
                current_tag_type = full_tag.replace('B-', '').replace('I-', '')
                
                if current_tag_type in acceptableLabels:
                    auxSentenceLabels.append(full_tag)
                else:
                    auxSentenceLabels.append(maskForUnacceptLabel)
            new_sentencesLabels.append(auxSentenceLabels)

        for sentenceTokenAndLabels_entry in self.sentencesTokenAndLabels:
            auxSentenceTokenAnLabels: list[str] = []
            for tokenTag_entry in sentenceTokenAndLabels_entry:
                parts = tokenTag_entry.split(sepTokenTag)
                token = parts[0]
                full_tag = parts[-1] # A tag é a última parte

                current_tag_type = full_tag.replace('B-', '').replace('I-', '')

                if current_tag_type in acceptableLabels:
                    auxSentenceTokenAnLabels.append(f"{token}{sepTokenTag}{full_tag}")
                else:
                    auxSentenceTokenAnLabels.append(f"{token}{sepTokenTag}{maskForUnacceptLabel}")
            new_sentencesTokenAndLabels.append(auxSentenceTokenAnLabels)

        self.filteredSentencesLabels = new_sentencesLabels
        self.filteredSentencesTokenAndLabels = new_sentencesTokenAndLabels
        
        return self.filteredSentencesLabels, self.filteredSentencesTokenAndLabels

    def filterPlainCorpusByCategory(self, 
                                    taggedSentence: str, 
                                    allPlainLabels: list[str], 
                                    acceptableLabels: list[str]
                                   ) -> list[str]:
        """
        Filtra uma sentença em texto plano já "taggeada" (ex: "Palavra <TAG>"),
        removendo tags que não estão em acceptableLabels.

        Args:
            taggedSentence: A sentença com tags inline.
            allPlainLabels: Todas as possíveis tags que podem aparecer (ex: ["<PER>", "<ORG>"]).
            acceptableLabels: Lista de tags aceitáveis.

        Returns:
            Lista contendo a sentença filtrada.
        """
        # Correção: iterativamente construir a nova sentença
        # Primeiro, identificar as labels não aceitáveis
        unAcceptLabels = [label for label in allPlainLabels if label not in acceptableLabels]
        
        # Substituir cada label não aceitável por uma string vazia
        # É mais seguro fazer isso token a token ou com regex mais cuidadoso se houver sobreposições
        # ou se "Palavra <TAG>" for a estrutura rígida.
        # Assumindo que as tags são bem separadas por espaços.
        
        # Uma abordagem mais simples se as tags são palavras isoladas:
        words = taggedSentence.split(' ')
        filtered_words = [word for word in words if word not in unAcceptLabels]
        newSentence = ' '.join(filtered_words)
        
        # Remover espaços múltiplos que podem surgir da remoção de palavras
        newSentence = re.sub(' +', ' ', newSentence).strip()
        
        return [newSentence]


    def _process_single_sentence_for_tagging(self, 
                                             sentence_text: str, 
                                             useTokenizer_flair: bool,
                                             maskNamedEntity: bool,
                                             sepTokenTag: str | None,
                                             entitiesToMask: list[str] | None,
                                             specialTokenToMaskNE: str | None,
                                             useAuxListNE: bool,
                                             auxListNE: list[str] | None,
                                             createOutputListSpans: bool
                                            ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """
        Método auxiliar para processar uma única sentença: aplicar NER, mascarar, extrair spans.
        """
        if self.tagger is None:
            raise ValueError("Modelo NER (tagger) não carregado. Chame loadNamedEntityModel() primeiro.")

        sentence_obj = Sentence(sentence_text.strip(), use_tokenizer=useTokenizer_flair)
        self.tagger.predict(sentence_obj)
        sentenceSpans = sentence_obj.get_spans(label_type='label') # 'label' é o tipo padrão no Flair

        current_masked_tokens: list[str] = []
        current_masked_token_and_label: list[str] = []
        current_sentence_named_entities: list[tuple[str,str]] = [] # (text, tag)

        if maskNamedEntity:
            if not all([sepTokenTag, entitiesToMask, specialTokenToMaskNE]):
                raise ValueError('Para mascaramento, "sepTokenTag", "entitiesToMask" e "specialTokenToMaskNE" são obrigatórios.')
            
            _toMaskIDX = self.__getMaskTokensIndex(sentenceSpans, entitiesToMask)

            if useAuxListNE and auxListNE:
                additional_mask_indices = set()
                for token in sentence_obj.tokens:
                    pTokens = self.__getPossiblesTokens(token.text)
                    if any(pT in auxListNE for pT in pTokens):
                        additional_mask_indices.add(token.idx)
                
                _toMaskIDX = sorted(list(set(_toMaskIDX) | additional_mask_indices))


            last_token_was_mask = False
            for token in sentence_obj.tokens:
                if token.idx in _toMaskIDX:
                    if not last_token_was_mask: # Adiciona o token de máscara apenas uma vez por sequência
                        current_masked_tokens.append(specialTokenToMaskNE)
                        # A tag associada ao token de máscara pode ser a do primeiro token da entidade mascarada
                        # ou uma tag genérica de máscara. Aqui, usa a tag do token atual.
                        current_masked_token_and_label.append(f"{specialTokenToMaskNE}{sepTokenTag}{token.get_tag('label').value}")
                        last_token_was_mask = True
                else:
                    current_masked_tokens.append(token.text)
                    current_masked_token_and_label.append(f"{token.text}{sepTokenTag}{token.get_tag('label').value}")
                    last_token_was_mask = False
        else: # Sem mascaramento, apenas texto tageado
            # self.unMaskedPlainSentences.append(sentence_obj.to_tagged_string())
            # Esta função auxiliar não deve modificar atributos de self diretamente para unMaskedPlainSentences
            # Ela deve retornar os dados para o chamador agregar.
            # Para consistência, vamos retornar tokens e tokens+labels como se fosse mascarado, mas sem máscara.
            for token in sentence_obj.tokens:
                current_masked_tokens.append(token.text) # Na verdade, são os tokens originais
                current_masked_token_and_label.append(f"{token.text}{sepTokenTag or ' '}{token.get_tag('label').value}")


        if createOutputListSpans:
            for span in sentenceSpans:
                current_sentence_named_entities.append((span.text, span.tag))
        
        # Se não houver mascaramento, current_masked_tokens é só a lista de tokens da sentença
        # e current_masked_token_and_label são os tokens com suas tags.
        # Se houver mascaramento, eles contêm os tokens/NEs mascarados.
        
        # Retornando os tokens processados (mascarados ou não) e suas labels, e os spans.
        # O chamador decidirá se são "masked" ou "unmasked" com base no parâmetro maskNamedEntity.
        return current_masked_tokens, current_masked_token_and_label, current_sentence_named_entities


    def _sequence_tagging_logic(self,
                                sentences_to_predict: list[str],
                                identifier: str, # Pode ser nome de arquivo ou ID de texto
                                useTokenizer_flair: bool,
                                maskNamedEntity: bool,
                                createOutputListSpans: bool,
                                createOutputFile: bool,
                                outputFilePath: str | Path | None = None,
                                outFormat: str | None = None,
                                sepTokenTag: str | None = ' ',
                                entitiesToMask: list[str] | None = None,
                                specialTokenToMaskNE: str | None = None,
                                useAuxListNE: bool = False,
                                auxListNE: list[str] | None = None
                               ) -> tuple[list[list[str]], dict[str, list[str]], dict[str, list], dict[str, list]]:
        """
        Lógica principal de tagging de sequência, compartilhada por `sequenceTaggingOnText` e `sequenceTaggingOnTheFly`.
        """
        if self.tagger is None:
            raise ValueError("Modelo NER (tagger) não carregado. Chame loadNamedEntityModel() primeiro.")

        # Listas para acumular resultados de todas as sentenças processadas sob este 'identifier'
        all_processed_tokens_for_identifier: list[list[str]] = [] # Lista de listas de tokens
        all_processed_token_labels_for_identifier: list[list[str]] = [] # Lista de listas de "token<sep>label"
        all_plain_tagged_sentences_for_identifier: list[str] = [] # Lista de sentenças como string tageada

        all_named_entities_for_identifier: list[tuple[str,str]] = [] # Acumula spans de todas as sents para este ID
        # generalNamedEntities é melhor acumulado fora, se for para todos os identifiers

        for sentence_text in sentences_to_predict:
            if not sentence_text.strip():
                continue

            processed_tokens, processed_token_labels, sentence_nes = \
                self._process_single_sentence_for_tagging(
                    sentence_text, useTokenizer_flair, maskNamedEntity,
                    sepTokenTag, entitiesToMask, specialTokenToMaskNE,
                    useAuxListNE, auxListNE, createOutputListSpans
                )
            
            all_processed_tokens_for_identifier.append(processed_tokens)
            all_processed_token_labels_for_identifier.append(processed_token_labels) # Para CoNLL output
            
            # Cria a string da sentença (mascarada ou não)
            # Se maskNamedEntity for True, processed_tokens já contêm o specialTokenToMaskNE.
            # Se maskNamedEntity for False, processed_tokens são os tokens originais,
            # e to_tagged_string() é o Flair quem faz.
            # Para manter a consistência com o código original que usava maskedPlainSentencesToken:
            if maskNamedEntity:
                 all_plain_tagged_sentences_for_identifier.append(' '.join(processed_tokens))
            else:
                # Recria o objeto Sentence para usar o to_tagged_string() original se não mascarar
                # Isso evita a necessidade de reimplementar to_tagged_string() perfeitamente.
                # No entanto, _process_single_sentence_for_tagging já retorna os tokens e labels
                # que podem ser usados para construir a string tageada.
                # O código original usava sentenceToPred.to_tagged_string() para o caso não mascarado.
                # Para simplificar e usar o que já temos:
                # Se não mascarou, processed_token_labels tem "token<sep>label"
                # Podemos juntá-los para formar a sentença tageada, mas o Flair `to_tagged_string`
                # é mais robusto pois lida com <...> em volta das entidades.
                # Vamos simular to_tagged_string() de forma básica ou usar o que temos:
                # A lista all_processed_token_labels_for_identifier é o que seria escrito no CoNLL.
                # Para a versão "plain tagged", se não mascarado, o ideal seria realmente:
                temp_sentence_obj = Sentence(sentence_text.strip(), use_tokenizer=useTokenizer_flair)
                self.tagger.predict(temp_sentence_obj)
                all_plain_tagged_sentences_for_identifier.append(temp_sentence_obj.to_tagged_string())


            if createOutputListSpans:
                all_named_entities_for_identifier.extend(sentence_nes)

        # Armazenar resultados para este identifier
        # O nome da chave no dicionário é o 'identifier' (nome do arquivo ou textId)
        self.taggedFilesDict[str(identifier)] = all_plain_tagged_sentences_for_identifier
        
        # self.maskedSentencesToken e self.maskedSentencesTokenAndLabel
        # Se a intenção é que estes guardem os resultados da ÚLTIMA chamada a sequenceTagging,
        # então devem ser atribuídos aqui. Se for para acumular entre chamadas, a lógica muda.
        # Pelo retorno de sequenceTaggingOnTheFly, parece que é o resultado da chamada atual.
        current_call_masked_tokens = all_processed_tokens_for_identifier # Pode ser mascarado ou não

        if createOutputFile:
            if not outputFilePath or not outFormat:
                raise ValueError('"outputFilePath" e "outputFormat" são obrigatórios para criar arquivo de saída.')
            
            output_file_path = Path(outputFilePath)
            output_file_path.mkdir(parents=True, exist_ok=True) # Garante que o diretório exista
            
            output_filename_base = output_file_path / f"ptTagged-{identifier}"
            
            if outFormat.lower() == 'plain':
                # all_plain_tagged_sentences_for_identifier já contém as sentenças corretas (mascaradas ou flair tagged)
                self.generateOutputFile(outputFileName=str(output_filename_base) + ".txt",
                                        sentences=all_plain_tagged_sentences_for_identifier,
                                        outputFormat='plain')
            elif outFormat.lower() == 'conll':
                # all_processed_token_labels_for_identifier é uma lista de listas [token<sep>label, ...]
                # A função generateOutputFile para CoNLL espera uma lista de listas de strings "token<sep>label"
                self.generateOutputFile(outputFileName=str(output_filename_base) + ".conll",
                                        sentences=all_processed_token_labels_for_identifier, # Passando a lista de listas
                                        outputFormat='CoNLL')
            else:
                print(f"Formato de saída '{outFormat}' não suportado para ptTagged.")


        if createOutputListSpans:
            # Named entities específicas para este identifier (arquivo/texto)
            nEsAndAmount_file, nGramsCountByFile, uniqueLabelsByFile = self.__getSpans(all_named_entities_for_identifier)
            self.namedEntitiesByFileDict[str(identifier)] = nEsAndAmount_file

            if createOutputFile and outputFilePath: # Verifica se o caminho de saída é válido
                output_file_path = Path(outputFilePath) # Garante que é um objeto Path
                fileSpansToOut: list[str] = []
                for uL in uniqueLabelsByFile:
                    fileSpansToOut.append(f'CATEGORY:{uL}\n')
                    for text, count, tag_val in nEsAndAmount_file:
                        if tag_val == uL:
                            fileSpansToOut.append(f'{text}: {count}\n') # Removido .join que não fazia sentido
                    fileSpansToOut.append('\n') # Adiciona uma linha em branco entre categorias

                fileSpansToOut.append('\n-------\n')
                for nGCG in nGramsCountByFile:
                    fileSpansToOut.append(f'{nGCG}\n') # Adiciona \n para cada linha

                self.generateOutputFile(
                    outputFileName=output_file_path / f"NamedEntities-{identifier}.txt",
                    sentences=fileSpansToOut, # Já é uma lista de strings prontas para escrever
                    outputFormat='plain'
                )
        
        # Retorna os tokens (potencialmente mascarados) da chamada atual,
        # o dicionário de arquivos tageados (que é um atributo de self, mas pode ser útil retornar),
        # e os dicionários de entidades.
        return current_call_masked_tokens, self.taggedFilesDict, self.namedEntitiesByFileDict, self.namedEntitiesDict


    def sequenceTaggingOnText(self,
                              rootFolderPath: str | Path,
                              fileExtension: str = '.txt',
                              useTokenizer_flair: bool = False, # Nome do parâmetro flair
                              maskNamedEntity: bool = False,
                              createOutputListSpans: bool = False,
                              createOutputFile: bool = False,
                              outputFilePath: str | Path | None = None,
                              outFormat: str | None = None, # 'plain' ou 'CoNLL'
                              sepTokenTag: str = ' ', # Usado se maskNamedEntity ou para CoNLL
                              entitiesToMask: list[str] | None = None,
                              specialTokenToMaskNE: str | None = None,
                              useAuxListNE: bool = False,
                              auxListNE: list[str] | None = None
                             ) -> tuple[dict[str, list[str]], dict[str, list], dict[str, list]]:
        """
        Aplica NER a todos os arquivos de texto em um diretório.

        Args:
            rootFolderPath: Caminho para a pasta com os arquivos de texto.
            fileExtension: Extensão dos arquivos a serem processados (ex: '.txt').
            useTokenizer_flair: Se o tokenizador interno do Flair deve ser usado para a sentença.
            maskNamedEntity: Se True, mascara as entidades nomeadas.
            createOutputListSpans: Se True, cria uma lista de spans de entidades.
            createOutputFile: Se True, gera arquivos de saída com os resultados.
            outputFilePath: Caminho da pasta para salvar os arquivos de saída.
            outFormat: Formato do arquivo de saída ('plain' ou 'CoNLL').
            sepTokenTag: Separador entre token e tag.
            entitiesToMask: Lista de tipos de entidade a serem mascarados.
            specialTokenToMaskNE: Token especial para substituir entidades mascaradas.
            useAuxListNE: Se True, usa uma lista auxiliar de NEs para mascaramento adicional.
            auxListNE: Lista auxiliar de NEs.

        Returns:
            Tupla (taggedFilesDict, namedEntitiesByFileDict, namedEntitiesDict (geral)).
        """
        if self.tagger is None:
            raise ValueError("Modelo NER (tagger) não carregado. Chame loadNamedEntityModel() primeiro.")

        root_path = Path(rootFolderPath)
        files = [f for f in root_path.iterdir() if f.is_file() and f.suffix == fileExtension]

        # Limpa/reseta dicionários de estado que são preenchidos por esta função
        self.taggedFilesDict.clear()
        self.namedEntitiesByFileDict.clear()
        self.namedEntitiesDict.clear() # Para as entidades gerais de todos os arquivos

        generalNamedEntities_all_files: list[tuple[str,str]] = []

        for file_path in files:
            print(f" :: Tagging Text: {file_path.name}")
            # Carrega o conteúdo do arquivo como uma lista de sentenças
            # Assume que loadCorpusInPlainFormat retorna uma lista de strings (sentenças)
            # e não lida com withNamedEntities=True aqui, pois o tagging é feito pelo Flair.
            sentencesToPredict = self.loadCorpusInPlainFormat(file_path, withNamedEntities=False)
            if not isinstance(sentencesToPredict, list): # Garante que é uma lista de strings
                 print(f"Aviso: loadCorpusInPlainFormat não retornou uma lista para {file_path.name}")
                 sentencesToPredict = []


            _, _, _, current_file_nes_dict = self._sequence_tagging_logic(
                sentences_to_predict=sentencesToPredict,
                identifier=file_path.name,
                useTokenizer_flair=useTokenizer_flair,
                maskNamedEntity=maskNamedEntity,
                createOutputListSpans=createOutputListSpans,
                createOutputFile=createOutputFile,
                outputFilePath=outputFilePath,
                outFormat=outFormat,
                sepTokenTag=sepTokenTag,
                entitiesToMask=entitiesToMask,
                specialTokenToMaskNE=specialTokenToMaskNE,
                useAuxListNE=useAuxListNE,
                auxListNE=auxListNE
            )
            
            # Acumula entidades para o relatório geral, se createOutputListSpans for True
            if createOutputListSpans:
                 # self.namedEntitiesByFileDict já foi preenchido por _sequence_tagging_logic
                 # Agora precisamos obter os spans brutos para acumular em generalNamedEntities
                 # Isso requer que _sequence_tagging_logic retorne os spans brutos do arquivo atual,
                 # o que não está fazendo. Precisamos refatorar para obter isso.
                 # Por agora, vamos assumir que podemos acessar os spans brutos de alguma forma.
                 # Temporariamente, vamos reconstruir a partir do dicionário (não ideal)
                 # Ou melhor, _sequence_tagging_logic deveria retornar os spans do identifier atual.
                 # Para este exemplo, vou simular pegando do dicionário.
                 # O ideal seria: all_nes_current_file = retornado_por_logic
                 # generalNamedEntities_all_files.extend(all_nes_current_file)
                 
                 # Assumindo que _getSpans foi chamado dentro de _sequence_tagging_logic
                 # e namedEntitiesByFileDict[file_path.name] contém [(text, count, tag), ...]
                 # Precisamos de [(text, tag), ...]
                 # Esta parte precisa de um retorno mais explícito de _sequence_tagging_logic
                 # ou que ele popule uma lista de spans brutos passada como argumento.
                 # Para uma solução rápida, mas não ideal:
                if file_path.name in self.namedEntitiesByFileDict:
                    spans_with_counts = self.namedEntitiesByFileDict[file_path.name]
                    for text, _, tag_val in spans_with_counts: # Ignora a contagem para a lista geral
                        generalNamedEntities_all_files.append((text, tag_val))


        if createOutputListSpans and generalNamedEntities_all_files:
            generalSpansToOut: list[str] = []
            generalNEsAndAmount, nGramsCountGeneral, uniqueLabels = self.__getSpans(generalNamedEntities_all_files)
            self.namedEntitiesDict['allFiles'] = generalNEsAndAmount # Armazena no atributo da classe

            if createOutputFile and outputFilePath:
                output_file_path = Path(outputFilePath)
                for uL in uniqueLabels:
                    generalSpansToOut.append(f'CATEGORY:{uL}\n')
                    for text, count, tag_val in generalNEsAndAmount:
                        if tag_val == uL:
                            generalSpansToOut.append(f'{text}: {count}\n')
                    generalSpansToOut.append('\n')

                generalSpansToOut.append('\n-------\n')
                for nGCG in nGramsCountGeneral:
                    generalSpansToOut.append(f'{nGCG}\n')

                self.generateOutputFile(
                    outputFileName=output_file_path / "GeneralNamedEntities.txt",
                    sentences=generalSpansToOut,
                    outputFormat='plain'
                )
        
        return self.taggedFilesDict, self.namedEntitiesByFileDict, self.namedEntitiesDict


    def sequenceTaggingOnTheFly(self,
                                textToPredict: str,
                                textId: int | str, # Permite ID de string também
                                useSentenceTokenize_nltk: bool = True, # Especifica que é NLTK
                                useTokenizer_flair: bool = False, # Para Flair Sentence
                                maskNamedEntity: bool = False,
                                createOutputListSpans: bool = False,
                                createOutputFile: bool = False,
                                outputFilePath: str | Path | None = None,
                                outFormat: str | None = None, # 'plain' ou 'CoNLL'
                                sepTokenTag: str = ' ', # Usado se maskNamedEntity ou para CoNLL
                                entitiesToMask: list[str] | None = None,
                                specialTokenToMaskNE: str | None = None,
                                useAuxListNE: bool = False,
                                auxListNE: list[str] | None = None
                               ) -> tuple[str | int, list[list[str]], dict[str, list[str]], dict[str, list], dict[str, list]]:
        """
        Aplica NER a um texto fornecido dinamicamente.

        Args:
            textToPredict: O texto a ser processado.
            textId: Um identificador para este texto.
            useSentenceTokenize_nltk: Se True, usa NLTK para dividir o texto em sentenças.
            useTokenizer_flair: Se o tokenizador interno do Flair deve ser usado para a sentença.
            maskNamedEntity: Se True, mascara as entidades nomeadas.
            ... (demais argumentos similares a sequenceTaggingOnText)

        Returns:
            Tupla (textId, maskedSentencesToken (da chamada atual), 
                   taggedFilesDict, namedEntitiesByFileDict, namedEntitiesDict (geral)).
        """
        if self.tagger is None:
            raise ValueError("Modelo NER (tagger) não carregado. Chame loadNamedEntityModel() primeiro.")

        if useSentenceTokenize_nltk:
            sentencesToPredict = self.__sentenceTokenizer(textToPredict)
        else:
            sentencesToPredict = [textToPredict]
        
        # Limpa/reseta dicionários de estado se esta função for projetada para operar isoladamente
        # ou se os resultados de chamadas anteriores não devem se misturar sem controle.
        # O comportamento original parecia acumular em self.taggedFilesDict, etc.
        # Se cada chamada "OnTheFly" deve ser independente para namedEntitiesDict['allFiles']:
        # self.namedEntitiesDict.clear() # Descomente se necessário um reset geral aqui.
        # self.namedEntitiesByFileDict.clear() # Limpa para este ID específico

        # A lógica de _sequence_tagging_logic já lida com a adição ao self.taggedFilesDict
        # e self.namedEntitiesByFileDict usando o 'textId'.

        current_call_masked_tokens, tagged_files_dict, id_specific_nes_dict, _ = \
            self._sequence_tagging_logic(
                sentences_to_predict=sentencesToPredict,
                identifier=str(textId),
                useTokenizer_flair=useTokenizer_flair,
                maskNamedEntity=maskNamedEntity,
                createOutputListSpans=createOutputListSpans,
                createOutputFile=createOutputFile,
                outputFilePath=outputFilePath,
                outFormat=outFormat,
                sepTokenTag=sepTokenTag,
                entitiesToMask=entitiesToMask,
                specialTokenToMaskNE=specialTokenToMaskNE,
                useAuxListNE=useAuxListNE,
                auxListNE=auxListNE
            )
        
        # Lógica para "GeneralNamedEntities" (acumulando de múltiplas chamadas OnTheFly)
        # Se createOutputListSpans for True, as entidades deste textId foram adicionadas
        # ao self.namedEntitiesByFileDict[str(textId)].
        # Se a intenção é ter um "GeneralNamedEntities.txt" que acumule de todas as chamadas
        # OnTheFly (ou uma combinação com OnText), a lógica de acumulação
        # para 'generalNamedEntities' e a sua escrita precisam ser cuidadosamente gerenciadas.
        # O código original parecia recriar/sobrescrever GeneralNamedEntities.txt a cada chamada.

        # Para este exemplo, vamos assumir que se createOutputListSpans e createOutputFile
        # forem verdadeiros, ele gera o NamedEntities-<textId>.txt e também tenta
        # atualizar/recriar o GeneralNamedEntities.txt com base no estado ATUAL de self.namedEntitiesByFileDict.
        # Esta parte pode precisar de um design mais robusto se o acúmulo for complexo.
        
        if createOutputListSpans and createOutputFile and outputFilePath:
            all_accumulated_nes: list[tuple[str,str]] = []
            for id_key in self.namedEntitiesByFileDict: # Acumula de todos os IDs processados até agora
                 spans_with_counts = self.namedEntitiesByFileDict[id_key]
                 for text, _, tag_val in spans_with_counts:
                     all_accumulated_nes.append((text, tag_val))
            
            if all_accumulated_nes: # Só gera se houver entidades
                generalSpansToOut: list[str] = []
                generalNEsAndAmount, nGramsCountGeneral, uniqueLabels = self.__getSpans(all_accumulated_nes)
                self.namedEntitiesDict['allFiles'] = generalNEsAndAmount # Atualiza o geral

                output_file_path = Path(outputFilePath)
                for uL in uniqueLabels:
                    generalSpansToOut.append(f'CATEGORY:{uL}\n')
                    for text, count, tag_val in generalNEsAndAmount:
                        if tag_val == uL:
                            generalSpansToOut.append(f'{text}: {count}\n')
                    generalSpansToOut.append('\n')

                generalSpansToOut.append('\n-------\n')
                for nGCG in nGramsCountGeneral:
                    generalSpansToOut.append(f'{nGCG}\n')

                self.generateOutputFile(
                    outputFileName= output_file_path / "GeneralNamedEntities.txt",
                    sentences=generalSpansToOut,
                    outputFormat='plain'
                )
        
        return textId, current_call_masked_tokens, tagged_files_dict, id_specific_nes_dict, self.namedEntitiesDict


    def generateOutputFile(self,
                           outputFileName: str | Path,
                           sentences: list[str] | list[list[str]], # Pode ser lista de sentenças (strings) ou lista de listas de "token-tag"
                           outputFormat: str,
                           shuffleSentences: bool = False,
                           encoding: str = 'utf-8'):
        """
        Gera um arquivo de saída com as sentenças processadas.

        Args:
            outputFileName: Nome/caminho do arquivo de saída.
            sentences: Lista de sentenças. Para CoNLL, espera-se uma lista de listas,
                       onde cada sublista contém strings "token<sep>tag".
                       Para Plain, uma lista de strings (sentenças).
            outputFormat: Formato de saída ('CoNLL' ou 'Plain').
            shuffleSentences: Se True, embaralha as sentenças antes de salvar.
            encoding: Encoding do arquivo de saída.
        """
        output_path = Path(outputFileName)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Garante que o diretório pai exista

        # Copia a lista para não modificar a original se shuffle for True
        sentences_to_write = list(sentences)

        if shuffleSentences:
            random.shuffle(sentences_to_write)

        try:
            with open(output_path, 'w', encoding=encoding) as outputFile:
                if outputFormat.lower() == 'conll':
                    # sentences_to_write é esperado ser list[list[str]]
                    # onde cada sub-lista são os "token<sep>tag" de uma sentença
                    for sentence_tokens_and_tags in sentences_to_write:
                        if isinstance(sentence_tokens_and_tags, list): # Verifica se é uma lista de "token-tag"
                            for tokenTag in sentence_tokens_and_tags:
                                outputFile.write(str(tokenTag) + '\n')
                            outputFile.write('\n') # Delimitador de sentença para CoNLL
                        else:
                            print(f"Aviso: Esperava uma lista de 'token-tag' para o formato CoNLL, mas recebi: {type(sentence_tokens_and_tags)}")
                
                elif outputFormat.lower() == 'plain':
                    # sentences_to_write é esperado ser list[str]
                    for sentence_line in sentences_to_write:
                        outputFile.write(str(sentence_line) + '\n')
                else:
                    print(f"Formato de saída '{outputFormat}' não reconhecido. Nenhum arquivo gerado.")
            
            # print(f"Arquivo gerado com sucesso: {output_path}")

        except IOError as e:
            print(f"Erro de I/O ao escrever o arquivo {output_path}: {e}")
        except Exception as e:
            print(f"Erro inesperado ao gerar o arquivo {output_path}: {e}")
