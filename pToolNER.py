import re
import os
from flair.data import Sentence
from flair.models import SequenceTagger
from unidecode import unidecode

class PortugueseToolNER:

	def __init__(self):
		self.taggedPlainTexts 		 : list[str] = []
		self.filteredTaggedSentences : list[str] = []
		self.sentenceWithMaskedNE 	 : list[str] = []
		self.taggedSentences 		 : list[str] = []
		self.sentencesPlain 		 : list[str] = []
		self.sentencesTokens 		 : list[str] = []
		self.sentencesLabels 		 : list[str] = []
		self.sentencesTokenAndLabels : list[str] = []
		self.filteredSentencesLabels : list[str] = []
		self.unMaskedPlainSentences	 : list[str] = []
		self.uniqueStringNames		 : list[str] = []
		self.taggedPlainTextTokenAndLabels	 : list[str] = []
		self.filteredSentencesTokenAndLabels : list[str] = []

	def __getListLabels(self):
		self.uniqueLabels : list = []
		for sentenceLabels in self.sentencesLabels:
			for label in sentenceLabels:
				if label not in self.uniqueLabels and label.find('I-') == -1:
					self.uniqueLabels.append(label)

		return self.uniqueLabels

	def __getMaskTokensIndex(self, spans, entitiesToMask):
		toMaskIDX = []
		for en in spans:
			for token in en.tokens:
				if token.get_tag('label').value in entitiesToMask:
					toMaskIDX.append(token.idx)
		return toMaskIDX

	def __getSpans(self, spans):
		uniqueLabels = []
		eNsAndAmount, uniqueNGrams = [], []
		fullNGrams, nGramsCount = [], []

		eNs  = [span[0] for span in spans]

		for eN in zip(eNs, spans):
			l = len(eN[0].split(' '))
			fullNGrams.append(l)
			tag = eN[1][1]
			if tag not in uniqueLabels:
				uniqueLabels.append(tag)
			if l not in uniqueNGrams:
				uniqueNGrams.append(l)
			if (eN[0], str(eNs.count(eN[0])), tag) not in eNsAndAmount:
				eNsAndAmount.append((eN[0], str(eNs.count(eN[0])), tag))

		uniqueNGrams.sort()

		for ng in uniqueNGrams:
			nGramsCount.append(str(ng)+'-gram: '+str(fullNGrams.count(ng)))

		return eNsAndAmount, nGramsCount, uniqueLabels

	def __getPossiblesTokens(self, token):
		tokenUniCode = unidecode(token)
		return [token, token.lower(), token.upper(), token.capitalize(), tokenUniCode, \
				tokenUniCode.lower(), tokenUniCode.upper(), tokenUniCode.capitalize()]

	def getUniqueNames(self, rawListNames, listStopNames):
		exhaustiveListStopNames = []

		for sN in listStopNames:
			pSNs = self.__getPossiblesTokens(sN)
			for pSN in pSNs:
				if pSN not in exhaustiveListStopNames:
					exhaustiveListStopNames.append(pSN)

		for rawName in rawListNames:
			tokens = rawName.split(' ')
			for token in tokens:
				if token not in exhaustiveListStopNames:
					if token not in self.uniqueStringNames:
						self.uniqueStringNames.append(token)
					if token.lower() not in self.uniqueStringNames:
						self.uniqueStringNames.append(token.lower())
					if token.upper() not in self.uniqueStringNames:
						self.uniqueStringNames.append(token.upper())

					tokenUniCode = unidecode(token)
					if tokenUniCode not in self.uniqueStringNames:
						self.uniqueStringNames.append(tokenUniCode)
					if tokenUniCode.lower() not in self.uniqueStringNames:
						self.uniqueStringNames.append(tokenUniCode.lower())
					if tokenUniCode.upper() not in self.uniqueStringNames:
						self.uniqueStringNames.append(tokenUniCode.upper())

	def loadCorpusInCoNLLFormat(self,
								inputFilePath,
								setEncoding : str = 'utf-8',
								onlyOneColumnTokens : bool = False,
								sepTokenTag : str = ' '):

		tokensInSentence, tagsInSentence, tokenAndTagInSentence = [], [], []

		if onlyOneColumnTokens == False:
			try:
				sentences: list[str] = open(
					str(inputFilePath), encoding=setEncoding
				).read().strip().split('\n\n')
				
				dataset = [line.split('\n') for line in sentences]
				
				for sentence in dataset:
					for line in sentence:
						token = line.split(sepTokenTag)[0]
						token = token.strip()
						tag = line.split(sepTokenTag)[-1]
						tag = tag.strip()
						if len(token) != 0 and len(tag) != 0:
							tokensInSentence.append(token)
							tagsInSentence.append(tag)
							tokenAndTagInSentence.append(token+sepTokenTag+tag)
					self.sentencesTokens.append(tokensInSentence)
					self.sentencesLabels.append(tagsInSentence)
					self.sentencesTokenAndLabels.append(tokenAndTagInSentence)
					tokensInSentence, tagsInSentence, tokenAndTagInSentence = [], [], []

				print('A dataset with '+str(len(self.sentencesTokenAndLabels))+' sentences was loaded!')
				
				return self.sentencesTokens, self.sentencesLabels, self.sentencesTokenAndLabels
			except:
				raise Exception('"'+inputFilePath+'" is not a valid file name.')

	def loadCorpusInPlainFormat(self, inputFilePath):
		try:
			self.sentencesPlain = []
			sentences: list[str] = open(
				str(inputFilePath), encoding="utf-8"
			).read().strip().split('\n')
			self.sentencesPlain = sentences
			return self.sentencesPlain
			print('A dataset with '+str(len(self.sentencesPlain))+' sentences was loaded!')
		except:
			raise Exception('"'+inputFilePath+'" is not a valid file name.')

	def filterCoNLLCorpusByCategories(self,
									  acceptableLabels,
									  maskForUnacceptLabel,
									  sepTokenTag : str = ' '
									  ):

		auxSentenceLabels : list[str] = []
		auxSentenceTokenAnLabels : list[str] = []
		new_sentencesLabels : list[str] = []
		new_sentencesTokenAndLabels : list[str] = []

		for sentenceLabels in self.sentencesLabels:
			for tag in sentenceLabels:
				if tag.find('B-') != -1:
					newTag = tag
					tag = tag.replace('B-','')

				if tag.find('I-') != -1:
					newTag = tag
					tag = tag.replace('I-','')

				if tag in acceptableLabels:
					auxSentenceLabels.append(newTag)
				else:
					auxSentenceLabels.append(maskForUnacceptLabel)
			new_sentencesLabels.append(auxSentenceLabels)
			auxSentenceLabels = []

		for sentence in self.sentencesTokenAndLabels:
			for tokenTag in sentence:
				token = tokenTag.split(sepTokenTag)[0]
				tag = tokenTag.split(sepTokenTag)[-1]

				if tag.find('B-') != -1:
					newTag = tag
					tag = tag.replace('B-','')

				if tag.find('I-') != -1:
					newTag = tag
					tag = tag.replace('I-','')

				if tag in acceptableLabels:
					auxSentenceTokenAnLabels.append(token+sepTokenTag+newTag)
				else:
					auxSentenceTokenAnLabels.append(token+sepTokenTag+maskForUnacceptLabel)
			new_sentencesTokenAndLabels.append(auxSentenceTokenAnLabels)
			auxSentenceTokenAnLabels = []

		self.sentencesLabels = new_sentencesLabels
		self.sentencesTokenAndLabels = new_sentencesTokenAndLabels

		return self.sentencesLabels, self.sentencesTokenAndLabels

	def filterPlainCorpusByCategory(self, taggedSentence, allPlainLabels, acceptableLabels):
		filteredPlainSentence = []
		unAcceptLabels = list(set(allPlainLabels) - set(acceptableLabels))
		
		for unAcptL in unAcceptLabels:
			newSentence = re.sub(' +', ' ', taggedSentence.replace(unAcptL, '').strip())
		filteredPlainSentence.append(newSentence)

		return filteredPlainSentence

	def sequenceTaggingOnText(self,
							  nerTrainedModelPath,
							  rootFolderPath,
							  fileExtension : str = '.txt',
							  useTokenizer : bool = False,
							  maskNamedEntity : bool = False,
							  createOutputListSpans : bool = False,
							  createOutputFile : bool = False,
							  **kwargs):

		self.maskedSentencesToken = []
		self.maskedSentencesTokenAndLabel = []
		self.taggedFilesDict = {}
		self.namedEntitiesDict = {}
		self.namedEntitiesByFileDict = {}

		allNamedEntitiesInFile = []
		generalNamedEntities = []
		maskedTokenAndLabel = []
		maskedToken = []

		files = [f for f in os.listdir(rootFolderPath) if f.find(fileExtension) != -1]
		tagger = SequenceTagger.load(nerTrainedModelPath)
		
		for file in files:
			sentencesToPredict = self.loadCorpusInPlainFormat(rootFolderPath+'/'+file)
			_auxTaggedPlainSentence = []
			for sentence in sentencesToPredict:
				sentence = sentence.strip()
				sentenceToPred = Sentence(sentence, use_tokenizer=useTokenizer)
				tagger.predict(sentenceToPred)

				sentenceSpans = sentenceToPred.get_spans(label_type='label')

				if maskNamedEntity == True:
					try:
						sepTokenTag = kwargs.get('sepTokenTag')
						entitiesToMask = kwargs.get('entitiesToMask')
						specialTokenToMaskNE = kwargs.get('specialTokenToMaskNE')
					except:
						raise Exception('You need set up these paramters: \
							"sepTokenTag", "entitiesToMask" and "specialTokenToMaskNE"')
					
					_toMaskIDX = self.__getMaskTokensIndex(sentenceSpans, entitiesToMask)

					if kwargs.get('useAuxListNE') == True:
						auxListNE = kwargs.get('auxListNE')
						for token in sentenceToPred.tokens:
							pTokens = self.__getPossiblesTokens(token.text)
							for pT in pTokens:
								if pT in auxListNE:
									if token.idx not in _toMaskIDX:
										_toMaskIDX.append(token.idx)
						_toMaskIDX.sort()

					for token in sentenceToPred.tokens:
						if token.idx in _toMaskIDX:
							if len(maskedTokenAndLabel) > 0:
								if maskedToken[-1] != specialTokenToMaskNE:
									maskedToken.append(specialTokenToMaskNE)
									maskedTokenAndLabel.append(specialTokenToMaskNE+sepTokenTag+token.get_tag('label').value)
							else:
								maskedToken.append(specialTokenToMaskNE)
								maskedTokenAndLabel.append(specialTokenToMaskNE+sepTokenTag+token.get_tag('label').value)
						else:
							maskedToken.append(token.text)
							maskedTokenAndLabel.append(token.text+sepTokenTag+token.get_tag('label').value)
					self.maskedSentencesToken.append(maskedToken)
					self.maskedSentencesTokenAndLabel.append(maskedTokenAndLabel)
					self.maskedPlainSentencesToken = [' '.join(s) for s in self.maskedSentencesToken] #Add tagged sentences
					_toMaskIDX, maskedToken, maskedTokenAndLabel = [], [], []
				else:
					self.unMaskedPlainSentences.append(sentenceToPred.to_tagged_string())

				if createOutputListSpans == True:
					for span in sentenceSpans:
						allNamedEntitiesInFile.append((span.text,span.tag))
						generalNamedEntities.append((span.text,span.tag))

			if createOutputFile == True:
				try:
					outputFilePath = kwargs.get('outputFilePath')
					outFormat = kwargs.get('outputFormat')
				except:
					raise Exception('You need set up these paramters: "outputFilePath" and "outputFormat"')

				if outFormat == 'plain':
					if maskNamedEntity == True:
						self.generateOutputFile(outputFileName = outputFilePath+'/ptTagged-'+str(file),
												sentences = [' '.join(s) for s in self.maskedSentencesToken],
												outputFormat = outFormat)
					else:
						self.generateOutputFile(outputFileName = outputFilePath+'/ptTagged-'+str(file),
												sentences = self.unMaskedPlainSentences,
												outputFormat = outFormat)
				
				if outFormat == 'CoNLL':
					if maskNamedEntity == True:
						self.generateOutputFile(outputFileName = outputFilePath+'/ptTagged-'+str(file),
												sentences = self.maskedSentencesToken,
												outputFormat = outFormat)
					else:
						raise Exception('Por implementar...')

			if maskNamedEntity == True:
				self.taggedFilesDict[str(file)] = self.maskedPlainSentencesToken
			else:
				self.taggedFilesDict[str(file)] = self.unMaskedPlainSentences

			if createOutputListSpans == True:
				fileSpansToOut = []
				nEsAndAmount, nGramsCountByFile, uniqueLabelsByFile = self.__getSpans(allNamedEntitiesInFile)
				self.namedEntitiesByFileDict[str(file)] = nEsAndAmount

				for uL in uniqueLabelsByFile:
					fileSpansToOut.append('CATEGORY:'+str(uL)+'\n')
					for t in nEsAndAmount:
						if t[2] == uL:
							fileSpansToOut.append(': '.join(t[0:2]))
					fileSpansToOut.append('\n')

				fileSpansToOut.append('\n-------\n')

				for nGCG in nGramsCountByFile:
					fileSpansToOut.append(nGCG)

				self.generateOutputFile(outputFileName = outputFilePath+'/NamedEntities-'+str(file),
											sentences = fileSpansToOut,
											outputFormat = 'plain')

			self.maskedSentencesToken, self.maskedPlainSentencesToken = [], []
			self.unMaskedPlainSentences, allNamedEntitiesInFile = [], []
		
		if createOutputListSpans == True:
			generalSpansToOut = []
			generalNEsAndAmount, nGramsCountGeneral, uniqueLabels = self.__getSpans(generalNamedEntities)
			self.namedEntitiesDict['allFiles'] = generalNEsAndAmount

			for uL in uniqueLabels:
				generalSpansToOut.append('CATEGORY:'+str(uL)+'\n')
				for t in generalNEsAndAmount:
					if t[2] == uL:
						generalSpansToOut.append(': '.join(t[0:2]))
				generalSpansToOut.append('\n')

			generalSpansToOut.append('\n-------\n')

			for nGCG in nGramsCountGeneral:
				generalSpansToOut.append(nGCG)

			self.generateOutputFile(outputFileName = outputFilePath+'/GeneralNamedEntities.txt',
											sentences = generalSpansToOut,
											outputFormat = 'plain')

		return self.taggedFilesDict, self.namedEntitiesByFileDict, self.namedEntitiesDict

	def generateOutputFile(self,
						   outputFileName,
						   sentences,
						   outputFormat):
		
		outputFile = open(outputFileName, 'w+', encoding='utf8')

		if outputFormat == 'CoNLL' or outputFormat == 'conll':
			for sentence in sentences:
				for tokenTag in sentence:
					outputFile.write(tokenTag+'\n')
				outputFile.write('\n')
			outputFile.close()
		
		if outputFormat == 'Plain' or outputFormat == 'plain':
			for sentence in sentences:
				outputFile.write(sentence+'\n')
			outputFile.close()
