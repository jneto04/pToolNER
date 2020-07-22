import re
import os
from flair.data import Sentence
from flair.models import SequenceTagger

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

	def __getListLabels(self):
		self.uniqueLabels : list = []
		for sentenceLabels in self.sentencesLabels:
			for label in sentenceLabels:
				if label not in self.uniqueLabels and label.find('I-') == -1:
					self.uniqueLabels.append(label)

		return self.uniqueLabels

	def __clearRepeatedMask(self, tokens):
		newList = []
		finded = False
		for i in range(len(tokens)):
			if tokens[i] != self.specialTokenToMaskNE:
				newList.append(tokens[i])
			else:
				if i == 0:
					newList.append(self.specialTokenToMaskNE)
				if newList[-1] != self.specialTokenToMaskNE:
					newList.append(self.specialTokenToMaskNE)

		return newList

	def __maskNamedEntityOnSentence(self, taggedSentence):
		self.sentenceWithMaskedNE = []
		tokens = taggedSentence.strip().split(' ')
		for i in range(len(tokens)):
			if i+1 < len(tokens):
				if tokens[i+1] not in self.acceptableLabels:
					if tokens[i] not in self.acceptableLabels:
						self.sentenceWithMaskedNE.append(tokens[i])
				if tokens[i+1] in self.acceptableLabels:
						self.sentenceWithMaskedNE.append(self.specialTokenToMaskNE)
			elif tokens[i] not in self.acceptableLabels:
				self.sentenceWithMaskedNE.append(tokens[i])
		self.sentenceWithMaskedNE = self.__clearRepeatedMask(self.sentenceWithMaskedNE)

		return self.sentenceWithMaskedNE

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
							  maskNamedEntity : bool = True,
							  createOutputFile : bool = False,
							  **kwargs):

		self.allPlainLabels = []
		files = [f for f in os.listdir(rootFolderPath) if f.find(fileExtension) != -1]
		tagger = SequenceTagger.load(nerTrainedModelPath)
		
		for file in files:
			sentencesToPredict = self.loadCorpusInPlainFormat(rootFolderPath+'/'+file)
			self.taggedSentences = []
			for sentence in sentencesToPredict:
				sentence = sentence.strip()
				sentenceToPred = Sentence(sentence)
				tagger.predict(sentenceToPred)
				taggedSentence = sentenceToPred.to_tagged_string()
				
				for i in sentenceToPred.get_spans('label'):
					bTag = '<B-'+i.tag+'>'
					iTag = '<I-'+i.tag+'>'
					if bTag not in self.allPlainLabels:
						self.allPlainLabels.append(bTag)
					if iTag not in self.allPlainLabels:
						self.allPlainLabels.append(iTag)

				if maskNamedEntity == True:
					self.acceptableLabels = kwargs.get('acceptableLabels')
					self.specialTokenToMaskNE = kwargs.get('specialTokenToMaskNE')
					maskedSentence = self.__maskNamedEntityOnSentence(taggedSentence)
					self.taggedSentences.append(maskedSentence)
				else:
					self.taggedSentences.append(taggedSentence)
				
			if maskNamedEntity == True:
				newTaggedSentences = []
				newSentence = []
				for sentence in self.taggedSentences:
					for token in sentence:
						if (token.find('<B-') == -1 or token.find('<I-') == -1) and token.find('>') == -1:
							newSentence.append(token)
					newTaggedSentences.append(' '.join(newSentence))
					newSentence = []
				self.taggedSentences : list[str] = []
				self.taggedSentences = newTaggedSentences

			if createOutputFile == True:
				outputFilePath = kwargs.get('outputFilePath')
				self.generateOutputFile(outputFileName = outputFilePath+'/ptTagged-'+str(file),
										sentences = self.taggedSentences,
										outputFormat = kwargs.get('outputFormat'))

			self.taggedPlainTexts.append(self.taggedSentences)

		return self.taggedPlainTexts


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